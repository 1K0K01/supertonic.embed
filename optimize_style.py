"""
optimize_style_v4.py (KAIST 오디오북 연구 기법 통합판)
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass) 특화 Supertonic3 스타일 최적화 코드 — v3 기반 v4

v4 주요 업데이트:
  - Multi-Scale Spectral Loss (MSS Loss) 추가: 주파수 대역 왜곡 및 Vocoder 잔차 제어
  - Style Regularization 추가: style_ttl std 제한(0.06~0.08) 및 초기 레퍼런스 앵커 L2 규제
  - 묵음 인식형 Duration Loss 개선: 단순 Trim 대신 순수 유성음(Speech) 길이 기반 style_dp 학습
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import random
import time
import glob
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf

import onnxslim
import onnx
from onnx import shape_inference
import onnx2torch
from onnx2torch import convert

from helper import load_text_to_speech, load_voice_style

# SSL 인증서 우회
os.environ.pop('SSL_CERT_FILE', None)
os.environ.pop('CURL_CA_BUNDLE', None)
os.environ.pop('REQUESTS_CA_BUNDLE', None)
import httpx
_orig_client = httpx.Client

class _NoVerifyClient(_orig_client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = _NoVerifyClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_SPEECHBRAIN = False

try:
    from speechbrain.inference.speaker import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
        HAS_SPEECHBRAIN = True
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. KAIST 연구 기반 신규 손실 함수 (Multi-Scale Spectral Loss)
# ─────────────────────────────────────────────────────────────────────────────
class SingleScaleSpectralLoss(torch.nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        if x.ndim > 1:
            x = x.squeeze()
        if y.ndim > 1:
            y = y.squeeze()
        
        window = torch.hann_window(self.win_length, device=x.device)
        s_x = torch.stft(x, self.n_fft, self.hop_length, self.win_length, window=window, return_complex=True).abs() + 1e-7
        s_y = torch.stft(y, self.n_fft, self.hop_length, self.win_length, window=window, return_complex=True).abs() + 1e-7
        
        # 주파수 축 및 시간 축 차원이 매칭되지 않는 경우 선형 보간 정렬
        if s_x.shape[-1] != s_y.shape[-1]:
            s_x = F.interpolate(s_x.unsqueeze(0), size=s_y.shape[-1], mode='linear', align_corners=False).squeeze(0)

        # Spectral Convergence & Log Magnitude L1 Loss
        converge_loss = torch.norm(s_y - s_x, p="fro") / (torch.norm(s_y, p="fro") + 1e-7)
        log_mag_loss = F.l1_loss(torch.log10(s_y), torch.log10(s_x))
        return converge_loss + log_mag_loss

class MultiScaleSpectralLoss(torch.nn.Module):
    def __init__(self, scales=[512, 1024, 2048], hops=[120, 240, 480], wins=[600, 1200, 2400]):
        super().__init__()
        self.losses = torch.nn.ModuleList([
            SingleScaleSpectralLoss(n_fft, hop, win) for n_fft, hop, win in zip(scales, hops, wins)
        ])

    def forward(self, x, y):
        return sum(loss_fn(x, y) for loss_fn in self.losses) / len(self.losses)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 모델 로딩 유틸리티 (ONNX 변환 파트)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_onnx2torch():
    def patched(m):
        if isinstance(m, str):
            m = onnx.load(m)
        try:
            return shape_inference.infer_shapes(m)
        except:
            return m
    onnx2torch.converter.safe_shape_inference = patched

def _fix_clip(model):
    for node in model.graph.node:
        if node.op_type == 'Clip':
            inputs = list(node.input)
            while inputs and inputs[-1] == '':
                inputs.pop()
            del node.input[:]
            node.input.extend(inputs)
    return model

def load_pt_model(name, onnx_dir="onnx"):
    slimmed = onnxslim.slim(os.path.join(onnx_dir, name))
    for opset in slimmed.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            opset.version = 17
    _fix_clip(slimmed)
    m = convert(slimmed)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 한국어 발음 교정 및 저음 특화 텍스트 세트
# ─────────────────────────────────────────────────────────────────────────────
KO_PHONETIC_BASS_TEXTS = [
    '어두운 복도 끝에서 그가 천천히 발걸음을 멈추었다. 아무런 말도 없이, 그저 먼 곳을 응시하는 그의 눈빛에는 오래된 무게가 가라앉아 있었다.',
    '그는 잠시 눈을 감았다가 천천히 뜨며, 낮고 묵직한 목소리로 한 마디를 내뱉었다. "그건 내가 결정할 일이오."',
    '어제도 그랬고, 오늘도 그럴 것이다. 말하지 않아도 알 수 있는 것들이 있고, 굳이 묻지 않아도 되는 것들이 있다.',
    '차갑게 식어버린 철창 너머로, 짙은 잿빛 구름이 잔잔하게 흩어지고 있었다. 진실은 늘 찰나의 순간에 자취를 감춘다.',
    '맑고 쾌청한 가을 하늘 아래, 붉은 단풍잎이 흙바닥 위로 넓게 깔려 있었다. 삶은 닭고기를 썰어 넣은 국물은 끓일수록 깊은 맛을 냈다.',
    '"그 서류를 내려놓으시죠." 그가 서늘한 시선으로 내 손끝을 응시하며 덧붙였다. "그 너머의 심연을 감당할 수 없다면, 여기서 멈추는 게 좋습니다."',
    '"착각하지 마." 낮게 깔린 목소리가 좁은 골목길의 적막을 파고들었다. "내가 당신의 얄팍한 거짓말에 속아준 건, 자비가 아니라 단지 필요에 의해서일 뿐이야."',
    '"결국 이렇게 되는군." 씁쓸한 미소와 함께 그가 코트 주머니에 깊숙이 손을 찔러 넣었다. "알면서도 모른 척 걸어온 대가치고는, 꽤나 혹독한 밤이야."',
    '부서진 잔해들 사이로 스며드는 옅은 달빛. 우리는 그 희미한 궤적을 밟으며 묵묵히 걸었다. 잃어버린 것들을 애도하기엔, 밤이 너무 짧았으므로.',
    '바람이 스치고 간 자리마다 붉은 녹이 슬었다. 시간은 누구에게나 공평하게 잔혹하고, 기억은 낡은 침전물처럼 바닥에 가라앉아 단단히 굳어갈 뿐이다.'
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. 특징 추출 및 매칭
# ─────────────────────────────────────────────────────────────────────────────
WAVLM_LAYERS        = (1, 3, 6, 9)
WAVLM_LAYER_WEIGHTS = (0.4, 1.0, 0.5, 0.3)
SEQ_LAYER           = 6   

def auto_select_preset_by_texture(wavlm, tts, target_feats_avg):
    print("[*] 4중 질감 레이어(WavLM) 기반 베이스 프리셋 최적화 검색 시작...")

    preset_paths = sorted(glob.glob("voice_styles/[M]*.json"))
    if not preset_paths:
        print("[경고] voice_styles 폴더 내 프리셋을 찾을 수 없어 M2.json으로 폴백합니다.")
        return "voice_styles/M2.json"

    compare_text = "어두운 복도 끝에서 그가 천천히 발걸음을 멈추었다."
    results = []

    for path in preset_paths:
        style = load_voice_style(path)
        wav_np, sr = tts(compare_text, "ko", style)
        wav_t = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)

        with torch.no_grad():
            out = wavlm(wav_t, output_hidden_states=True)

        dist = 0.0
        for layer, weight in zip(WAVLM_LAYERS, WAVLM_LAYER_WEIGHTS):
            gen_feat = out.hidden_states[layer]
            tgt_mean, tgt_std = target_feats_avg[layer]
            layer_dist = (F.mse_loss(gen_feat.mean(dim=1), tgt_mean)
                          + F.mse_loss(gen_feat.std(dim=1), tgt_std))
            dist += (weight * layer_dist).item()

        results.append((path, dist))

    results.sort(key=lambda x: x[1])
    best_preset = results[0][0]

    print(f"  ▶ 질감 매칭 1위 프리셋: {os.path.basename(best_preset)} (최단 거리 점수: {results[0][1]:.4f})")
    return best_preset


def load_wavlm():
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _extract_wavlm_single(wavlm, target_wav, layers=WAVLM_LAYERS, keep_seq_layer=None):
    if target_wav.ndim == 1:
        target_wav = target_wav.unsqueeze(0)
    wav_16k = torchaudio.functional.resample(target_wav, 44100, 16000)
    with torch.no_grad():
        out = wavlm(wav_16k, output_hidden_states=True)
        feats = {}
        for layer in layers:
            feat = out.hidden_states[layer]
            feats[layer] = (feat.mean(dim=1), feat.std(dim=1))
        seq = None
        if keep_seq_layer is not None:
            seq = out.hidden_states[keep_seq_layer].detach()
        return feats, seq

def extract_wavlm_targets_per_wav(wavlm, target_wavs, layers=WAVLM_LAYERS, keep_seq_layer=None):
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    feats_list, seq_list = [], []
    for w in target_wavs:
        feats, seq = _extract_wavlm_single(wavlm, w, layers, keep_seq_layer)
        feats_list.append(feats)
        seq_list.append(seq)
    return feats_list, seq_list

def average_wavlm_features(feats_list, layers=WAVLM_LAYERS):
    avg_feats = {}
    for layer in layers:
        means = torch.stack([f[layer][0] for f in feats_list], dim=0).mean(dim=0)
        stds  = torch.stack([f[layer][1] for f in feats_list], dim=0).mean(dim=0)
        avg_feats[layer] = (means, stds)
    return avg_feats

def load_ecapa():
    if not HAS_SPEECHBRAIN:
        return None
    import logging
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": str(DEVICE)}
        )
    except Exception:
        return None
    for p in classifier.parameters():
        p.requires_grad_(False)
    return classifier

def _extract_ecapa_single(ecapa, target_wav_t):
    wav_16k = torchaudio.functional.resample(target_wav_t, 44100, 16000)
    if wav_16k.ndim == 1:
        wav_16k = wav_16k.unsqueeze(0)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav_16k)
        emb = F.normalize(emb, dim=2).squeeze(1)
    return emb

def extract_ecapa_targets_per_wav(ecapa, target_wavs):
    if ecapa is None:
        return None
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    return [_extract_ecapa_single(ecapa, w) for w in target_wavs]

def ecapa_loss_fn(ecapa, gen_wav, target_emb):
    if ecapa is None or target_emb is None:
        return 0.0
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_emb = ecapa.encode_batch(gen_wav_16k)
    gen_emb = F.normalize(gen_emb, dim=2).squeeze(1)
    cos_sim = F.cosine_similarity(gen_emb, target_emb)
    return 1.0 - cos_sim.mean()


# ── [LTAS (장기 평균 스펙트럼) 손실] ─────────────────────────────────────────
N_LTAS_BANDS = 24
_LTAS_EDGES  = None

def _ltas_band_edges(device):
    global _LTAS_EDGES
    if _LTAS_EDGES is None:
        _LTAS_EDGES = torch.logspace(
            np.log10(80.0), np.log10(11000.0), N_LTAS_BANDS + 1
        ).to(device)
    return _LTAS_EDGES.to(device)

def compute_ltas_distribution(wav, sr=44100):
    if wav.ndim > 1:
        wav = wav.squeeze()
    spec  = torch.fft.rfft(wav)
    power = (torch.abs(spec) ** 2)
    freqs = torch.fft.rfftfreq(wav.shape[-1], d=1.0 / sr).to(wav.device)
    edges = _ltas_band_edges(wav.device)
    bands = []
    for i in range(N_LTAS_BANDS):
        mask = ((freqs >= edges[i]) & (freqs < edges[i + 1])).float()
        bands.append((power * mask).sum())
    dist = torch.stack(bands)
    return dist / (dist.sum() + 1e-10)

def ltas_loss_fn(gen_wav, target_ltas):
    gen_ltas = compute_ltas_distribution(gen_wav)
    return F.l1_loss(torch.log10(gen_ltas + 1e-8),
                     torch.log10(target_ltas + 1e-8))


def wavlm_primary_loss(wavlm, gen_wav, target_features, layer=3):
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features[layer]
    return (F.mse_loss(gen_feat.mean(dim=1), tgt_mean)
            + F.mse_loss(gen_feat.std(dim=1), tgt_std))


def wavlm_hybrid_feature_loss(wavlm, gen_wav, target_features,
                               hf_weight=0.05,
                               layers=WAVLM_LAYERS, weights=WAVLM_LAYER_WEIGHTS,
                               target_seq=None, seq_loss_weight=0.0):
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)

    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)

    wavlm_loss = 0.0
    for layer, weight in zip(layers, weights):
        gen_feat = gen_out.hidden_states[layer]
        tgt_mean, tgt_std = target_features[layer]
        layer_loss = (F.mse_loss(gen_feat.mean(dim=1), tgt_mean)
                      + F.mse_loss(gen_feat.std(dim=1), tgt_std))
        wavlm_loss = wavlm_loss + weight * layer_loss

    total_loss = wavlm_loss
    if hf_weight > 0:
        gen_fft     = torch.fft.rfft(gen_wav)
        gen_fft_mag = torch.abs(gen_fft)
        freqs       = torch.fft.rfftfreq(gen_wav.shape[-1], d=1/44100.0).to(gen_wav.device)
        hf_mask     = (freqs > 4000.0).float()
        rfft_loss   = (gen_fft_mag * hf_mask).abs().mean()
        total_loss  = total_loss + hf_weight * rfft_loss

    if target_seq is not None and seq_loss_weight > 0:
        gen_seq = gen_out.hidden_states[SEQ_LAYER]
        T_ALIGN = 64
        g = F.adaptive_avg_pool1d(gen_seq.transpose(1, 2), T_ALIGN)
        t = F.adaptive_avg_pool1d(target_seq.transpose(1, 2), T_ALIGN)
        seq_loss   = F.mse_loss(g, t)
        total_loss = total_loss + seq_loss_weight * seq_loss

    return total_loss, gen_out


def wavlm_primary_loss_from_out(gen_out, target_features, layer=3):
    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features[layer]
    return (F.mse_loss(gen_feat.mean(dim=1), tgt_mean)
            + F.mse_loss(gen_feat.std(dim=1), tgt_std)).item()


# ─────────────────────────────────────────────────────────────────────────────
# 5. TTS forward / 파일 저장
# ─────────────────────────────────────────────────────────────────────────────
def tts_forward(text_ids, text_mask, style_ttl, style_dp,
                dp_model, te_model, ve_model, voc_model,
                total_step, speed, noisy_latent, latent_mask):
    dur      = dp_model(text_ids, style_dp, text_mask) / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt       = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask,
                      current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur

def save_style(path, style_ttl, style_dp, source_file=None):
    source_meta = (list(source_file)
                   if isinstance(source_file, (list, tuple))
                   else (source_file or "unknown"))
    style_json = {
        "style_ttl": {
            "data": style_ttl.detach().cpu().numpy().tolist(),
            "dims": [1, 50, 256],
            "type": "float32"
        },
        "style_dp": {
            "data": style_dp.detach().cpu().numpy().tolist(),
            "dims": [1, 8, 16],
            "type": "float32"
        },
        "metadata": {
            "source_file":        source_meta,
            "source_sample_rate": 44100,
            "target_sample_rate": 44100,
            "extracted_at":       datetime.now().isoformat()
        }
    }
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(style_json, f, ensure_ascii=False, indent=2)


def find_latest_checkpoint(log_dir, name):
    pattern = os.path.join(log_dir, f"{name}_*.json")
    files   = [f for f in glob.glob(pattern)
               if "train_config" not in f and "_final" not in f]
    if not files:
        return None, 0
    def _step(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        except ValueError:
            return 0
    latest   = max(files, key=_step)
    step_num = _step(latest)
    return latest, step_num


def build_per_text_latents(text_inputs, ref_style, dp_model, tts, speed, seed):
    rng = np.random.RandomState(seed)
    latents, masks = [], []
    ref_dp_t = torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for ids, mask in text_inputs:
            dur = (dp_model(ids, ref_dp_t, mask) / speed)
            latent_len = int(np.ceil(
                (dur.item() * 44100)
                / (tts.base_chunk_size * tts.chunk_compress_factor)
            ))
            latent_len = max(latent_len, 4)
            noise = torch.tensor(
                rng.randn(1, tts.ldim * tts.chunk_compress_factor, latent_len)
                .astype(np.float32)
            ).to(DEVICE)
            latents.append(noise)
            masks.append(torch.ones(1, 1, latent_len,
                                    dtype=torch.float32).to(DEVICE))
    return latents, masks


# ─────────────────────────────────────────────────────────────────────────────
# 8. 메인 최적화 루프
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _patch_onnx2torch()

    arg = sys.argv[1] if len(sys.argv) > 1 else "configs/caelus.json"

    if os.path.exists(arg):
        config_path = arg
    elif os.path.exists(f"configs/{arg}.json"):
        config_path = f"configs/{arg}.json"
    elif os.path.exists(f"configs/{arg}"):
        config_path = f"configs/{arg}"
    else:
        config_path = f"configs/{arg}.json"

    print(f"Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name             = cfg["name"]
    target_wav_paths = (list(cfg["target_wavs"]) if "target_wavs" in cfg
                        else [cfg["target_wav"]])
    multi_wav_mode   = cfg.get("multi_wav_mode", "stochastic")
    if multi_wav_mode == "rotate":
        multi_wav_mode = "stochastic"

    seed          = cfg.get("seed", 42)
    lr            = cfg.get("lr", 2e-4)
    dp_lr_ratio   = cfg.get("dp_lr_ratio", 0.008)
    train_dp      = cfg.get("train_style_dp", True)
    num_steps     = cfg.get("num_steps", 3000)
    total_step    = cfg.get("total_step", 5)
    speed         = cfg.get("speed", 0.95)
    save_every    = cfg.get("save_every", 100)
    threshold     = cfg.get("early_stop_loss_threshold", 0.18)
    use_ecapa     = cfg.get("use_ecapa_loss", True)
    ecapa_weight  = cfg.get("ecapa_loss_weight", 0.3)

    hf_weight       = cfg.get("hf_weight", 0.05)
    ltas_weight     = cfg.get("ltas_weight", 0.5)
    seq_loss_weight = cfg.get("seq_loss_weight", 0.3)
    dur_loss_weight = cfg.get("dur_loss_weight", 0.3)
    paired_ratio    = cfg.get("paired_ratio", 0.7)

    # [v4 KAIST 적용] 신규 손실 제어 파라미터 로딩
    mss_loss_weight  = cfg.get("mss_loss_weight", 0.20)
    style_reg_weight = cfg.get("style_reg_weight", 0.05)

    target_texts = cfg.get("target_texts", None)
    paired_mode  = bool(target_texts)
    if paired_mode and len(target_texts) != len(target_wav_paths):
        print(f"[경고] target_texts({len(target_texts)}) ≠ target_wavs({len(target_wav_paths)}) — 페어드 모드 비활성화")
        paired_mode = False

    if train_dp and not paired_mode:
        print("\n[알림] target_texts 미제공 → style_dp에 gradient가 흐르지 않아")
        print("        train_style_dp를 자동 비활성화합니다. (레퍼런스 dp 유지)")
        print("        style_dp를 학습하려면 config에 target_texts를 추가하세요.")
        train_dp = False

    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)

    latest_ckpt, start_step = find_latest_checkpoint(log_dir, name)
    if latest_ckpt:
        print(f"\n[Resume] 기존 체크포인트 발견: {latest_ckpt} (step {start_step})")
        print("[Resume] 해당 스텝부터 이어서 학습합니다.")
    else:
        print("\n[신규 학습] 체크포인트 없음. 처음부터 시작합니다.")

    # ── 타겟 WAV 로드 및 유성음 추출 (VAD 기반 묵음 인식) ────────────────────
    print(f"\nDevice: {DEVICE}")
    print(f"Name  : {name}")
    print(f"Mode  : {'PAIRED(대사 페어링)' if paired_mode else 'UNPAIRED(일반)'}")
    print(f"\nLoading target WAV(s) ({len(target_wav_paths)}개): {target_wav_paths}")
    target_wav_ts   = []
    target_dur_secs = []   
    for p in target_wav_paths:
        w, _ = librosa.load(p, sr=44100)
        target_wav_ts.append(torch.tensor(w, dtype=torch.float32).to(DEVICE))
        
        # [v4 KAIST 적용]: 단순 Trim 대신 대화 중간의 모든 무음부를 제외한 순수 발화 길이를 타겟으로 계산
        intervals = librosa.effects.split(w, top_db=30)
        pure_speech_dur = sum((e - s) for s, e in intervals) / 44100.0
        target_dur_secs.append(pure_speech_dur)
        print(f"  - {p}: 총 {len(w)/44100:.2f}s (순수 발화 구간 합: {pure_speech_dur:.2f}s)")

    # ── WavLM 로드 및 타겟 피처 추출 ─────────────────────────────────────────
    wavlm = load_wavlm()
    target_feats_list, target_seq_list = extract_wavlm_targets_per_wav(
        wavlm, target_wav_ts,
        keep_seq_layer=(SEQ_LAYER if paired_mode else None)
    )
    target_feats_avg = average_wavlm_features(target_feats_list)

    # [LTAS 분포 구축]
    target_ltas_list = [compute_ltas_distribution(w).detach() for w in target_wav_ts]
    target_ltas_avg  = torch.stack(target_ltas_list, dim=0).mean(dim=0)
    target_ltas_avg  = target_ltas_avg / target_ltas_avg.sum()

    # [v4 KAIST 적용]: STFT Multi-Scale Spectral Loss 함수 인스턴스화
    if mss_loss_weight > 0:
        mss_loss_fn = MultiScaleSpectralLoss().to(DEVICE)

    # ── TTS 엔진 로드 ─────────────────────────────────────────────────────────
    tts = load_text_to_speech("onnx")

    # ── 레퍼런스 프리셋 결정 ──────────────────────────────────────────────────
    ref_style_path = cfg.get("reference_style", "auto")
    if ref_style_path == "auto":
        ref_style_path = auto_select_preset_by_texture(wavlm, tts, target_feats_avg)

    # ── ECAPA ────────────────────────────────────────────────────────────────
    ecapa                 = None
    target_ecapa_emb_list = None
    target_ecapa_emb_avg  = None
    if use_ecapa:
        ecapa = load_ecapa()
        if ecapa is not None:
            target_ecapa_emb_list = extract_ecapa_targets_per_wav(ecapa, target_wav_ts)
            target_ecapa_emb_avg = F.normalize(
                torch.stack(target_ecapa_emb_list, dim=0).mean(dim=0), dim=1
            )

    # ── ONNX → PyTorch 모델 변환 ─────────────────────────────────────────────
    dp_model  = load_pt_model("duration_predictor.onnx")
    te_model  = load_pt_model("text_encoder.onnx")
    ve_model  = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")

    # ── 텍스트 전처리 ────────────────────────────────────────────────────────
    generic_text_inputs = []
    for text in KO_PHONETIC_BASS_TEXTS:
        ids_np, mask_np = tts.text_processor(text, "ko")
        generic_text_inputs.append((
            torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))
    paired_text_inputs = []
    if paired_mode:
        for text in target_texts:
            ids_np, mask_np = tts.text_processor(text, "ko")
            paired_text_inputs.append((
                torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
                torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
            ))

    # ── Seed 고정 ────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n[레퍼런스 베이스 스타일 설정] -> {os.path.basename(ref_style_path)}")
    ref_style = load_voice_style(ref_style_path)

    # [v4 KAIST 적용]: 초기 스타일 기준 앵커 규제용 영구 고정 복사본 생성
    ref_style_ttl_frozen = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone()
    ref_style_ttl_frozen.requires_grad_(False)

    print("[*] 텍스트별 고정 latent 생성 중... (FIX-C)")
    gen_latents, gen_masks = build_per_text_latents(
        generic_text_inputs, ref_style, dp_model, tts, speed, seed)
    if paired_mode:
        pr_latents, pr_masks = [], []
        rng = np.random.RandomState(seed + 1)
        for (ids, mask), tgt_dur in zip(paired_text_inputs, target_dur_secs):
            latent_len = int(np.ceil(
                (tgt_dur * 44100)
                / (tts.base_chunk_size * tts.chunk_compress_factor)
            ))
            latent_len = max(latent_len, 4)
            noise = torch.tensor(
                rng.randn(1, tts.ldim * tts.chunk_compress_factor, latent_len)
                .astype(np.float32)).to(DEVICE)
            pr_latents.append(noise)
            pr_masks.append(torch.ones(1, 1, latent_len,
                                       dtype=torch.float32).to(DEVICE))

    # ── 스타일 벡터 초기화 ───────────────────────────────────────────────────
    if latest_ckpt:
        resumed = load_voice_style(latest_ckpt)
        style_ttl = torch.tensor(resumed.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(resumed.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
        print(f"[Resume] style 벡터를 체크포인트에서 복원했습니다.")
    else:
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    if train_dp:
        optimizer = torch.optim.Adam([
            {"params": [style_ttl], "lr": lr},
            {"params": [style_dp],  "lr": lr * dp_lr_ratio}
        ])
    else:
        optimizer = torch.optim.Adam([style_ttl], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01,
        last_epoch=(start_step - 1) if start_step > 0 else -1
    )

    best_loss      = float('inf')
    best_ttl       = None
    best_dp        = style_dp.detach().clone()
    warmdown_mode  = False
    warmdown_steps = 0
    MAX_WARMDOWN   = 50

    initial_gap = None
    start_time  = time.time()

    print(f"\n[Dynamic Optimization 구동] 목표 임계치: {threshold}")
    print(f"  speed={speed} | dp_lr_ratio={dp_lr_ratio} | multi_wav_mode={multi_wav_mode}")
    print(f"  hf_weight={hf_weight} | ltas_weight={ltas_weight} | mss_weight={mss_loss_weight} | style_reg={style_reg_weight}")
    print(f"  학습 구간: step {start_step + 1} → {num_steps}\n")

    for step in range(start_step, num_steps):
        optimizer.zero_grad()

        if train_dp and not style_dp.requires_grad:
            if best_loss <= 0.30 or step >= start_step + 50:
                style_dp.requires_grad_(True)
                print(f"\n>>> [DP 활성화] Step {step+1}: L3={best_loss:.4f} → style_dp 학습 시작")

        # ── 샘플 선택 ────────────────────────────────────────────────────────
        use_paired = (paired_mode and random.random() < paired_ratio)

        if use_paired:
            pi = random.randrange(len(paired_text_inputs))
            text_ids, text_mask   = paired_text_inputs[pi]
            noisy_latent, l_mask  = pr_latents[pi], pr_masks[pi]
            current_target_feats  = target_feats_list[pi]
            current_target_seq    = target_seq_list[pi]
            current_target_ltas   = target_ltas_list[pi]
            current_dur_target    = target_dur_secs[pi]
            if use_ecapa and ecapa is not None and target_ecapa_emb_list is not None:
                current_target_ecapa = target_ecapa_emb_list[pi]
            else:
                current_target_ecapa = None
        else:
            text_idx = step % len(generic_text_inputs)
            text_ids, text_mask  = generic_text_inputs[text_idx]
            noisy_latent, l_mask = gen_latents[text_idx], gen_masks[text_idx]
            current_target_seq   = None
            current_dur_target   = None
            current_target_ltas  = target_ltas_avg
            if multi_wav_mode == "stochastic" and len(target_feats_list) > 1:
                k             = min(2, len(target_feats_list))
                batch_indices = random.sample(range(len(target_feats_list)), k)
                current_target_feats = average_wavlm_features(
                    [target_feats_list[i] for i in batch_indices]
                )
                if use_ecapa and ecapa is not None and target_ecapa_emb_list is not None:
                    current_target_ecapa = F.normalize(
                        torch.stack([target_ecapa_emb_list[i] for i in batch_indices], dim=0)
                        .mean(dim=0), dim=1
                    )
                else:
                    current_target_ecapa = None
            else:
                current_target_feats = target_feats_avg
                current_target_ecapa = target_ecapa_emb_avg

        # ── Forward & Loss ───────────────────────────────────────────────────
        wav_out, dur = tts_forward(
            text_ids, text_mask, style_ttl, style_dp,
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent, l_mask
        )
        gen_wav = wav_out.squeeze()

        loss, gen_out = wavlm_hybrid_feature_loss(
            wavlm, gen_wav, current_target_feats,
            hf_weight=hf_weight,
            target_seq=current_target_seq,
            seq_loss_weight=seq_loss_weight
        )

        if ltas_weight > 0:
            loss = loss + ltas_weight * ltas_loss_fn(gen_wav, current_target_ltas)

        # [v4 KAIST 적용]: Multi-Scale Spectral Loss 추가 (페어드 매칭 시 직접 파형 비교 가능)
        if use_paired and mss_loss_weight > 0:
            loss = loss + mss_loss_weight * mss_loss_fn(gen_wav, target_wav_ts[pi])

        # [v4 KAIST 적용]: Style Regularization (Std 경계 및 레퍼런스 앵커 L2 거리 보존)
        if style_reg_weight > 0:
            current_std = style_ttl.std()
            std_penalty = 0.0
            if current_std < 0.06:
                std_penalty = (0.06 - current_std) ** 2
            elif current_std > 0.08:
                std_penalty = (current_std - 0.08) ** 2
            
            ref_dist_loss = F.mse_loss(style_ttl, ref_style_ttl_frozen)
            # 가중합산 규제 추가
            loss = loss + style_reg_weight * ((1.0 * std_penalty) + (0.5 * ref_dist_loss))

        if use_paired and train_dp and style_dp.requires_grad and dur_loss_weight > 0:
            dur_natural = dur.squeeze() * speed
            dur_loss    = ((dur_natural - current_dur_target)
                           / max(current_dur_target, 1e-3)) ** 2
            loss = loss + dur_loss_weight * dur_loss.mean()

        if use_ecapa and ecapa is not None and current_target_ecapa is not None:
            loss = loss + (ecapa_weight * ecapa_loss_fn(ecapa, gen_wav, current_target_ecapa))

        with torch.no_grad():
            primary_loss = wavlm_primary_loss_from_out(gen_out, current_target_feats)

        # ── Backward ─────────────────────────────────────────────────────────
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in [style_ttl, style_dp] if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()

        if not warmdown_mode:
            scheduler.step()

        if (not use_paired) and primary_loss < best_loss:
            best_loss = primary_loss
            best_ttl  = style_ttl.detach().clone()
            best_dp   = style_dp.detach().clone()
        elif use_paired and best_ttl is None:
            best_ttl = style_ttl.detach().clone()
            best_dp  = style_dp.detach().clone()

        # ── 로그 출력 ────────────────────────────────────────────────────────
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step+1}/{num_steps} | Loss(total): {loss.item():.4f} | "
                  f"Loss(L3): {primary_loss:.4f} | LR: {current_lr:.5f} | "
                  f"Best(L3): {best_loss:.4f} | TTL_Std: {style_ttl.std().item():.4f}")

            if initial_gap is None and best_loss < float('inf'):
                initial_gap = max(0.001, best_loss - threshold)
            if initial_gap is not None:
                current_gap = max(0.0, best_loss - threshold)
                percentage  = (max(0.0, (1.0 - (current_gap / initial_gap)) * 100.0)
                               if current_gap > 0 else 100.0)
                num_blocks  = min(20, max(0, int(percentage / 5)))
                bar_str     = "█" * num_blocks + "-" * (20 - num_blocks)
                print(f"    [{bar_str}]  {percentage:.1f}%  "
                      f"(best {best_loss:.4f} -> 목표 {threshold:.4f}, gap +{current_gap:.4f})")

        # ── 체크포인트 저장 ───────────────────────────────────────────────────
        if (step + 1) % save_every == 0 and best_ttl is not None:
            ckpt_path = f"{log_dir}/{name}_{step+1:04d}.json"
            save_style(ckpt_path, best_ttl, best_dp, target_wav_paths)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        # ── Warmdown ─────────────────────────────────────────────────────────
        if not warmdown_mode and best_loss <= threshold:
            print(f"\n>>> [Phase Transition] Best L3 Loss Threshold({threshold}) 도달!")
            print(">>> Dynamic Warm-down 모드 활성화 (LR x0.05, 최대 50스텝 추가 정렬)")
            warmdown_mode = True
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.05

        if warmdown_mode:
            warmdown_steps += 1
            if warmdown_steps >= MAX_WARMDOWN:
                print(f">>> [Early Stop] {MAX_WARMDOWN}스텝 미세 정렬 완료. 종료합니다.")
                break

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    final_path = f"{log_dir}/{name}_final.json"
    save_style(final_path, best_ttl if best_ttl is not None else style_ttl,
               best_dp, target_wav_paths)

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("[학습 완료 결과 다차원 평가 리포트]")
    print(f"- 총 학습 소요 시간  : {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"- 최종 학습 진행 스텝: {step + 1} / {num_steps}")
    print(f"- 최소 음색 손실값   : {best_loss:.4f}")
    print(f"- 학습 모드          : {'PAIRED' if paired_mode else 'UNPAIRED'}"
          f" | hf_weight={hf_weight} | ltas_weight={ltas_weight}")
    print("-" * 60)
    if warmdown_mode:
        print("▶ 최종 판정: [✨ 웜다운 방어 수렴 성공]")
    else:
        print("▶ 최종 판정: [✅ 만기 완주 수렴]")
    print("=" * 60)


if __name__ == "__main__":
    main()
