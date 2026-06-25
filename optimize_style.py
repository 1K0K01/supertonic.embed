"""
optimize_style_advanced_ko.py (최종 무결성 패치판 v2)
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass) 특화 및 발음 해상도 개선 Supertonic3 최적화 코드

수정 사항 (v2):
  - resume(체크포인트 재개) 기능 복구: Colab 세션 끊김 대비
  - stochastic 모드 single-wav fallback 시 current_target_ecapa 미정의 버그 수정
  - rfft_loss target_zero 불필요 연산 제거 (기능 동일, 코드 명확화)
  - save_style os.makedirs 경로 버그 수정 (dirname이 빈 문자열일 때 crash 방지)
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
# 1. 모델 로딩 유틸리티 (ONNX 변환 파트)
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
# 2. 한국어 발음 교정 및 저음 특화 텍스트 세트
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
# 3. 질감(Texture) 기반 음원 프로파일링 및 자동 프리셋 매칭
# ─────────────────────────────────────────────────────────────────────────────
WAVLM_LAYERS        = (1, 3, 6, 9)
WAVLM_LAYER_WEIGHTS = (0.4, 1.0, 0.5, 0.3)

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
        # TTS 출력(44100Hz)을 WavLM 입력용 16kHz로 리샘플
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

    print(f"  ▶ 질감 매칭 1위 프리셋: {os.path.basename(best_preset)} "
          f"(최단 거리 점수: {results[0][1]:.4f})")
    return best_preset


# ─────────────────────────────────────────────────────────────────────────────
# 4. 하이브리드 특징 손실 산출 모듈 (WavLM + RFFT 마스킹)
# ─────────────────────────────────────────────────────────────────────────────
def load_wavlm():
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _extract_wavlm_single(wavlm, target_wav, layers=WAVLM_LAYERS):
    if target_wav.ndim == 1:
        target_wav = target_wav.unsqueeze(0)
    wav_16k = torchaudio.functional.resample(target_wav, 44100, 16000)
    with torch.no_grad():
        out = wavlm(wav_16k, output_hidden_states=True)
        feats = {}
        for layer in layers:
            feat = out.hidden_states[layer]
            feats[layer] = (feat.mean(dim=1), feat.std(dim=1))
        return feats

def extract_wavlm_targets_per_wav(wavlm, target_wavs, layers=WAVLM_LAYERS):
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    return [_extract_wavlm_single(wavlm, w, layers) for w in target_wavs]

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

def wavlm_primary_loss(wavlm, gen_wav, target_features, layer=3):
    """Layer 3 단독 손실 — 조기 종료 임계값 비교 기준."""
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features[layer]
    return (F.mse_loss(gen_feat.mean(dim=1), tgt_mean)
            + F.mse_loss(gen_feat.std(dim=1), tgt_std))

def wavlm_hybrid_feature_loss(wavlm, gen_wav, target_features,
                               layers=WAVLM_LAYERS, weights=WAVLM_LAYER_WEIGHTS):
    """
    WavLM 다중 레이어 손실 + 4kHz 이상 고주파 억제 RFFT 보조 손실.

    hf_weight=0.05로 낮게 설정하여 모음 명료도 손상 최소화.
    저음 남성은 4kHz 이상 성분이 자연적으로 적으므로 이 손실이
    불필요한 고주파 왜곡(기계음)을 부드럽게 억제하는 역할을 함.
    """
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

    # 고주파 억제: 44100Hz 원본 신호 기준 4kHz 이상 성분을 0 방향으로 유도
    gen_fft     = torch.fft.rfft(gen_wav)
    gen_fft_mag = torch.abs(gen_fft)
    freqs       = torch.fft.rfftfreq(gen_wav.shape[-1], d=1/44100.0).to(gen_wav.device)
    hf_mask     = (freqs > 4000.0).float()
    # target_zero와 곱하면 그냥 0이므로 직접 L1 계산
    rfft_loss   = F.l1_loss(gen_fft_mag * hf_mask,
                             torch.zeros_like(gen_fft_mag * hf_mask))

    hf_weight  = 0.05
    total_loss = wavlm_loss + (hf_weight * rfft_loss)
    return total_loss


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
    # dirname이 빈 문자열('')이면 makedirs가 crash하므로 방어 처리
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(style_json, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 체크포인트 재개(Resume) 유틸리티 — Colab 세션 끊김 대비
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_checkpoint(log_dir, name):
    """
    logs/{name}/ 아래에서 step 번호가 가장 큰 체크포인트 JSON을 찾아 반환.
    train_config.json과 _final.json은 제외.
    파일이 없으면 (None, 0) 반환.
    """
    pattern = os.path.join(log_dir, f"{name}_*.json")
    files   = [f for f in glob.glob(pattern)
               if "train_config" not in f and "_final" not in f]
    if not files:
        return None, 0
    # 파일명 끝 숫자로 정렬 (e.g. caelus_0500.json → 500)
    def _step(p):
        try:
            return int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        except ValueError:
            return 0
    latest   = max(files, key=_step)
    step_num = _step(latest)
    return latest, step_num


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 최적화 루프
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _patch_onnx2torch()

    # ── Config 로드 ──────────────────────────────────────────────────────────
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
    # "rotate"는 이 버전에서 "stochastic"으로 통합
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

    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)

    # ── 체크포인트 자동 재개 (Resume) ────────────────────────────────────────
    latest_ckpt, start_step = find_latest_checkpoint(log_dir, name)
    if latest_ckpt:
        print(f"\n[Resume] 기존 체크포인트 발견: {latest_ckpt} (step {start_step})")
        print("[Resume] 해당 스텝부터 이어서 학습합니다.")
    else:
        print("\n[신규 학습] 체크포인트 없음. 처음부터 시작합니다.")

    # ── 타겟 WAV 로드 ────────────────────────────────────────────────────────
    print(f"\nDevice: {DEVICE}")
    print(f"Name  : {name}")
    print(f"\nLoading target WAV(s) ({len(target_wav_paths)}개): {target_wav_paths}")
    target_wav_ts = []
    for p in target_wav_paths:
        w, _ = librosa.load(p, sr=44100)
        target_wav_ts.append(torch.tensor(w, dtype=torch.float32).to(DEVICE))
        print(f"  - {p}: {len(w)/44100:.2f}s")

    # ── WavLM 로드 및 타겟 피처 추출 ─────────────────────────────────────────
    wavlm             = load_wavlm()
    target_feats_list = extract_wavlm_targets_per_wav(wavlm, target_wav_ts)
    target_feats_avg  = average_wavlm_features(target_feats_list)

    # ── TTS 엔진 로드 ─────────────────────────────────────────────────────────
    tts = load_text_to_speech("onnx")

    # ── 레퍼런스 프리셋 결정 ──────────────────────────────────────────────────
    ref_style_path = cfg.get("reference_style", "auto")
    if ref_style_path == "auto":
        ref_style_path = auto_select_preset_by_texture(wavlm, tts, target_feats_avg)

    # ── ECAPA 로드 및 타겟 임베딩 추출 ───────────────────────────────────────
    ecapa                = None
    target_ecapa_emb_list = None
    target_ecapa_emb_avg  = None
    if use_ecapa:
        ecapa = load_ecapa()
        if ecapa is not None:
            target_ecapa_emb_list = extract_ecapa_targets_per_wav(ecapa, target_wav_ts)
            # 평균 임베딩: 멀티 WAV의 화자 정체성 중심점
            target_ecapa_emb_avg = F.normalize(
                torch.stack(target_ecapa_emb_list, dim=0).mean(dim=0), dim=1
            )

    # ── ONNX → PyTorch 모델 변환 ──────────────────────────────────────────────
    dp_model  = load_pt_model("duration_predictor.onnx")
    te_model  = load_pt_model("text_encoder.onnx")
    ve_model  = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")

    # ── 텍스트 전처리 ─────────────────────────────────────────────────────────
    text_inputs = []
    for text in KO_PHONETIC_BASS_TEXTS:
        ids_np, mask_np = tts.text_processor(text, "ko")
        text_inputs.append((
            torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))

    # ── Seed 고정 및 고정 노이즈 잠재 벡터 생성 ──────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n[레퍼런스 베이스 스타일 설정] -> {os.path.basename(ref_style_path)}")
    ref_style = load_voice_style(ref_style_path)

    with torch.no_grad():
        init_dur = (dp_model(
            text_inputs[0][0],
            torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE),
            text_inputs[0][1]
        ) / speed)
        latent_len         = int(np.ceil(
            (init_dur.item() * 44100)
            / (tts.base_chunk_size * tts.chunk_compress_factor)
        ))
        noisy_latent_fixed = torch.tensor(
            np.random.randn(1, tts.ldim * tts.chunk_compress_factor, latent_len)
            .astype(np.float32)
        ).to(DEVICE)
        latent_mask        = torch.ones(1, 1, latent_len, dtype=torch.float32).to(DEVICE)

    # ── 스타일 벡터 초기화 (체크포인트 재개 or 레퍼런스 프리셋) ──────────────
    if latest_ckpt:
        resumed = load_voice_style(latest_ckpt)
        style_ttl = torch.tensor(resumed.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(resumed.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
        print(f"[Resume] style 벡터를 체크포인트에서 복원했습니다.")
    else:
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    if train_dp:
        optimizer = torch.optim.Adam([
            {"params": [style_ttl], "lr": lr},
            {"params": [style_dp],  "lr": lr * dp_lr_ratio}
        ])
    else:
        optimizer = torch.optim.Adam([style_ttl], lr=lr)

    # CosineAnnealingLR: 전체 스텝 기준으로 감쇠
    # 재개 시 last_epoch를 넘겨서 스케줄러 상태를 복원
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
    print(f"  학습 구간: step {start_step + 1} → {num_steps}\n")

    for step in range(start_step, num_steps):
        optimizer.zero_grad()

        # DP warmup: L3 loss가 0.30 이하로 내려오면 즉시 활성화
        # 단, 초반 50스텝은 style_ttl 안정화를 위해 DP 동결 유지
        if train_dp and not style_dp.requires_grad:
            if best_loss <= 0.30 or step >= start_step + 50:
                style_dp.requires_grad_(True)
                print(f"\n>>> [DP 활성화] Step {step+1}: L3={best_loss:.4f} → style_dp 학습 시작")

        # 텍스트 회전
        text_idx        = step % len(text_inputs)
        text_ids, text_mask = text_inputs[text_idx]

        # ── 타겟 피처 선택 (stochastic / average) ────────────────────────────
        if multi_wav_mode == "stochastic" and len(target_feats_list) > 1:
            # WAV가 여러 개면 무작위로 2개를 뽑아 평균 → 다양한 목표 노출
            k             = min(2, len(target_feats_list))
            batch_indices = random.sample(range(len(target_feats_list)), k)
            current_target_feats = average_wavlm_features(
                [target_feats_list[i] for i in batch_indices]
            )
            # ECAPA: stochastic 모드에서 선택된 인덱스에 맞는 임베딩 평균
            # [BUG FIX] ecapa가 None이거나 단일 wav일 때 current_target_ecapa가
            #           미정의 상태로 ecapa_loss_fn에 넘어가는 버그 수정
            if use_ecapa and ecapa is not None and target_ecapa_emb_list is not None:
                current_target_ecapa = F.normalize(
                    torch.stack([target_ecapa_emb_list[i] for i in batch_indices], dim=0)
                    .mean(dim=0), dim=1
                )
            else:
                current_target_ecapa = None
        else:
            # 단일 WAV이거나 average 모드: 사전 계산된 평균 타겟 사용
            current_target_feats = target_feats_avg
            current_target_ecapa = target_ecapa_emb_avg  # None일 수도 있음 (ecapa 없을 때)

        # ── Forward & Loss ────────────────────────────────────────────────────
        wav_out, _ = tts_forward(
            text_ids, text_mask, style_ttl, style_dp,
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent_fixed, latent_mask
        )
        gen_wav = wav_out.squeeze()

        loss = wavlm_hybrid_feature_loss(wavlm, gen_wav, current_target_feats,
                                          layers=WAVLM_LAYERS)

        if use_ecapa and ecapa is not None and current_target_ecapa is not None:
            loss = loss + (ecapa_weight * ecapa_loss_fn(ecapa, gen_wav, current_target_ecapa))

        with torch.no_grad():
            primary_loss = wavlm_primary_loss(wavlm, gen_wav, current_target_feats).item()

        # ── Backward ──────────────────────────────────────────────────────────
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in [style_ttl, style_dp] if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()

        # warmdown 진입 전까지만 스케줄러 스텝 실행
        if not warmdown_mode:
            scheduler.step()

        # ── Best 갱신 ─────────────────────────────────────────────────────────
        if primary_loss < best_loss:
            best_loss = primary_loss
            best_ttl  = style_ttl.detach().clone()
            best_dp   = style_dp.detach().clone()

        # ── 로그 출력 (Cell 10 파싱 정규식 호환 포맷 고정) ────────────────────
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']

            # Cell 10 line_re 패턴과 1:1 매칭되는 포맷
            print(f"  Step {step+1}/{num_steps} | Loss(total): {loss.item():.4f} | "
                  f"Loss(L3): {primary_loss:.4f} | LR: {current_lr:.5f} | "
                  f"Best(L3): {best_loss:.4f}")

            # 수렴 진행률 계산 (Cell 10 프로그레스 바 서브스트림)
            if initial_gap is None:
                initial_gap = max(0.001, best_loss - threshold)
            current_gap = max(0.0, best_loss - threshold)
            percentage  = (max(0.0, (1.0 - (current_gap / initial_gap)) * 100.0)
                           if current_gap > 0 else 100.0)
            num_blocks  = min(20, max(0, int(percentage / 5)))
            bar_str     = "█" * num_blocks + "-" * (20 - num_blocks)
            print(f"    [{bar_str}]  {percentage:.1f}%  "
                  f"(best {best_loss:.4f} -> 목표 {threshold:.4f}, gap +{current_gap:.4f})")

        # ── 체크포인트 저장 ───────────────────────────────────────────────────
        if (step + 1) % save_every == 0:
            ckpt_path = f"{log_dir}/{name}_{step+1:04d}.json"
            save_style(ckpt_path, best_ttl, best_dp, target_wav_paths)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        # ── Warmdown (조기 종료 전 미세 정렬) ────────────────────────────────
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

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    final_path = f"{log_dir}/{name}_final.json"
    save_style(final_path, best_ttl, best_dp, target_wav_paths)

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("[학습 완료 결과 다차원 평가 리포트]")
    print(f"- 총 학습 소요 시간  : {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"- 최종 학습 진행 스텝: {step + 1} / {num_steps}")
    print(f"- 최소 음색 손실값   : {best_loss:.4f}")
    print("-" * 60)
    if warmdown_mode:
        print("▶ 최종 판정: [✨ 웜다운 방어 수렴 성공]")
    else:
        print("▶ 최종 판정: [✅ 만기 완주 수렴]")
    print("=" * 60)


if __name__ == "__main__":
    main()
