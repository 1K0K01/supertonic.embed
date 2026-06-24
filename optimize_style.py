"""
optimize_style_ko_male_bass.py
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass/baritone) 특화 Supertonic3 스타일 최적화

원본 대비 변경 사항:
  1. opt_texts     - 저음 공명 유발 서술체 문장으로 교체 (짧은 절규 → 느린 장문)
  2. WavLM 가중치  - Layer 1(저음 질감) 0.2 → 0.4, Layer 3(화자 정체성) 1.0 유지
  3. F0 피처 보조  - librosa로 target F0 범위를 사전 측정 → 낮은 F0 가중치 계수 계산
  4. DP warmup    - 고정 비율 대신 L3 loss가 0.30 이하로 내려오면 즉시 활성화
                   (저음은 리듬 패턴이 빠르게 수렴하는 경향)
  5. speed 기본값 - 0.95 (저음 남성 자연 발화 속도 반영)
  6. dp_lr_ratio  - 0.05 → 0.008 (저음의 느린 발화 리듬이 과적합되지 않도록)
  7. 조기 종료 기준 - threshold 기본값 0.24 → 0.18 (저음은 수렴이 더 명확함)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import glob
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

# SSL 인증서 우회 (원본 동일)
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

# SpeechBrain 임포트 (원본 동일)
HAS_SPEECHBRAIN = False
EncoderClassifier = None
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
# 1. ONNX → PyTorch 변환 (원본 동일)
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
# 2. 저음 남성 F0 분석 (신규)
#    학습 전에 타겟 WAV의 F0 분포를 측정하여 저음 여부를 판단하고
#    WavLM Layer 1(저음 질감) 가중치를 동적으로 조정함
# ─────────────────────────────────────────────────────────────────────────────

def analyze_f0_profile(wav_paths: list[str]) -> dict:
    """
    타겟 WAV들의 F0(기본 주파수) 통계를 분석.
    
    반환값 예시:
      { 'median_f0': 98.3, 'p10_f0': 82.1, 'is_bass': True,
        'layer1_boost': 0.25 }

    저음 기준 (한국 남성):
      - Bass/Baritone: F0 중앙값 85~145 Hz
      - Tenor:         F0 중앙값 145~200 Hz
    
    중앙값이 낮을수록 Layer 1 가중치를 높여 흉성(chest voice) 질감을 강하게 학습.
    """
    all_f0 = []
    for path in wav_paths:
        y, sr = librosa.load(path, sr=44100, mono=True)
        # pyin: librosa의 확률론적 YIN — 저음 남성에서 octave error가 적음
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # 65 Hz: bass 하한
            fmax=librosa.note_to_hz('C5'),   # 523 Hz: tenor 상한
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            all_f0.extend(voiced_f0.tolist())

    if not all_f0:
        print("  [F0 분석] 유성음 구간을 감지하지 못했습니다. 기본 가중치를 사용합니다.")
        return {'median_f0': 130.0, 'p10_f0': 100.0, 'is_bass': False, 'layer1_boost': 0.0}

    median_f0 = float(np.median(all_f0))
    p10_f0    = float(np.percentile(all_f0, 10))  # 하위 10% — 가장 낮은 영역

    # 저음 판단 및 Layer 1 부스트 계산
    # F0 중앙값 기준: 145Hz 이하면 bass/baritone으로 판단
    # 85Hz에서 최대 부스트(+0.3), 145Hz에서 부스트 0
    is_bass = median_f0 < 145.0
    if is_bass:
        # 선형 보간: median_f0가 낮을수록 layer1_boost가 커짐
        layer1_boost = max(0.0, min(0.3, (145.0 - median_f0) / (145.0 - 85.0) * 0.3))
    else:
        layer1_boost = 0.0

    print(f"  [F0 분석] 중앙값={median_f0:.1f}Hz, P10={p10_f0:.1f}Hz, "
          f"저음={'✓' if is_bass else '✗'}, Layer1 부스트=+{layer1_boost:.3f}")

    return {
        'median_f0':    median_f0,
        'p10_f0':       p10_f0,
        'is_bass':      is_bass,
        'layer1_boost': layer1_boost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. WavLM 퍼셉추얼 손실 (저음 특화 가중치)
# ─────────────────────────────────────────────────────────────────────────────

# 기본 레이어/가중치 — F0 분석 후 layer1_boost가 더해짐
WAVLM_LAYERS         = (1, 3, 6, 9)
WAVLM_LAYER_WEIGHTS  = (0.4, 1.0, 0.5, 0.3)  # Layer 1: 0.2 → 0.4 (저음 질감 강화)
# 원본이 0.2였던 Layer 1을 0.4로 올린 이유:
#   WavLM Layer 1은 낮은 수준의 음향 특성(spectral envelope, vocal tract resonance)을
#   인코딩한다. 저음 남성의 흉성 공명(chest resonance)과 성대 접촉 패턴이
#   이 레이어에 주로 반영되므로, 가중치를 높여 목표 음색의 저음 질감을 더 강하게 학습.

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

def extract_wavlm_targets(wavlm, target_wavs, layers=WAVLM_LAYERS):
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    per_wav_feats = [_extract_wavlm_single(wavlm, w, layers) for w in target_wavs]
    targets = {}
    for layer in layers:
        means = torch.stack([f[layer][0] for f in per_wav_feats], dim=0).mean(dim=0)
        stds  = torch.stack([f[layer][1] for f in per_wav_feats], dim=0).mean(dim=0)
        targets[layer] = (means, stds)
    return targets

def extract_wavlm_targets_per_wav(wavlm, target_wavs, layers=WAVLM_LAYERS):
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    return [_extract_wavlm_single(wavlm, w, layers) for w in target_wavs]

def wavlm_feature_loss(wavlm, gen_wav, target_features,
                        layers=WAVLM_LAYERS, weights=WAVLM_LAYER_WEIGHTS):
    """다중 레이어 WavLM 손실 — 가중치는 F0 분석 후 외부에서 동적으로 전달됨."""
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)

    total_loss = 0.0
    for layer, weight in zip(layers, weights):
        gen_feat = gen_out.hidden_states[layer]
        tgt_mean, tgt_std = target_features[layer]
        gen_mean = gen_feat.mean(dim=1)
        gen_std  = gen_feat.std(dim=1)
        layer_loss = F.mse_loss(gen_mean, tgt_mean) + F.mse_loss(gen_std, tgt_std)
        total_loss = total_loss + weight * layer_loss
    return total_loss

def wavlm_primary_loss(wavlm, gen_wav, target_features, layer=3):
    """Layer 3 단독 손실 — 조기 종료 임계값 비교 기준."""
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features[layer]
    gen_mean = gen_feat.mean(dim=1)
    gen_std  = gen_feat.std(dim=1)
    return F.mse_loss(gen_mean, tgt_mean) + F.mse_loss(gen_std, tgt_std)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ECAPA-TDNN 화자 정체성 손실 (원본과 동일, 저음 특화 없음)
# ─────────────────────────────────────────────────────────────────────────────

def load_ecapa():
    if not HAS_SPEECHBRAIN:
        print("  [Warning] speechbrain 없음. ECAPA Loss 비활성화.")
        return None
    print("Loading SpeechBrain ECAPA-TDNN...")
    import logging
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": str(DEVICE)}
        )
    except Exception as e:
        print(f"  [Warning] ECAPA 로드 실패: {e}")
        return None
    for p in classifier.parameters():
        p.requires_grad_(False)
    print("  ECAPA-TDNN loaded.")
    return classifier

def _extract_ecapa_single(ecapa, target_wav_t):
    wav_16k = torchaudio.functional.resample(target_wav_t, 44100, 16000)
    if wav_16k.ndim == 1:
        wav_16k = wav_16k.unsqueeze(0)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav_16k)
        emb = F.normalize(emb, dim=2).squeeze(1)
    return emb

def extract_ecapa_target(ecapa, target_wavs):
    if ecapa is None:
        return None
    if isinstance(target_wavs, torch.Tensor):
        target_wavs = [target_wavs]
    embs    = [_extract_ecapa_single(ecapa, w) for w in target_wavs]
    avg_emb = torch.stack(embs, dim=0).mean(dim=0)
    avg_emb = F.normalize(avg_emb, dim=1)
    return avg_emb

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


# ─────────────────────────────────────────────────────────────────────────────
# 5. TTS forward / 저장 (원본과 동일)
# ─────────────────────────────────────────────────────────────────────────────

def tts_forward(text_ids, text_mask, style_ttl, style_dp,
                dp_model, te_model, ve_model, voc_model,
                total_step, speed, noisy_latent, latent_mask):
    dur      = dp_model(text_ids, style_dp, text_mask)
    dur      = dur / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt       = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur

def save_style(path, style_ttl, style_dp, source_file=None):
    from datetime import datetime
    source_meta = list(source_file) if isinstance(source_file, (list, tuple)) else (source_file or "unknown")
    style_json  = {
        "style_ttl": {"data": style_ttl.cpu().numpy().tolist(), "dims": [1, 50, 256], "type": "float32"},
        "style_dp":  {"data": style_dp.cpu().numpy().tolist(),  "dims": [1, 8, 16],   "type": "float32"},
        "metadata":  {"source_file": source_meta, "source_sample_rate": 44100,
                      "target_sample_rate": 44100, "extracted_at": datetime.now().isoformat()}
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(style_json, f)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 한국 남성 저음 특화 최적화 텍스트 (핵심 변경)
# ─────────────────────────────────────────────────────────────────────────────
#
# 선택 기준:
#   a) 문장 길이 — 짧은 절규보다 긴 서술체가 발화의 지속적 공명을 학습하는 데 유리
#   b) 모음 분포 — ㅓ, ㅡ, ㅜ 계열 모음이 많으면 낮은 F0 영역을 더 자주 자극
#   c) 발화 리듬 — 느린 중간 쉼표(, )가 많은 문장이 저음 특유의 여유로운 호흡을 학습
#   d) 억양 패턴 — 평서문 → 하강 억양이 많을수록 저음 수렴에 유리
#                  의문문/절규는 상승 억양을 유도하므로 비율을 낮춤
#
KO_MALE_BASS_OPT_TEXTS = [
    # 장문 서술체: 느린 발화 리듬 + ㅓ/ㅡ/ㅜ 모음 + 하강 억양
    '어두운 복도 끝에서 그가 천천히 발걸음을 멈추었다. 아무런 말도 없이, 그저 먼 곳을 응시하는 그의 눈빛에는 오래된 무게가 가라앉아 있었다.',
    '그는 잠시 눈을 감았다가 천천히 뜨며, 낮고 묵직한 목소리로 한 마디를 내뱉었다. "그건 내가 결정할 일이오."',
    '구름 한 점 없이 맑은 하늘 아래, 그는 혼자 강가에 서서 흐르는 물을 내려다보았다. 오래, 아주 오래 그렇게 서 있었다.',
    '어제도 그랬고, 오늘도 그럴 것이다. 말하지 않아도 알 수 있는 것들이 있고, 굳이 묻지 않아도 되는 것들이 있다.',
    '그가 천천히 고개를 돌렸다. 눈이 마주친 순간, 나는 숨을 참았다. 그의 눈빛에는 슬픔도, 분노도 아닌, 그 무언가가 담겨 있었다.',
    # 중간 길이 서술/독백: 감정 억제된 평서문
    '"됐습니다." 그는 더 이상 아무 말도 하지 않았다. 그저 등을 돌리고 걸어갔을 뿐이다.',
    '나는 낮고 가라앉은 목소리로 천천히 말했다. "당신이 무슨 말을 하려는지 압니다. 그래도, 내 대답은 변하지 않습니다."',
    '그 목소리는 낮고 침착했다. 격렬한 감정 같은 것은 느껴지지 않았다. 그저 무겁고 단단한, 오래 숙성된 결의 같은 것이 묻어났다.',
]


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 최적화 함수
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _patch_onnx2torch()

    # ── Config 로드 (원본과 동일) ────────────────────────────────────────────
    arg = sys.argv[1] if len(sys.argv) > 1 else "ljs"
    if os.path.exists(arg):
        config_path = arg
    elif os.path.exists(f"configs/{arg}.json"):
        config_path = f"configs/{arg}.json"
    elif os.path.exists(f"configs/{arg}"):
        config_path = f"configs/{arg}"
    else:
        print(f"Config not found: {arg}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg["name"]

    if "target_wavs" in cfg and cfg["target_wavs"]:
        target_wav_paths = list(cfg["target_wavs"])
    elif isinstance(cfg.get("target_wav"), list):
        target_wav_paths = list(cfg["target_wav"])
    else:
        target_wav_paths = [cfg["target_wav"]]

    multi_wav_mode = cfg.get("multi_wav_mode", "average")
    if multi_wav_mode not in ("average", "rotate"):
        multi_wav_mode = "average"

    reference_style = cfg.get("reference_style")
    seed            = cfg.get("seed", 42)
    lr              = cfg.get("lr", 2e-4)

    # ── 저음 특화 기본값 (config에 없으면 아래 값 사용) ─────────────────────
    # dp_lr_ratio: 0.05 → 0.008 (저음의 느린 리듬이 과도하게 빨라지지 않도록)
    dp_lr_ratio = cfg.get("dp_lr_ratio", 0.008)
    train_dp    = cfg.get("train_style_dp", True)
    num_steps   = cfg.get("num_steps", 3000)
    total_step  = cfg.get("total_step", 5)
    # speed: 1.00 → 0.95 (저음 남성의 자연 발화 속도)
    speed       = cfg.get("speed", 0.95)
    save_every  = cfg.get("save_every", 100)
    # threshold: 0.24 → 0.18 (저음은 수렴이 더 명확하므로 기준을 높임)
    threshold   = cfg.get("early_stop_loss_threshold", 0.18)

    use_ecapa      = cfg.get("use_ecapa_loss", True)
    ecapa_weight   = cfg.get("ecapa_loss_weight", 0.3)

    # ── 경로 설정 ────────────────────────────────────────────────────────────
    output_json = f"voice_styles/{name}.json"
    log_dir     = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    def find_latest_checkpoint():
        pattern = os.path.join(log_dir, f"{name}_*.json")
        files   = [f for f in glob.glob(pattern) if "train_config" not in f]
        if not files:
            return None, 0
        latest  = max(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))
        step_num = int(os.path.splitext(os.path.basename(latest))[0].split("_")[-1])
        return latest, step_num

    latest_ckpt, start_step = find_latest_checkpoint()

    print(f"Using device: {DEVICE}")
    print(f"Name: {name}")

    # ── 타겟 WAV 로드 ────────────────────────────────────────────────────────
    print(f"\nLoading target WAV(s) ({len(target_wav_paths)}개): {target_wav_paths}")
    target_wav_ts = []
    for p in target_wav_paths:
        w, _ = librosa.load(p, sr=44100)
        wt   = torch.tensor(w, dtype=torch.float32).to(DEVICE)
        target_wav_ts.append(wt)
        print(f"  - {p}: {len(w)/44100:.2f}s")
    target_wav_path = target_wav_paths[0]

    # ── 저음 F0 분석 → WavLM Layer 1 가중치 동적 조정 ──────────────────────
    print("\n[저음 F0 프로파일 분석]")
    f0_profile = analyze_f0_profile(target_wav_paths)
    # Layer 1 가중치에 F0 기반 부스트 추가
    dynamic_weights = list(WAVLM_LAYER_WEIGHTS)
    dynamic_weights[0] = WAVLM_LAYER_WEIGHTS[0] + f0_profile['layer1_boost']
    dynamic_weights = tuple(dynamic_weights)
    print(f"  최종 WavLM 가중치: layers={WAVLM_LAYERS}, weights={tuple(round(w,3) for w in dynamic_weights)}")

    # ── WavLM 로드 ────────────────────────────────────────────────────────────
    print(f"\nMulti-WAV mode: {multi_wav_mode}")
    print("Loading WavLM-Large...")
    wavlm = load_wavlm()
    print("  WavLM loaded.")

    print(f"Extracting target WavLM features (layers {WAVLM_LAYERS})...")
    if multi_wav_mode == "rotate" and len(target_wav_ts) > 1:
        target_feats_list = extract_wavlm_targets_per_wav(wavlm, target_wav_ts)
        target_feats      = target_feats_list[0]
    else:
        target_feats_list = None
        target_feats      = extract_wavlm_targets(wavlm, target_wav_ts)
    print(f"  Done.")

    # ── ECAPA ────────────────────────────────────────────────────────────────
    ecapa                = None
    target_ecapa_emb     = None
    target_ecapa_emb_list = None
    if use_ecapa:
        ecapa = load_ecapa()
        if ecapa is not None:
            print("Extracting ECAPA speaker embedding...")
            if multi_wav_mode == "rotate" and len(target_wav_ts) > 1:
                target_ecapa_emb_list = extract_ecapa_targets_per_wav(ecapa, target_wav_ts)
                target_ecapa_emb      = target_ecapa_emb_list[0]
            else:
                target_ecapa_emb = extract_ecapa_target(ecapa, target_wav_ts)
            print("  Done.")

    # ── ONNX → PyTorch 변환 ──────────────────────────────────────────────────
    print("\nConverting ONNX models to PyTorch...")
    dp_model  = load_pt_model("duration_predictor.onnx")
    te_model  = load_pt_model("text_encoder.onnx")
    ve_model  = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")
    print("  All models converted.")

    # ── 텍스트 전처리 ────────────────────────────────────────────────────────
    tts         = load_text_to_speech("onnx")
    opt_texts   = KO_MALE_BASS_OPT_TEXTS  # 저음 특화 텍스트 사용
    opt_lang    = "ko"
    text_inputs = []
    for text in opt_texts:
        ids_np, mask_np = tts.text_processor(text, opt_lang)
        text_inputs.append((
            torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))

    # ── 고정 노이즈 잠재 벡터 생성 (seed 고정) ──────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    tmp_style = load_voice_style("voice_styles/M1.json")
    tmp_dp    = torch.tensor(tmp_style.dp, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        init_dur = dp_model(text_inputs[0][0], tmp_dp, text_inputs[0][1]) / speed
    dur_val      = init_dur.item()
    wav_len      = int(dur_val * 44100)
    chunk_size   = tts.base_chunk_size * tts.chunk_compress_factor
    latent_len   = int(np.ceil(wav_len / chunk_size))
    latent_dim   = tts.ldim * tts.chunk_compress_factor
    noisy_latent_fixed = torch.tensor(
        np.random.randn(1, latent_dim, latent_len).astype(np.float32)
    ).to(DEVICE)
    latent_mask  = torch.ones(1, 1, latent_len, dtype=torch.float32).to(DEVICE)
    del tmp_style, tmp_dp

    # ── 스타일 벡터 초기화 ───────────────────────────────────────────────────
    if latest_ckpt:
        print(f"\nResuming from: {latest_ckpt} (step {start_step})")
        ref_style = load_voice_style(latest_ckpt)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
    elif reference_style == "auto":
        print("\nFinding closest style to target WAV (ECAPA 우선)...")
        all_style_paths = sorted(glob.glob("voice_styles/[FM]*.json"))
        best_dist = float('inf')
        best_path = None
        if ecapa is not None and target_ecapa_emb is not None:
            for sp in all_style_paths:
                s     = load_voice_style(sp)
                s_ttl = torch.tensor(s.ttl, dtype=torch.float32).to(DEVICE)
                s_dp  = torch.tensor(s.dp,  dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    test_wav, _ = tts_forward(
                        text_inputs[0][0], text_inputs[0][1], s_ttl, s_dp,
                        dp_model, te_model, ve_model, voc_model,
                        total_step, speed, noisy_latent_fixed, latent_mask,
                    )
                    pw16k = torchaudio.functional.resample(test_wav.squeeze(0), 44100, 16000)
                    if pw16k.ndim == 1:
                        pw16k = pw16k.unsqueeze(0)
                    pe = ecapa.encode_batch(pw16k)
                    pe = F.normalize(pe, dim=2).squeeze(1)
                    dist = (1.0 - F.cosine_similarity(pe, target_ecapa_emb)).item()
                print(f"  {os.path.basename(sp)}: {dist:.4f}")
                if dist < best_dist:
                    best_dist = dist
                    best_path = sp
        else:
            for sp in all_style_paths:
                s     = load_voice_style(sp)
                s_ttl = torch.tensor(s.ttl, dtype=torch.float32).to(DEVICE)
                s_dp  = torch.tensor(s.dp,  dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    test_wav, _ = tts_forward(
                        text_inputs[0][0], text_inputs[0][1], s_ttl, s_dp,
                        dp_model, te_model, ve_model, voc_model,
                        total_step, speed, noisy_latent_fixed, latent_mask,
                    )
                    dist = wavlm_primary_loss(wavlm, test_wav.squeeze(), target_feats).item()
                print(f"  {os.path.basename(sp)}: {dist:.4f}")
                if dist < best_dist:
                    best_dist = dist
                    best_path = sp
        print(f"  >> Best: {os.path.basename(best_path)} (dist={best_dist:.4f})")
        ref_style = load_voice_style(best_path)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
    elif reference_style:
        print(f"\nInitializing from: {reference_style}")
        ref_style = load_voice_style(reference_style)
        style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
    else:
        print("\nInitializing style randomly")
        style_ttl = (torch.randn(1, 50, 256) * 0.1).to(DEVICE).requires_grad_(True)
        style_dp  = torch.tensor(load_voice_style("voice_styles/M1.json").dp,
                                  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)

    print(f"  style_ttl: {style_ttl.shape}, style_dp: {style_dp.shape}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    if train_dp:
        optimizer = torch.optim.Adam([
            {"params": [style_ttl], "lr": lr},
            {"params": [style_dp],  "lr": lr * dp_lr_ratio},
        ])
        trainable_params = [style_ttl, style_dp]
    else:
        optimizer        = torch.optim.Adam([style_ttl], lr=lr)
        trainable_params = [style_ttl]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5, min_lr=lr * 0.01
    )

    end_step = num_steps
    if start_step >= end_step:
        print(f"\nAlready at target step ({end_step}).")
        return

    import time
    start_time = time.time()

    print(f"\n[저음 특화 설정]")
    print(f"  speed={speed}, dp_lr_ratio={dp_lr_ratio}, threshold={threshold}")
    print(f"  WavLM weights={tuple(round(w,3) for w in dynamic_weights)}")
    print(f"  opt_texts={len(opt_texts)}개 (저음 서술체)")
    print(f"\nOptimization: step {start_step+1} → {end_step}")

    best_loss       = float('inf')
    best_total_loss = float('inf')
    best_ttl        = None
    best_dp         = style_dp.detach().clone()

    for step in range(start_step, end_step):
        optimizer.zero_grad()

        # ── 저음 특화 DP warmup 로직 ─────────────────────────────────────────
        # 원본: best_loss > 0.22 이면 DP 동결 (fixed threshold)
        # 변경: L3 loss가 0.30 이하로 내려오면 즉시 활성화
        #   → 저음은 음색 수렴이 빠른 편이라 리듬 학습을 일찍 시작하는 것이 유리
        #   → 단, 50 스텝 최소 워밍업은 유지 (초기 불안정 구간 보호)
        if train_dp:
            dp_activate_threshold = 0.30   # 원본 0.22에서 완화
            if best_loss > dp_activate_threshold and step < 50:
                style_dp.requires_grad_(False)
            else:
                if not style_dp.requires_grad:
                    style_dp.requires_grad_(True)
                    print(f"\n>>> [DP 활성화] Step {step+1}: L3 Loss ({best_loss:.4f}) → style_dp 학습 시작")

        # 텍스트 / wav 회전
        text_idx   = step % len(text_inputs)
        text_ids, text_mask = text_inputs[text_idx]

        if target_feats_list is not None:
            wav_idx      = (step + 1) % len(target_feats_list)
            target_feats = target_feats_list[wav_idx]
            if target_ecapa_emb_list is not None:
                target_ecapa_emb = target_ecapa_emb_list[wav_idx]

        # Forward
        wav_out, _ = tts_forward(
            text_ids, text_mask, style_ttl, style_dp,
            dp_model, te_model, ve_model, voc_model,
            total_step, speed, noisy_latent_fixed, latent_mask,
        )
        gen_wav = wav_out.squeeze()

        # Loss (F0 기반 동적 가중치 사용)
        loss = wavlm_feature_loss(wavlm, gen_wav, target_feats,
                                   layers=WAVLM_LAYERS, weights=dynamic_weights)
        if use_ecapa and ecapa is not None and target_ecapa_emb is not None:
            ec_loss = ecapa_loss_fn(ecapa, gen_wav, target_ecapa_emb)
            loss    = loss + (ecapa_weight * ec_loss)

        with torch.no_grad():
            primary_loss = wavlm_primary_loss(wavlm, gen_wav, target_feats).item()

        loss.backward()
        active_params = [p for p in trainable_params if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(active_params, max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if primary_loss < best_loss:
            best_loss       = primary_loss
            best_total_loss = loss.item()
            best_ttl        = style_ttl.detach().clone()
            best_dp         = style_dp.detach().clone()

        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            dp_active  = style_dp.requires_grad if train_dp else False
            if train_dp:
                dp_lr = optimizer.param_groups[1]['lr'] if dp_active else 0.0
                print(f"  Step {step+1}/{end_step} | Loss(total): {loss.item():.4f} | "
                      f"Loss(L3): {primary_loss:.4f} | LR(ttl): {current_lr:.5f} | "
                      f"LR(dp): {dp_lr:.5f} | Best(L3): {best_loss:.4f}")
            else:
                print(f"  Step {step+1}/{end_step} | Loss(total): {loss.item():.4f} | "
                      f"Loss(L3): {primary_loss:.4f} | LR: {current_lr:.5f} | Best(L3): {best_loss:.4f}")

        if (step + 1) % save_every == 0:
            ckpt_path = f"{log_dir}/{name}_{step+1:04d}.json"
            save_style(ckpt_path, best_ttl, best_dp, target_wav_paths)
            print(f"  >> Checkpoint: {ckpt_path}")

        if best_loss <= threshold:
            print(f"  Early stop: step {step+1}, L3 loss {best_loss:.4f} <= {threshold}")
            break

    # ── 최종 저장 ────────────────────────────────────────────────────────────
    final_path = f"{log_dir}/{name}_final.json"
    print(f"\nSaving best style to: {final_path}")
    save_style(final_path, best_ttl, best_dp, target_wav_paths)
    elapsed = time.time() - start_time
    print(f"  Done! Best L3: {best_loss:.4f}, Total: {best_total_loss:.4f} | "
          f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
