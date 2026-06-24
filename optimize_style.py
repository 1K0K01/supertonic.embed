"""
optimize_style_advanced_ko.py (최종 무결성 패치판)
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass) 특화 및 발음 해상도 개선 Supertonic3 최적화 코드
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import random
import time
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
# 3. F0 기반 음원 프로파일링 및 자동 프리셋 매칭
# ─────────────────────────────────────────────────────────────────────────────
def analyze_f0_profile(wav_paths: list[str]) -> dict:
    all_f0 = []
    for path in wav_paths:
        y, sr = librosa.load(path, sr=44100, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'), sr=sr
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            all_f0.extend(voiced_f0.tolist())

    if not all_f0:
        return {'median_f0': 130.0, 'recommended_preset': 'voice_styles/M2.json'}

    median_f0 = float(np.median(all_f0))

    if median_f0 < 115.0:
        recommended_preset = "voice_styles/M5.json"
    elif 115.0 <= median_f0 < 130.0:
        recommended_preset = "voice_styles/M4.json"
    else:
        recommended_preset = "voice_styles/M2.json"

    print(f"  [F0 프로파일링] 유효 발성 구간 측정 중앙값: {median_f0:.1f}Hz")
    print(f"  [자동 앵커 매칭] 추천 베이스 레퍼런스 스타일: {recommended_preset}")
    
    return {'median_f0': median_f0, 'recommended_preset': recommended_preset}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 하이브리드 특징 손실 산출 모듈 (WavLM + RFFT 마스킹)
# ─────────────────────────────────────────────────────────────────────────────
WAVLM_LAYERS         = (1, 3, 6, 9)
WAVLM_LAYER_WEIGHTS  = (0.4, 1.0, 0.5, 0.3)

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
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    gen_feat = gen_out.hidden_states[layer]
    tgt_mean, tgt_std = target_features[layer]
    return F.mse_loss(gen_feat.mean(dim=1), tgt_mean) + F.mse_loss(gen_feat.std(dim=1), tgt_std)

def wavlm_hybrid_feature_loss(wavlm, gen_wav, target_features, median_f0, layers=WAVLM_LAYERS, weights=WAVLM_LAYER_WEIGHTS):
    if gen_wav.ndim == 1:
        gen_wav = gen_wav.unsqueeze(0)
        
    gen_wav_16k = torchaudio.functional.resample(gen_wav, 44100, 16000)
    gen_out = wavlm(gen_wav_16k, output_hidden_states=True)
    
    wavlm_loss = 0.0
    for layer, weight in zip(layers, weights):
        gen_feat = gen_out.hidden_states[layer]
        tgt_mean, tgt_std = target_features[layer]
        layer_loss = F.mse_loss(gen_feat.mean(dim=1), tgt_mean) + F.mse_loss(gen_feat.std(dim=1), tgt_std)
        wavlm_loss = wavlm_loss + weight * layer_loss
        
    gen_fft = torch.fft.rfft(gen_wav)
    gen_fft_mag = torch.abs(gen_fft)
    
    freqs = torch.fft.rfftfreq(gen_wav.shape[-1], d=1/44100.0).to(gen_wav.device)
    hf_mask = (freqs > 4000.0).float()
    
    target_zero = torch.zeros_like(gen_fft_mag)
    rfft_loss = F.l1_loss(gen_fft_mag * hf_mask, target_zero * hf_mask)
    
    hf_weight = 0.05 * (115.0 / max(median_f0, 1.0))
    total_loss = wavlm_loss + (hf_weight * rfft_loss)
    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# 5. 메인 최적화 루프 및 파일 저장 모듈
# ─────────────────────────────────────────────────────────────────────────────
def tts_forward(text_ids, text_mask, style_ttl, style_dp, dp_model, te_model, ve_model, voc_model, total_step, speed, noisy_latent, latent_mask):
    dur = dp_model(text_ids, style_dp, text_mask) / speed
    text_emb = te_model(text_ids, style_ttl, text_mask)
    xt = noisy_latent * latent_mask
    total_step_t = torch.tensor([total_step], dtype=torch.float32).to(DEVICE)
    for step in range(total_step):
        current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
        xt = ve_model(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)
    wav = voc_model(xt)
    return wav, dur

def save_style(path, style_ttl, style_dp, source_file=None):
    source_meta = list(source_file) if isinstance(source_file, (list, tuple)) else (source_file or "unknown")
    
    style_json  = {
        "style_ttl": {"data": style_ttl.detach().cpu().numpy().tolist(), "dims": [1, 50, 256], "type": "float32"},
        "style_dp":  {"data": style_dp.detach().cpu().numpy().tolist(),  "dims": [1, 8, 16],   "type": "float32"},
        "metadata":  {"source_file": source_meta, "source_sample_rate": 44100, "target_sample_rate": 44100, "extracted_at": datetime.now().isoformat()}
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(style_json, f, ensure_ascii=False, indent=2)


def main():
    _patch_onnx2torch()
    # 인자 추출 방식 인덱싱 버그 픽스 완료
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

    name = cfg["name"]
    target_wav_paths = list(cfg["target_wavs"]) if "target_wavs" in cfg else [cfg["target_wav"]]
    multi_wav_mode = cfg.get("multi_wav_mode", "stochastic")
    if multi_wav_mode == "rotate":
        multi_wav_mode = "stochastic" 
        
    seed            = cfg.get("seed", 42)
    lr              = cfg.get("lr", 2e-4)
    dp_lr_ratio     = cfg.get("dp_lr_ratio", 0.008)
    train_dp        = cfg.get("train_style_dp", True)
    num_steps       = cfg.get("num_steps", 3000)
    total_step      = cfg.get("total_step", 5)
    speed           = cfg.get("speed", 0.95)
    save_every      = cfg.get("save_every", 100)
    threshold       = cfg.get("early_stop_loss_threshold", 0.18)
    use_ecapa       = cfg.get("use_ecapa_loss", True)
    ecapa_weight    = cfg.get("ecapa_loss_weight", 0.3)

    log_dir = f"logs/{name}"
    os.makedirs(log_dir, exist_ok=True)

    target_wav_ts = []
    for p in target_wav_paths:
        w, _ = librosa.load(p, sr=44100)
        target_wav_ts.append(torch.tensor(w, dtype=torch.float32).to(DEVICE))

    f0_profile = analyze_f0_profile(target_wav_paths)
    ref_style_path = cfg.get("reference_style", "auto")
    
    if ref_style_path == "auto":
        ref_style_path = f0_profile['recommended_preset']

    wavlm = load_wavlm()
    target_feats_list = extract_wavlm_targets_per_wav(wavlm, target_wav_ts)
    target_feats_avg = average_wavlm_features(target_feats_list)

    ecapa, target_ecapa_emb_list, target_ecapa_emb_avg = None, None, None
    if use_ecapa:
        ecapa = load_ecapa()
        if ecapa is not None:
            target_ecapa_emb_list = extract_ecapa_targets_per_wav(ecapa, target_wav_ts)
            target_ecapa_emb_avg = F.normalize(torch.stack(target_ecapa_emb_list, dim=0).mean(dim=0), dim=1)

    dp_model  = load_pt_model("duration_predictor.onnx")
    te_model  = load_pt_model("text_encoder.onnx")
    ve_model  = load_pt_model("vector_estimator.onnx")
    voc_model = load_pt_model("vocoder.onnx")
    tts = load_text_to_speech("onnx")

    text_inputs = []
    for text in KO_PHONETIC_BASS_TEXTS:
        ids_np, mask_np = tts.text_processor(text, "ko")
        text_inputs.append((
            torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n[레퍼런스 베이스 스타일 설정] -> {os.path.basename(ref_style_path)}")
    ref_style = load_voice_style(ref_style_path)

    with torch.no_grad():
        init_dur = dp_model(text_inputs[0][0], torch.tensor(ref_style.dp, dtype=torch.float32).to(DEVICE), text_inputs[0][1]) / speed
        latent_len = int(np.ceil((init_dur.item() * 44100) / (tts.base_chunk_size * tts.chunk_compress_factor)))
        noisy_latent_fixed = torch.tensor(np.random.randn(1, tts.ldim * tts.chunk_compress_factor, latent_len).astype(np.float32)).to(DEVICE)
        latent_mask = torch.ones(1, 1, latent_len, dtype=torch.float32).to(DEVICE)

    style_ttl = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
    style_dp  = torch.tensor(ref_style.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)

    optimizer = torch.optim.Adam([
        {"params": [style_ttl], "lr": lr},
        {"params": [style_dp],  "lr": lr * dp_lr_ratio}
    ]) if train_dp else torch.optim.Adam([style_ttl], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr*0.01)

    best_loss = float('inf')
    best_ttl, best_dp = None, style_dp.detach().clone()
    warmdown_mode = False
    warmdown_steps = 0
    MAX_WARMDOWN = 50 

    start_time = time.time()
    print(f"\n[Dynamic Optimization 구동] 목표 임계치: {threshold}")

    for step in range(num_steps):
        optimizer.zero_grad()

        if train_dp and best_loss <= 0.30 and not style_dp.requires_grad:
            style_dp.requires_grad_(True)

        text_idx = step % len(text_inputs)
        text_ids, text_mask = text_inputs[text_idx]

        if multi_wav_mode == "stochastic" and len(target_feats_list) > 1:
            k = min(2, len(target_feats_list))
            batch_indices = random.sample(range(len(target_feats_list)), k)
            current_target_feats = average_wavlm_features([target_feats_list[i] for i in batch_indices])
            if use_ecapa and ecapa:
                current_target_ecapa = F.normalize(torch.stack([target_ecapa_emb_list[i] for i in batch_indices], dim=0).mean(dim=0), dim=1)
        else:
            current_target_feats = target_feats_avg
            current_target_ecapa = target_ecapa_emb_avg

        wav_out, _ = tts_forward(text_ids, text_mask, style_ttl, style_dp, dp_model, te_model, ve_model, voc_model, total_step, speed, noisy_latent_fixed, latent_mask)
        gen_wav = wav_out.squeeze()

        loss = wavlm_hybrid_feature_loss(wavlm, gen_wav, current_target_feats, f0_profile['median_f0'], layers=WAVLM_LAYERS)

        if use_ecapa and ecapa is not None:
            loss = loss + (ecapa_weight * ecapa_loss_fn(ecapa, gen_wav, current_target_ecapa))

        with torch.no_grad():
            primary_loss = wavlm_primary_loss(wavlm, gen_wav, current_target_feats).item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in [style_ttl, style_dp] if p.requires_grad], max_norm=1.0)
        optimizer.step()

        if not warmdown_mode:
            scheduler.step()

        if primary_loss < best_loss:
            best_loss = primary_loss
            best_ttl = style_ttl.detach().clone()
            best_dp = style_dp.detach().clone()

        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            mode_str = "[웜다운 정렬]" if warmdown_mode else "[일반 탐색]"
            print(f"  Step {step+1}/{num_steps} {mode_str} | LR: {current_lr:.7f} | Loss(total): {loss.item():.4f} | Loss(L3): {primary_loss:.4f} | Best(L3): {best_loss:.4f}")

        if (step + 1) % save_every == 0:
            save_style(f"{log_dir}/{name}_{step+1:04d}.json", best_ttl, best_dp, target_wav_paths)

        if not warmdown_mode and best_loss <= threshold:
            print(f"\n>>> [Phase Transition] Best L3 Loss Threshold({threshold}) 도달!")
            print(">>> 즉시 종료하지 않고 Dynamic Warm-down(동적 위상 스무딩) 모드를 활성화합니다.")
            print(">>> Learning Rate를 기존 5% 수준으로 강제 드롭하여 가중치 행렬 표면을 미세 정렬합니다.\n")
            
            warmdown_mode = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.05
                
        if warmdown_mode:
            warmdown_steps += 1
            if warmdown_steps >= MAX_WARMDOWN:
                print(f">>> [Early Stop] {MAX_WARMDOWN}스텝 미세 정렬 완료. 고주파 왜곡을 최소화하고 안전하게 종료합니다.")
                break

    save_style(f"{log_dir}/{name}_final.json", best_ttl, best_dp, target_wav_paths)
    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("[학습 완료 결과 다차원 평가 리포트]")
    print(f"- 총 학습 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"- 최종 학습 진행 스텝: {step + 1} / {num_steps}")
    print(f"- 최소 음색 손실값 (Best L3 Loss): {best_loss:.4f}")
    print("-" * 60)
    
    if warmdown_mode:
        print("▶ 최종 판정: [✨ 웜다운 방어 수렴 성공]")
    else:
        print("▶ 최종 판정: [✅ 만기 완주 수렴]")
    print("=" * 60)

if __name__ == "__main__":
    main()
