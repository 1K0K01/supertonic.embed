"""
optimize_style_v5.py (v4 기반 — v4g/Gemini 리뷰 반영판)
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass) 특화 Supertonic3 스타일 최적화 코드 — v4 기반 v5

v5 수정/신규 사항 (v4g[Gemini]가 제안한 3개 기법 검토 후 반영):
  [ADOPT] style_reg_weight (std 0.06~0.08 경계 페널티)
          가이드 문서의 진단 기준을 로그 경고에서 실제 loss 항으로 격상.
          v4g 원안 그대로 채택.
  [ADOPT-PARTIAL] style_reg_ref_anchor_weight (기본값 0, 비활성)
          v4g는 초기 레퍼런스 프리셋으로의 L2 앵커를 std 페널티와 함께 항상 켰지만,
          이는 "레퍼런스에서 타겟 목소리 쪽으로 이동"이라는 학습 목적 자체와
          상충할 위험이 있어(카일루스 사례처럼 최적점이 초기 프리셋에서 먼 경우)
          기본 비활성화. 필요 시에만 config로 켤 것.
  [ADOPT-GATED] mss_loss_weight (Multi-Scale Spectral Loss, 기본값 0)
          다중 해상도 STFT 손실 자체는 유효한 기법이나, v4g 구현은 길이가 다른
          파형을 F.interpolate로 강제 정렬해 duration 미수렴 구간(학습 초반)에
          오정렬 프레임 비교로 인한 노이즈 그래디언트 위험이 있음. dur 오차율이
          mss_dur_tolerance(기본 0.15) 이내인 페어드 스텝에서만 게이팅 적용하도록
          안전장치 추가.
  [REJECT] VAD 기반 순수 발화 길이(librosa.effects.split)로 dur_loss 타겟 변경
          v4g는 문장 내부 모든 무음을 제거한 "순수 유성음 합"을 dur 타겟으로
          썼으나, 모델이 예측하는 dur은 문장부호에 따른 자연스러운 쉼을 포함한
          "전체 발화 길이"이므로 의미가 어긋남. 특히 쉼표가 많은 저음 문어체
          텍스트(가이드 7장)에서는 style_dp가 쉼을 깎아내는 방향으로 왜곡될
          위험이 있어 채택하지 않음. 기존 librosa.effects.trim(앞뒤 무음만 제거)
          방식 유지.

v4 수정/신규 사항 (컨텍스트 md의 카일루스 사례 대응):
  [FIX-D] paired_ratio 웜다운 하향 실제 구현
          문서(md 3장)에는 "웜다운 진입 시 paired_ratio를 0.3으로 하향"이라고
          적혀 있었으나 실제 루프에는 반영되어 있지 않았음. warmdown_paired_ratio
          config로 조절 가능 (기본 0.3).
  [FIX-E] EMA 기반 조기종료 판정 (카일루스 900스텝 언더핏 사례 원인 대응)
          기존에는 매 스텝 단일 텍스트의 L3 손실이 threshold 이하이면 즉시
          웜다운 진입 → 텍스트 난이도 편차로 "운 좋은 샘플 1개"에 조기 락인.
          smoothed_l3(EMA, early_stop_ema_alpha)로 판정하도록 변경. best 체크
          포인트 저장 로직(순간 최저값 추적)은 기존 그대로 유지.
  [FIX-F] ltas_weight 기본값 0.5 → 0.2
          가이드 문서 자체에 명시된 권장 범위(0.15~0.25)와 기존 기본값이
          불일치했음. config에 명시하면 그 값이 우선하므로 하위 호환.
  [NEW-3] style_ttl std 실시간 모니터링 + 정상범위(0.06~0.08) 이탈 경고

v3 수정/신규 사항:
  [FIX-A] hf_weight를 config에서 읽어 손실 함수에 실제 반영
          (v2까지는 0.05 하드코딩 — Cell 7/9 파이프라인이 죽은 코드였음)
  [FIX-B] style_dp gradient 부재 버그 대응:
          - 페어드 모드(target_texts 제공 시): dur 손실로 진짜 gradient 공급
          - 언페어드 모드: train_style_dp 자동 비활성화 + 경고 (거짓 동작 제거)
  [FIX-C] 텍스트별 고정 latent 사전 생성
          (v2는 텍스트 1번 길이의 latent 하나를 모든 텍스트에 재사용
           → 장문 압축/단문 늘어짐 → 텍스트 길이 의존 센트로이드 잔차의 원인)
  [NEW-1] LTAS(장기 평균 스펙트럼) 양방향 매칭 손실 (ltas_weight)
          - 기존 rfft 억제는 "밝음"만 잡는 단방향. LTAS는 원본의 대역별
            에너지 분포를 타겟으로 하여 밝음/탁함 양쪽 모두 교정.
          - 분포 정규화라 텍스트 길이에 불변.
  [NEW-2] 페어드 텍스트 모드 (target_texts)
          - 각 WAV의 실제 대사를 합성해 동일 문장끼리 비교
          - WavLM Layer 6 시퀀스(시간축) 손실로 prosody 감독 (seq_loss_weight)
          - 원본 발화 길이 vs 예측 dur 손실 (dur_loss_weight) → style_dp 학습
          - paired_ratio로 페어드/일반 텍스트 혼합 비율 조절

신규 config 키 (모두 선택적, 기본값 有):
  "target_texts":              ["대사1", "대사2", ...]  # target_wavs와 1:1 순서 대응
  "hf_weight":                  0.05
  "ltas_weight":                0.2                      # v4: 기본값 하향
  "seq_loss_weight":            0.3
  "dur_loss_weight":            0.3
  "paired_ratio":               0.7
  "warmdown_paired_ratio":      0.3                      # v4 신규
  "early_stop_ema_alpha":       0.08                      # v4 신규
  "style_reg_weight":           0.05                      # v5 신규 (std 경계 페널티)
  "style_reg_ref_anchor_weight":0.0                       # v5 신규 (기본 꺼짐)
  "mss_loss_weight":            0.0                       # v5 신규 (기본 꺼짐)
  "mss_dur_tolerance":          0.15                      # v5 신규
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
SEQ_LAYER           = 6   # 페어드 시퀀스 손실용 레이어 (운율 패턴)

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

    print(f"  ▶ 질감 매칭 1위 프리셋: {os.path.basename(best_preset)} "
          f"(최단 거리 점수: {results[0][1]:.4f})")
    return best_preset


# ─────────────────────────────────────────────────────────────────────────────
# 1b. [v5-NEW] Multi-Scale Spectral Loss (v4g 검토 후 게이팅 조건부 차용)
#     페어드 스텝에서 생성 파형 vs 원본 파형을 직접 다중 해상도 STFT로 비교.
#     주의: 길이가 다르면 F.interpolate로 시간축을 강제 정렬하는데, 이는 DTW가
#     아닌 단순 선형 리샘플이라 duration이 크게 어긋난 상태(학습 초반, style_dp
#     미수렴 시)에서는 잘못 정렬된 프레임끼리 비교해 노이즈 그래디언트를 줄 수
#     있음. 그래서 main() 루프에서 dur 오차가 mss_dur_tolerance 이내일 때만
#     활성화하도록 게이팅한다. 기본 가중치는 0(꺼짐).
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
        s_x = torch.stft(x, self.n_fft, self.hop_length, self.win_length,
                          window=window, return_complex=True).abs() + 1e-7
        s_y = torch.stft(y, self.n_fft, self.hop_length, self.win_length,
                          window=window, return_complex=True).abs() + 1e-7

        if s_x.shape[-1] != s_y.shape[-1]:
            s_x = F.interpolate(s_x.unsqueeze(0), size=s_y.shape[-1],
                                 mode='linear', align_corners=False).squeeze(0)

        converge_loss = torch.norm(s_y - s_x, p="fro") / (torch.norm(s_y, p="fro") + 1e-7)
        log_mag_loss  = F.l1_loss(torch.log10(s_y), torch.log10(s_x))
        return converge_loss + log_mag_loss

class MultiScaleSpectralLoss(torch.nn.Module):
    def __init__(self, scales=(512, 1024, 2048), hops=(120, 240, 480), wins=(600, 1200, 2400)):
        super().__init__()
        self.losses = torch.nn.ModuleList([
            SingleScaleSpectralLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(scales, hops, wins)
        ])

    def forward(self, x, y):
        return sum(loss_fn(x, y) for loss_fn in self.losses) / len(self.losses)


def load_wavlm():
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained('microsoft/wavlm-large').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _extract_wavlm_single(wavlm, target_wav, layers=WAVLM_LAYERS,
                          keep_seq_layer=None):
    """mean/std 통계 추출. keep_seq_layer 지정 시 해당 레이어의
    시퀀스 전체([1,T,D])도 함께 반환 (페어드 시퀀스 손실용)."""
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

def extract_wavlm_targets_per_wav(wavlm, target_wavs, layers=WAVLM_LAYERS,
                                  keep_seq_layer=None):
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


# ── [NEW-1] LTAS (장기 평균 스펙트럼) ─────────────────────────────────────────
# 원본 목소리의 "주파수 대역별 에너지 분포"를 타겟으로 하는 양방향 손실.
# 기존 rfft 억제(고주파→0)는 밝은 왜곡만 잡고 탁한 왜곡(-센트로이드)은 못 잡음.
# 분포는 합=1로 정규화되어 텍스트 길이/음량에 불변.

N_LTAS_BANDS = 24
_LTAS_EDGES  = None  # lazy init

def _ltas_band_edges(device):
    global _LTAS_EDGES
    if _LTAS_EDGES is None:
        # 80Hz ~ 11kHz 로그 간격 24밴드
        _LTAS_EDGES = torch.logspace(
            np.log10(80.0), np.log10(11000.0), N_LTAS_BANDS + 1
        ).to(device)
    return _LTAS_EDGES.to(device)

def compute_ltas_distribution(wav, sr=44100):
    """wav([T] or [1,T]) → 정규화된 대역 에너지 분포 [N_LTAS_BANDS]. 미분 가능."""
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
    """log 분포 L1 — 원본 대비 밝음(+)/탁함(-) 양방향 모두 패널티."""
    gen_ltas = compute_ltas_distribution(gen_wav)
    return F.l1_loss(torch.log10(gen_ltas + 1e-8),
                     torch.log10(target_ltas + 1e-8))


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
                               hf_weight=0.05,
                               layers=WAVLM_LAYERS, weights=WAVLM_LAYER_WEIGHTS,
                               target_seq=None, seq_loss_weight=0.0):
    """
    [FIX-A] hf_weight를 인자로 받아 config 값이 실제 반영되도록 수정.
    [NEW-2] target_seq 제공 시(페어드 모드) WavLM Layer 6 시퀀스 손실 추가:
            생성/타겟 시퀀스를 시간축 adaptive pooling으로 동일 길이(64)로
            정렬한 뒤 MSE — 문장 내 운율 궤적(억양 흐름)을 감독.
    반환: (total_loss, gen_out)  ← gen_out 재사용으로 WavLM 중복 forward 제거
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

    # 고주파 억제 (레거시, 단방향): 44100Hz 기준 4kHz 이상 성분을 0 방향으로
    total_loss = wavlm_loss
    if hf_weight > 0:
        gen_fft     = torch.fft.rfft(gen_wav)
        gen_fft_mag = torch.abs(gen_fft)
        freqs       = torch.fft.rfftfreq(gen_wav.shape[-1], d=1/44100.0).to(gen_wav.device)
        hf_mask     = (freqs > 4000.0).float()
        rfft_loss   = (gen_fft_mag * hf_mask).abs().mean()
        total_loss  = total_loss + hf_weight * rfft_loss

    # [NEW-2] 페어드 시퀀스 손실 (운율 궤적 정렬)
    if target_seq is not None and seq_loss_weight > 0:
        gen_seq = gen_out.hidden_states[SEQ_LAYER]          # [1, T1, D]
        T_ALIGN = 64
        g = F.adaptive_avg_pool1d(gen_seq.transpose(1, 2), T_ALIGN)      # [1,D,64]
        t = F.adaptive_avg_pool1d(target_seq.transpose(1, 2), T_ALIGN)   # [1,D,64]
        seq_loss   = F.mse_loss(g, t)
        total_loss = total_loss + seq_loss_weight * seq_loss

    return total_loss, gen_out


def wavlm_primary_loss_from_out(gen_out, target_features, layer=3):
    """이미 계산된 gen_out에서 L3 손실 재계산 (중복 forward 방지)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 6. 체크포인트 재개(Resume) 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 7. 텍스트별 고정 latent 생성 [FIX-C]
# ─────────────────────────────────────────────────────────────────────────────
def build_per_text_latents(text_inputs, ref_style, dp_model, tts, speed, seed):
    """각 텍스트의 예측 길이에 맞는 고정 노이즈 latent/mask 생성.
    v2는 텍스트 1번 길이의 latent를 모든 텍스트에 재사용 →
    장문은 압축, 단문은 늘어진 채 학습되어 텍스트 길이 의존 왜곡 발생."""
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

    # [FIX-A] hf_weight config 연동 + 신규 손실 가중치들
    # [v4] ltas_weight 기본값을 0.5→0.2로 조정: 가이드 문서(9장 카일루스 교훈)에서
    #      0.5는 과도한 저음 유도로 먹먹함을 유발한다고 자체 확인됨. 명시적으로
    #      config에 값을 넣으면 그 값이 우선하므로 기존 실험 재현성은 유지됨.
    hf_weight       = cfg.get("hf_weight", 0.05)
    ltas_weight     = cfg.get("ltas_weight", 0.2)
    seq_loss_weight = cfg.get("seq_loss_weight", 0.3)
    dur_loss_weight = cfg.get("dur_loss_weight", 0.3)
    paired_ratio    = cfg.get("paired_ratio", 0.7)
    # [v4-NEW] 웜다운 진입 시 하향할 paired_ratio 상한 (문서에는 있었으나
    # 실제 루프에 미반영이었던 기능을 구현)
    warmdown_paired_ratio = cfg.get("warmdown_paired_ratio", 0.3)
    # [v4-NEW] 조기종료 판정용 EMA 평활 계수. 값이 작을수록 더 많은 스텝의
    # 평균을 반영해 "운 좋은 샘플 1개"로 인한 조기 락인을 방지한다.
    early_stop_ema_alpha = cfg.get("early_stop_ema_alpha", 0.08)

    # [v5-NEW] v4g(Gemini) 리뷰 후 조건부 차용
    # - style_reg_weight: std 0.06~0.08 경계 이탈 시 페널티 (v4g 원안 채택)
    # - style_reg_ref_anchor_weight: 초기 레퍼런스로의 L2 앵커. 기본 0(끔).
    #   최적점이 레퍼런스에서 먼 경우(카일루스 사례처럼) 매칭을 방해할 수 있어
    #   기본 비활성화 — 과적합/스타일 발산이 실제로 관측될 때만 켤 것.
    style_reg_weight            = cfg.get("style_reg_weight", 0.05)
    style_reg_ref_anchor_weight = cfg.get("style_reg_ref_anchor_weight", 0.0)
    # - mss_loss_weight: Multi-Scale Spectral Loss. 기본 0(끔). duration이
    #   mss_dur_tolerance 이내로 수렴한 페어드 스텝에서만 게이팅 적용(v4g 원안엔
    #   없던 안전장치 — 초반 duration 미수렴 구간의 오정렬 프레임 비교 방지).
    mss_loss_weight   = cfg.get("mss_loss_weight", 0.0)
    mss_dur_tolerance = cfg.get("mss_dur_tolerance", 0.15)

    # [NEW-2] 페어드 텍스트 (target_wavs와 1:1)
    target_texts = cfg.get("target_texts", None)
    paired_mode  = bool(target_texts)
    if paired_mode and len(target_texts) != len(target_wav_paths):
        print(f"[경고] target_texts({len(target_texts)}) ≠ "
              f"target_wavs({len(target_wav_paths)}) — 페어드 모드 비활성화")
        paired_mode = False

    # [FIX-B] 언페어드 모드에서 style_dp는 gradient가 흐르지 않음 (dur 미사용)
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

    # ── 타겟 WAV 로드 ────────────────────────────────────────────────────────
    print(f"\nDevice: {DEVICE}")
    print(f"Name  : {name}")
    print(f"Mode  : {'PAIRED(대사 페어링)' if paired_mode else 'UNPAIRED(일반)'}")
    print(f"\nLoading target WAV(s) ({len(target_wav_paths)}개): {target_wav_paths}")
    target_wav_ts   = []
    target_dur_secs = []   # 무음 트리밍 후 발화 길이 (dur 손실 타겟)
    for p in target_wav_paths:
        w, _ = librosa.load(p, sr=44100)
        target_wav_ts.append(torch.tensor(w, dtype=torch.float32).to(DEVICE))
        w_trim, _ = librosa.effects.trim(w, top_db=35)
        target_dur_secs.append(len(w_trim) / 44100.0)
        print(f"  - {p}: {len(w)/44100:.2f}s (트리밍 후 {len(w_trim)/44100:.2f}s)")

    # ── WavLM 로드 및 타겟 피처 추출 ─────────────────────────────────────────
    wavlm = load_wavlm()
    target_feats_list, target_seq_list = extract_wavlm_targets_per_wav(
        wavlm, target_wav_ts,
        keep_seq_layer=(SEQ_LAYER if paired_mode else None)
    )
    target_feats_avg = average_wavlm_features(target_feats_list)

    # [NEW-1] 타겟 LTAS 분포 (per-wav + 평균)
    target_ltas_list = [compute_ltas_distribution(w).detach()
                        for w in target_wav_ts]
    target_ltas_avg  = torch.stack(target_ltas_list, dim=0).mean(dim=0)
    target_ltas_avg  = target_ltas_avg / target_ltas_avg.sum()

    # [v5-NEW] Multi-Scale Spectral Loss 함수 (mss_loss_weight > 0일 때만)
    mss_loss_fn = MultiScaleSpectralLoss().to(DEVICE) if mss_loss_weight > 0 else None

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
    # 일반(회전) 텍스트
    generic_text_inputs = []
    for text in KO_PHONETIC_BASS_TEXTS:
        ids_np, mask_np = tts.text_processor(text, "ko")
        generic_text_inputs.append((
            torch.tensor(ids_np,  dtype=torch.long).to(DEVICE),
            torch.tensor(mask_np, dtype=torch.float32).to(DEVICE)
        ))
    # 페어드 텍스트
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

    # [v5-NEW] 앵커 정규화용 초기 레퍼런스 고정 복사본 (style_reg_ref_anchor_weight>0일 때만 사용)
    ref_style_ttl_frozen = torch.tensor(ref_style.ttl, dtype=torch.float32).to(DEVICE).clone()
    ref_style_ttl_frozen.requires_grad_(False)

    # [FIX-C] 텍스트별 고정 latent 생성
    print("[*] 텍스트별 고정 latent 생성 중... (FIX-C)")
    gen_latents, gen_masks = build_per_text_latents(
        generic_text_inputs, ref_style, dp_model, tts, speed, seed)
    if paired_mode:
        # 페어드 텍스트는 원본 발화 길이에 맞춰 latent 크기 결정
        # → 생성 길이가 원본과 유사해져 시퀀스 손실 정렬이 안정됨
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

    # [v4-NEW] EMA 기반 조기종료 지표 + 동적 paired_ratio
    smoothed_l3        = None
    current_paired_ratio = paired_ratio

    initial_gap = None
    start_time  = time.time()

    print(f"\n[Dynamic Optimization 구동] 목표 임계치: {threshold}")
    print(f"  speed={speed} | dp_lr_ratio={dp_lr_ratio} | multi_wav_mode={multi_wav_mode}")
    print(f"  hf_weight={hf_weight} | ltas_weight={ltas_weight} | "
          f"style_reg={style_reg_weight}(anchor={style_reg_ref_anchor_weight}) | "
          f"mss={mss_loss_weight}(tol={mss_dur_tolerance})"
          + (f" | seq={seq_loss_weight} | dur={dur_loss_weight} | paired_ratio={paired_ratio}"
             if paired_mode else ""))
    print(f"  학습 구간: step {start_step + 1} → {num_steps}\n")

    for step in range(start_step, num_steps):
        optimizer.zero_grad()

        # DP warmup (페어드 모드에서만 의미 있음)
        if train_dp and not style_dp.requires_grad:
            gate_metric = smoothed_l3 if smoothed_l3 is not None else best_loss
            if gate_metric <= 0.30 or step >= start_step + 50:
                style_dp.requires_grad_(True)
                print(f"\n>>> [DP 활성화] Step {step+1}: L3(EMA)={gate_metric:.4f} → style_dp 학습 시작")

        # ── 샘플 선택: 페어드 vs 일반 ───────────────────────────────────────
        use_paired = (paired_mode and random.random() < current_paired_ratio)

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

        # [NEW-1] LTAS 양방향 스펙트럼 손실
        if ltas_weight > 0:
            loss = loss + ltas_weight * ltas_loss_fn(gen_wav, current_target_ltas)

        # [v5-NEW] 페어드 스텝의 duration 오차율 계산 (MSS 게이팅 + dur_loss 공용)
        dur_ratio_err = None
        if use_paired:
            dur_natural   = dur.squeeze() * speed  # speed로 나눈 값을 원속도로 환원
            dur_ratio_err = (torch.abs(dur_natural - current_dur_target)
                              / max(current_dur_target, 1e-3)).item()

        # [FIX-B] 페어드 dur 손실 → style_dp에 실제 gradient 공급
        if use_paired and train_dp and style_dp.requires_grad and dur_loss_weight > 0:
            dur_loss = ((dur_natural - current_dur_target)
                        / max(current_dur_target, 1e-3)) ** 2
            loss = loss + dur_loss_weight * dur_loss.mean()

        # [v5-NEW] Multi-Scale Spectral Loss — v4g(Gemini) 리뷰 후 게이팅 조건부 차용.
        # duration이 아직 크게 어긋난 상태(학습 초반/dp 미수렴)에서는 STFT 프레임
        # 정렬 자체가 부정확해 노이즈 그래디언트가 될 수 있어, 오차율이
        # mss_dur_tolerance 이내로 수렴했을 때만 활성화한다.
        if (use_paired and mss_loss_fn is not None
                and dur_ratio_err is not None and dur_ratio_err <= mss_dur_tolerance):
            loss = loss + mss_loss_weight * mss_loss_fn(gen_wav, target_wav_ts[pi])

        # [v5-NEW] Style 정규화 (v4g 리뷰 후 조건부 차용)
        # - std 경계(0.06~0.08) 페널티: 가이드 문서 진단 기준을 실제 loss로 반영 (채택)
        # - 레퍼런스 앵커 L2: 기본 가중치 0으로 비활성화. 최적점이 초기 레퍼런스
        #   프리셋에서 먼 경우(카일루스 사례) 매칭을 방해할 위험이 있어 필요할 때만
        #   style_reg_ref_anchor_weight를 config에서 켜서 사용할 것.
        if style_reg_weight > 0:
            current_std = style_ttl.std()
            std_penalty = torch.clamp(0.06 - current_std, min=0) ** 2 \
                        + torch.clamp(current_std - 0.08, min=0) ** 2
            style_reg_loss = std_penalty
            if style_reg_ref_anchor_weight > 0:
                style_reg_loss = style_reg_loss + style_reg_ref_anchor_weight \
                                  * F.mse_loss(style_ttl, ref_style_ttl_frozen)
            loss = loss + style_reg_weight * style_reg_loss

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

        # ── Best 갱신 (일반 텍스트 스텝 기준 — 페어드는 분포가 달라 제외) ──
        if not use_paired:
            # [v4-NEW] EMA는 스텝별 노이즈를 평활해 "운 좋은 샘플 1개"로
            # 웜다운이 조기 발동하는 것을 방지 (카일루스 900스텝 언더핏 사례 대응)
            smoothed_l3 = (primary_loss if smoothed_l3 is None
                           else early_stop_ema_alpha * primary_loss
                                + (1 - early_stop_ema_alpha) * smoothed_l3)
        if (not use_paired) and primary_loss < best_loss:
            best_loss = primary_loss
            best_ttl  = style_ttl.detach().clone()
            best_dp   = style_dp.detach().clone()
        elif use_paired and best_ttl is None:
            # 초반 전부 페어드일 때 best 미설정 방지
            best_ttl = style_ttl.detach().clone()
            best_dp  = style_dp.detach().clone()

        # ── 로그 출력 (Cell 10 파싱 정규식 호환 포맷 고정) ────────────────────
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            ttl_std    = style_ttl.detach().std().item()
            ema_str    = f"{smoothed_l3:.4f}" if smoothed_l3 is not None else "N/A"
            print(f"  Step {step+1}/{num_steps} | Loss(total): {loss.item():.4f} | "
                  f"Loss(L3): {primary_loss:.4f} | LR: {current_lr:.5f} | "
                  f"Best(L3): {best_loss:.4f} | EMA(L3): {ema_str} | "
                  f"ttl_std: {ttl_std:.4f}")
            if ttl_std > 0.09 or ttl_std < 0.05:
                print(f"    [주의] style_ttl std({ttl_std:.4f})가 정상 범위(0.06~0.08) 밖입니다. "
                      f"과적합/붕괴 가능성 — lr 또는 threshold 재검토 권장.")

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
        # [v4] 단일 샘플 best_loss 대신 EMA(smoothed_l3)로 판정 → 조기 락인 방지.
        # EMA가 아직 형성되지 않은 초반 스텝에는 종전처럼 best_loss로 폴백.
        warmdown_gate = smoothed_l3 if smoothed_l3 is not None else best_loss
        if not warmdown_mode and warmdown_gate <= threshold:
            print(f"\n>>> [Phase Transition] EMA(L3) Threshold({threshold}) 도달! "
                  f"(EMA={warmdown_gate:.4f})")
            print(">>> Dynamic Warm-down 모드 활성화 (LR x0.05, 최대 50스텝 추가 정렬)")
            warmdown_mode = True
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.05
            # [v4-NEW] 문서화되어 있었으나 미구현이던 paired_ratio 하향 실제 적용
            if paired_mode and current_paired_ratio > warmdown_paired_ratio:
                print(f">>> [paired_ratio 하향] {current_paired_ratio:.2f} → "
                      f"{warmdown_paired_ratio:.2f} (페어드 과적합 방지)")
                current_paired_ratio = warmdown_paired_ratio

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
    final_ttl_std = (best_ttl if best_ttl is not None else style_ttl).detach().std().item()
    print("\n" + "="*60)
    print("[학습 완료 결과 다차원 평가 리포트]")
    print(f"- 총 학습 소요 시간  : {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"- 최종 학습 진행 스텝: {step + 1} / {num_steps}")
    print(f"- 최소 음색 손실값   : {best_loss:.4f}")
    print(f"- EMA 음색 손실값    : {smoothed_l3:.4f}" if smoothed_l3 is not None else "- EMA 음색 손실값    : N/A")
    print(f"- 최종 style_ttl std : {final_ttl_std:.4f} (정상범위 0.06~0.08)")
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
