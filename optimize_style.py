"""
optimize_style_v18.py
─────────────────────────────────────────────────────────────────────────────
한국 남성 저음(bass) 특화 Supertonic3 스타일 최적화 코드

신규/수정사항 (누적, 최신순):
  [v18]   저역(low-band) 손실에 항목별 가중치(low_band_item_weights) 추가.
          welt12~15 실험에서 특정 항목(예: item2)이 다른 항목과 저역 방향이
          구조적으로 상충하며(item1의 duration 정체와 동일 패턴), weighted
          샘플링으로 그 항목에 그래디언트를 더 줘도 오히려 그 항목 자신과
          무관한 다른 항목까지 같이 악화되는 게 실측으로 확인됨(공유 style_ttl
          벡터로는 만족 불가능한 상충). low_band_item_weights로 특정 항목의
          저역 손실 기여도를 개별적으로 낮춰(0에 가깝게) 그 항목이 다른 항목들의
          저역 매칭을 방해하는 간섭을 줄일 수 있는지 테스트 가능. 길이 불일치/
          미설정 시 전부 1.0으로 폴백 — 기본값에서는 기존과 100% 동일 동작.
  [v17]   HF Xet CDN(us.gcp.cdn.hf.co) 특정 PoP 장애로 microsoft/wavlm-large,
          speechbrain/spkrec-ecapa-voxceleb 다운로드가 403(invalid key pair id)로
          막히는 사고가 실제로 있었음. load_wavlm()/load_ecapa()가 각각
          WAVLM_LOCAL_DIR/ECAPA_LOCAL_DIR(기본값: Drive의 supertonic-3/hf_cache/
          하위) 로컬 사본이 있으면 그걸 우선 쓰고, 없으면 기존과 동일하게 온라인
          다운로드로 폴백하도록 수정(opt-in, 미설정 시 v16 이전과 100% 동일 동작).
  [v16]   페어드 항목(pi) 선택에 opt-in weighted 모드 추가
          (config: paired_sampling_mode="uniform"|"weighted", sampling_explore_ratio).
          기존엔 매 페어드 스텝마다 5개 항목을 완전 균등 샘플링해서, 이미 잘 맞은
          항목과 안 맞은 항목에 동일한 그래디언트 예산을 썼다. weighted 모드는
          duration EMA + 저역(low-band) EMA가 큰 항목에 샘플링 확률을 더 배정한다
          (정체 판정된 항목은 낮은 가중치로 유지, 완전 배제는 안 함). 저역 비율
          오차도 duration과 동일하게 항목별 EMA로 추적하도록 신규 추가— 로그에
          EMA(low_err) 필드, 최종 리포트에 항목별 EMA 진단 노트 추가. LTAS 분포
          계산도 ltas_loss/low_band_loss/EMA 추적이 각각 재계산하던 걸 스텝당
          1회로 통합(rfft 중복 제거). 기본값(uniform)에서는 기존과 100% 동일하게
          동작 — select_paired_index()의 단위 시뮬레이션으로 검증(본문 코드 하단
          self-test 참고).
  [v15]   duration 최적 스냅샷의 step 번호(best_dur_step)도 함께 추적/저장하도록
          추가. 기존엔 EMA 값만 리포트에 찍혀서 몇 스텝에서의 스냅샷인지
          로그로 역추적할 방법이 없었음 — style_ttl 쪽(best_step)과 동일한
          형태로 리포트에 표시.
  [v14_2] duration 최적 스냅샷(best_dp_for_duration) 선정 기준에서 정체
          판정된 항목을 제외하도록 수정. 포함 시 정체 항목의 노이즈성 등락에
          선정 시점이 왜곡되는 문제가 있었음(시뮬레이션으로 확인).
  [v14]   style_ttl과 style_dp를 서로 다른 기준으로 독립 저장하도록 변경.
          기존엔 style_dp도 style_ttl과 동일하게 "음색 손실 최저 시점"에만
          스냅샷됐는데, 그 이후 duration이 계속 개선돼도 저장본에 전혀
          반영되지 않는 문제가 있었음. 이제 "가장 안 풀린 항목의 duration이
          최저였던 시점"을 별도 추적해 저장. 학습 종료 시 실측 검증 리포트도
          실제 저장되는 dp로 합성하도록 통일.
  [v13]   웜다운 duration 게이트에 "정체(stall) 판정" 추가. 특정 항목이 일정
          스텝 동안 충분히 개선되지 않으면(아직 tolerance 초과 상태일 때만)
          게이트 필수조건에서 제외 — 구조적으로 안 풀리는 항목 하나 때문에
          나머지가 다 풀렸는데도 전체 스텝 예산을 낭비하는 것을 방지.
  [v12]   웜다운 duration 게이트가 페어드 항목 각각을 독립된 EMA로 추적하도록
          변경. 이전엔 단일 pooled EMA만 써서, 특정 항목이 안 풀렸어도 최근
          샘플링 운으로 게이트가 잘못 통과될 확률이 유의미하게 높았음
          (몬테카를로 시뮬레이션으로 확인).
  [v11]   웜다운이 음색(WavLM) EMA만 보고 판정되던 것에, duration EMA도 별도
          tolerance 이내여야 통과하는 게이트를 추가. 학습 로그에 EMA(dur_err)
          필드 추가, loss curve 그래프에도 보조축으로 함께 표시.
  [v10]   재개(resume) 시 EMA/best_loss/paired_ratio 상태를 체크포인트에서
          복원하도록 수정 — 이전엔 재개 직후 EMA가 새로 형성되며 조기 웜다운이
          발동할 위험이 있었음. 안전장치로 resume_grace_steps 추가.
  [v9]    웜다운 진입 후에도 EMA가 다시 threshold를 넘는 경우를 감지해 최종
          판정 문구를 구분 표시. 저장되는 체크포인트가 best 스냅샷임을 리포트에
          명시.
  [v8]    학습 종료 시 페어드 타겟별 duration/저역 오차를 실측해 리포트.
  [v7]    저역(100-500Hz) 비율 타겟 손실(low_band_weight) 추가. LTAS 손실만으론
          저역 신호가 여러 밴드에 분산되어 상대적으로 약해지는 문제 대응.
  [v6]    effective dp lr(=lr×dp_lr_ratio)이 지나치게 작으면 style_dp가 사실상
          학습되지 않아 duration이 어긋나는 문제 경고 추가. 웜다운 시 style_dp
          그룹 LR을 style_ttl과 별도 스케일(warmdown_dp_lr_scale)로 조절 가능.
  [v5]    style_reg_weight(std 경계 페널티), Multi-Scale Spectral Loss(안전
          게이팅 포함) 추가.
  [v4]    웜다운 판정을 단일 스텝 손실이 아닌 EMA 기준으로 변경(우연히 쉬운
          샘플 하나로 조기 락인되는 문제 방지). 문서화만 되어있고 미구현이던
          웜다운 시 paired_ratio 하향을 실제로 반영.
  [v3]    style_dp 그래디언트가 실제로 흐르지 않던 버그 수정(언페어드 모드에선
          자동 비활성화). 텍스트별 고정 latent 사전 생성(길이 의존 잔차 제거).
          LTAS 양방향 매칭 손실 추가.
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
# 1b. [v5-NEW] Multi-Scale Spectral Loss (게이팅 조건부 적용)
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
        # [FIX] torch.stft는 win_length <= n_fft를 항상 요구한다. 이전 버전은
        # (n_fft=512, win=600) 같은 조합이 들어가 있어 학습 첫 스텝부터
        # RuntimeError로 죽었다(win_length=600 > n_fft=512). __init__ 시점에
        # 검증해 잘못된 값이 들어오면 학습을 몇 분~몇 시간 돌리지 않고
        # 즉시 명확한 에러로 알려준다.
        assert 0 < win_length <= n_fft, (
            f"win_length({win_length})는 0보다 크고 n_fft({n_fft}) 이하여야 합니다."
        )
        assert 0 < hop_length < win_length, (
            f"hop_length({hop_length})는 0보다 크고 win_length({win_length}) 미만이어야 합니다."
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        if x.ndim > 1:
            x = x.squeeze()
        if y.ndim > 1:
            y = y.squeeze()

        # [FIX] 오디오 길이가 n_fft보다 짧으면(페어드 대상이 극단적으로 짧은 경우)
        # torch.stft가 또 다른 형태로 에러를 낼 수 있어 방어적으로 스킵.
        if x.numel() < self.n_fft or y.numel() < self.n_fft:
            return x.new_zeros(())

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
    # [FIX] 기존 (n_fft, win) 짝이 win>n_fft로 어긋나 있던 버그 수정.
    # ParallelWaveGAN 계열에서 널리 쓰이는 표준 multi-resolution STFT 설정으로 교체.
    # win_length <= n_fft, hop_length < win_length 가 항상 성립하도록 짝을 맞춤.
    def __init__(self, scales=(512, 1024, 2048), hops=(50, 120, 240), wins=(240, 600, 1200)):
        super().__init__()
        self.losses = torch.nn.ModuleList([
            SingleScaleSpectralLoss(n_fft, hop, win)
            for n_fft, hop, win in zip(scales, hops, wins)
        ])

    def forward(self, x, y):
        return sum(loss_fn(x, y) for loss_fn in self.losses) / len(self.losses)


# [v17-NEW] HF Xet CDN(us.gcp.cdn.hf.co) 특정 PoP 장애로 microsoft/wavlm-large,
# speechbrain/spkrec-ecapa-voxceleb 등 HF에서 받는 보조 모델이 403(invalid key
# pair id)로 막히는 사고가 실제로 있었음(2026-07). Supertonic3 본체는 Google Drive
# 사전 배치로 우회했지만, WavLM/ECAPA는 optimize_style.py 내부에서 별도로
# HF Hub/SpeechBrain을 통해 자체 다운로드해서 같은 장애를 다시 맞았다.
# 아래 두 경로에 미리 파일을 받아 넣어두면(Drive 마운트 후 Colab 로컬로 복사되어
# 있는 상태) 그 로컬 사본을 쓰고, 없으면 기존과 동일하게 온라인에서 받는다 —
# 즉 이 우회 기능은 완전히 opt-in이고, 아무것도 안 해두면 v16 이전과 100% 동일.
WAVLM_LOCAL_DIR = os.environ.get(
    "WAVLM_LOCAL_DIR", "/content/drive/MyDrive/supertonic-3/hf_cache/wavlm-large"
)
ECAPA_LOCAL_DIR = os.environ.get(
    "ECAPA_LOCAL_DIR", "/content/drive/MyDrive/supertonic-3/hf_cache/spkrec-ecapa-voxceleb"
)

def load_wavlm():
    from transformers import WavLMModel
    if os.path.isdir(WAVLM_LOCAL_DIR) and os.path.exists(
        os.path.join(WAVLM_LOCAL_DIR, "config.json")
    ):
        print(f"[load_wavlm] 로컬 사본 사용: {WAVLM_LOCAL_DIR} (HF 온라인 다운로드 안 함)")
        model_src = WAVLM_LOCAL_DIR
    else:
        print("[load_wavlm] 로컬 사본 없음 → microsoft/wavlm-large 온라인 다운로드 시도")
        print(f"  (로컬로 우회하려면 {WAVLM_LOCAL_DIR} 에 config.json/"
              f"preprocessor_config.json/pytorch_model.bin을 미리 받아두세요)")
        model_src = 'microsoft/wavlm-large'
    model = WavLMModel.from_pretrained(model_src).to(DEVICE).eval()
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
    # [v17-NEW] load_wavlm()과 동일한 이유로 로컬 Drive 사본 우선 사용.
    # speechbrain은 source가 로컬 디렉토리면 그 안의 파일을 그대로 쓰고
    # 네트워크를 전혀 타지 않는다(hyperparams.yaml/embedding_model.ckpt/
    # classifier.ckpt/mean_var_norm_emb.ckpt/label_encoder.txt가 갖춰져 있어야 함).
    if os.path.isdir(ECAPA_LOCAL_DIR) and os.path.exists(
        os.path.join(ECAPA_LOCAL_DIR, "hyperparams.yaml")
    ):
        print(f"[load_ecapa] 로컬 사본 사용: {ECAPA_LOCAL_DIR} (HF 온라인 다운로드 안 함)")
        ecapa_source = ECAPA_LOCAL_DIR
    else:
        print("[load_ecapa] 로컬 사본 없음 → speechbrain/spkrec-ecapa-voxceleb 온라인 다운로드 시도")
        print(f"  (로컬로 우회하려면 {ECAPA_LOCAL_DIR} 에 hyperparams.yaml/embedding_model.ckpt/"
              f"classifier.ckpt/mean_var_norm_emb.ckpt/label_encoder.txt를 미리 받아두세요)")
        ecapa_source = "speechbrain/spkrec-ecapa-voxceleb"
    try:
        classifier = EncoderClassifier.from_hparams(
            source=ecapa_source,
            savedir="pretrained_models/ecapa",
            run_opts={"device": str(DEVICE)}
        )
    except Exception as e:
        print(f"[load_ecapa] 로드 실패, ECAPA 손실 없이 진행합니다: {e}")
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

def ltas_loss_from_dist(gen_ltas, target_ltas):
    """[v16-NEW] log 분포 L1. gen_ltas를 이미 계산해둔 경우 rfft 재계산 없이 사용."""
    return F.l1_loss(torch.log10(gen_ltas + 1e-8),
                     torch.log10(target_ltas + 1e-8))

def ltas_loss_fn(gen_wav, target_ltas):
    """log 분포 L1 — 원본 대비 밝음(+)/탁함(-) 양방향 모두 패널티."""
    return ltas_loss_from_dist(compute_ltas_distribution(gen_wav), target_ltas)


# [v7-NEW] 저역(중후함/가슴공명대) 타겟 손실.
# ─────────────────────────────────────────────────────────────────────────────
# 음향 분석에서 F0/centroid는 원본과 근접했지만 실제 청취 시 "가볍다"는
# 인상이 확인됨. 원인 분석 결과 centroid(가중평균 하나의 스칼라)는 저역 비율의
# 미세한 변화나 음량 차이로 인한 청감 효과를 못 잡는다는 게 확인됨. LTAS 손실
# (24밴드 로그분포 L1)은 전체 분포 형태를 맞추려 하지만, 24개 밴드에 걸쳐
# gradient가 분산되므로 특정 대역(저역)에 대한 신호가 상대적으로 약해질 수 있음.
# → LTAS와 동일한 24밴드 분포(compute_ltas_distribution)에서 100~500Hz에 해당하는
#   밴드(인덱스 1~8, 실측 98.2~506.9Hz)의 부분합만 뽑아 그 비율을 직접 맞추는
#   보조 손실을 추가. LTAS 계산을 그대로 재사용하므로 추가 STFT/rfft 없이 거의
#   공짜로 계산됨(학습 속도에 영향 거의 없음).
LOW_BAND_IDX = list(range(1, 9))  # 98.2~506.9Hz, 100-500Hz 목표 범위와 거의 정확히 일치

def low_band_ratio(ltas_dist):
    """[N_LTAS_BANDS] 정규화 분포 → 저역(100-500Hz) 에너지 비율 스칼라."""
    return ltas_dist[LOW_BAND_IDX].sum()

def low_band_loss_from_dist(gen_ltas, target_ltas):
    """[v16-NEW] gen_ltas를 이미 계산해둔 경우 재사용. (loss, gen_low, tgt_low) 반환 —
    gen_low/tgt_low는 항목별 저역 오차 EMA 추적 및 weighted 샘플링에도 재사용된다."""
    gen_low = low_band_ratio(gen_ltas)
    tgt_low = low_band_ratio(target_ltas)
    return F.mse_loss(gen_low, tgt_low), gen_low, tgt_low

def low_band_loss_fn(gen_wav, target_ltas):
    """생성 음성과 타겟의 저역 비율 차이를 직접 패널티. LTAS 분포를 재사용."""
    loss, _, _ = low_band_loss_from_dist(compute_ltas_distribution(gen_wav), target_ltas)
    return loss


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

def save_style(path, style_ttl, style_dp, source_file=None, train_state=None):
    source_meta = (list(source_file)
                   if isinstance(source_file, (list, tuple))
                   else (source_file or "unknown"))
    metadata = {
        "source_file":        source_meta,
        "source_sample_rate": 44100,
        "target_sample_rate": 44100,
        "extracted_at":       datetime.now().isoformat()
    }
    # [v10-NEW] 재개(resume) 시 EMA(smoothed_l3)와 best_loss가 리셋되어, 재개
    # 직후 첫 unpaired 샘플 손실이 그대로 초기 EMA가 되면서 "운 좋은 샘플 1개"로
    # 즉시 웜다운이 발동하는 문제가 실제로 관측됨(재개 직후 9스텝만에 웜다운
    # 진입, duration 오차 21.5%로 재개 세션이 사실상 학습을 못 함). train_state를
    # 체크포인트에 함께 저장해 재개 시 EMA를 정확히 이어받도록 한다.
    if train_state is not None:
        metadata["train_state"] = train_state

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
        "metadata": metadata
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
# [v16-NEW] 페어드 항목 샘플링 (uniform vs weighted)
# ─────────────────────────────────────────────────────────────────────────────
# 배경(welt12~welt14 실측): 저역(100-500Hz) 비율 오차가 item마다 반대 방향으로
# 벌어짐(예: item[1]은 저역 부족(-), item[2]는 저역 과다(+)). style_ttl은 5개
# 페어드 텍스트가 공유하는 단일 벡터라, low_band_weight를 아무리 올려도 상충하는
# 두 방향을 동시에 만족시킬 수 없고 압력만 세지는 현상이 4번의 학습(low_band_weight
# 0.1/0.12, dp_lr_ratio 0.045/0.06)에서 반복 확인됨. 기존 `random.randrange`
# 균등 샘플링은 이미 잘 맞은 항목과 안 맞은 항목에 매 스텝 동일한 그래디언트
# 예산을 쓰므로, 안 풀린 항목에 상대적으로 더 많은 스텝을 배정하는 실험용 옵션을
# 추가한다(opt-in, 기본은 기존과 동일한 uniform).
def select_paired_index(
    n_items, dur_err_ema, low_band_err_ema, stalled_items,
    mode="uniform", explore_ratio=0.2, stalled_weight=0.3, rng=None
):
    """다음 페어드 스텝에서 학습할 항목 인덱스를 고른다.

    mode="uniform": 기존과 완전히 동일한 random.randrange(n_items) 동작
                     (하위 호환, 기본값).
    mode="weighted": 아직 한 번도 관측 안 된 항목이 있으면 그 항목을 우선
                     선택(coverage-first, 초반 EMA 형성 보장). 이후에는
                     explore_ratio 확률로 무작위 탐색을 섞고, 나머지 확률은
                     "duration EMA + 저역 EMA를 각각 항목 평균으로 정규화해
                     합산한 가중치"에 비례해 선택한다. dur_gate에서 이미
                     정체(stalled)로 판정된 항목은 stalled_weight로 낮춰서
                     완전히 배제하진 않되(재정체 여부 재확인 여지는 남김)
                     예산을 과도하게 뺏기지 않게 한다.

    rng: random.Random 인스턴스(테스트/시뮬레이션에서 결정론적 검증을 위해
         주입 가능). None이면 전역 random 모듈 사용.
    """
    _r = rng if rng is not None else random

    if mode != "weighted":
        return _r.randrange(n_items)

    dur_err_ema = dur_err_ema or {}
    low_band_err_ema = low_band_err_ema or {}
    stalled_items = stalled_items or set()

    unobserved = [i for i in range(n_items)
                  if i not in dur_err_ema and i not in low_band_err_ema]
    if unobserved:
        return _r.choice(unobserved)

    if _r.random() < explore_ratio:
        return _r.randrange(n_items)

    dur_vals = [dur_err_ema.get(i) for i in range(n_items)]
    low_vals = [low_band_err_ema.get(i) for i in range(n_items)]
    dur_obs  = [v for v in dur_vals if v is not None]
    low_obs  = [v for v in low_vals if v is not None]
    dur_mean = (sum(dur_obs) / len(dur_obs)) if dur_obs else 1.0
    low_mean = (sum(low_obs) / len(low_obs)) if low_obs else 1.0
    dur_mean = dur_mean if dur_mean > 1e-6 else 1.0
    low_mean = low_mean if low_mean > 1e-6 else 1.0

    weights = []
    for i in range(n_items):
        if i in stalled_items:
            weights.append(stalled_weight)
            continue
        dn = (dur_vals[i] / dur_mean) if dur_vals[i] is not None else 1.0
        ln = (low_vals[i] / low_mean) if low_vals[i] is not None else 1.0
        weights.append(max(0.05, 0.5 * dn + 0.5 * ln))

    total = sum(weights)
    threshold_r = _r.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if threshold_r <= acc:
            return i
    return n_items - 1  # 부동소수점 오차로 못 걸릴 경우의 폴백


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
    # [v11-NEW] dp_lr_ratio가 이미 권장 범위(0.03)여도, 타겟 blend가
    # 좋아 style_ttl(음색) EMA가 duration보다 훨씬 빨리 threshold에 도달하면
    # 웜다운이 먼저 걸려 style_dp가 전체 num_steps 예산을 다 못 쓰고 조기 종료됨
    # (실측: 8000 예산 중 3693 스텝에서 종료, duration 평균 오차 21.3%로 방치).
    # None(기본값)이면 기존과 동일하게 음색 EMA만으로 판정(하위 호환). 값을
    # 넣으면(예: 0.15) duration EMA도 이 오차율 이내여야 웜다운이 걸림.
    dur_gate_tolerance = cfg.get("early_stop_dur_gate_tolerance", None)
    # [v13-NEW] 정체 판정 파라미터. patience=None이면 정체 판정 자체를 끔(하위호환 —
    # v12와 동일하게 모든 항목이 무조건 tolerance를 만족해야 함).
    dur_gate_stall_patience_steps  = cfg.get("dur_gate_stall_patience_steps", None)
    dur_gate_stall_min_improvement = cfg.get("dur_gate_stall_min_improvement", 0.05)
    use_ecapa     = cfg.get("use_ecapa_loss", True)
    # [v6-NEW] dp_lr_ratio가 지나치게 작으면 style_dp가 사실상 학습되지 않아
    # duration(발화 길이) 매칭이 실패한다. dp_lr_ratio=0.001로 설정했던 사례에서
    # effective lr이 1e-7대까지 떨어져 합성 음성이 원본보다 최대 25% 길어지는
    # 문제가 실제로 관측되어 경고를 추가한다.
    _DP_LR_RATIO_WARN_THRESHOLD = 0.005
    if train_dp and dp_lr_ratio < _DP_LR_RATIO_WARN_THRESHOLD:
        print(f"[경고] dp_lr_ratio={dp_lr_ratio}가 매우 작습니다 "
              f"(effective dp lr = {lr * dp_lr_ratio:.2e}). style_dp가 사실상 학습되지 "
              f"않아 duration(발화 속도/길이) 매칭이 실패할 수 있습니다. "
              f"0.01~0.03 정도를 권장합니다.")
    ecapa_weight  = cfg.get("ecapa_loss_weight", 0.3)

    # [FIX-A] hf_weight config 연동 + 신규 손실 가중치들
    # [v4] ltas_weight 기본값을 0.5→0.2로 조정: 가이드 문서(9장)에서
    #      0.5는 과도한 저음 유도로 먹먹함을 유발한다고 자체 확인됨. 명시적으로
    #      config에 값을 넣으면 그 값이 우선하므로 기존 실험 재현성은 유지됨.
    hf_weight       = cfg.get("hf_weight", 0.05)
    ltas_weight     = cfg.get("ltas_weight", 0.2)
    # [v7-NEW] 저역(100-500Hz, 중후함/가슴공명대) 타겟 손실 가중치.
    # LTAS(24밴드 전체 분포) 손실과 별개로, 저역 비율만 직접 맞추는 보조 손실.
    # ltas_weight보다 작게 시작하는 걸 권장(과하면 저역만 밀어붙이다 다른 대역
    # 밸런스가 깨질 수 있음). 기본 0.1, 0=끔.
    low_band_weight = cfg.get("low_band_weight", 0.1)
    # [v18-NEW] welt12~15 4번의 실험(low_band_weight 0.1/0.12, dp_lr_ratio 0.045/0.06,
    # paired_sampling_mode uniform/weighted) 전부에서 item2의 저역 오차가 단 한 번도
    # +11.9%p 밑으로 내려간 적이 없고, weighted 샘플링으로 item2에 그래디언트를
    # 더 줬을 땐 오히려 item2 자신(+17.9%p)과 무관한 item4(-10.0%p)까지 같이
    # 악화됨 — 공유 style_ttl 벡터로는 만족 불가능한 상충(item1의 duration과 동일
    # 패턴)으로 사실상 확정. low_band_item_weights로 특정 항목의 저역 손실
    # 기여도를 개별적으로 낮추면(0에 가깝게), 그 항목이 다른 항목들의 저역 매칭을
    # 방해하는 그래디언트 간섭을 줄일 수 있는지 테스트 가능. 길이는 페어드
    # 항목 수와 같아야 하며, None(기본값)이면 전부 1.0(기존과 100% 동일 동작).
    # 예: item2(인덱스1)만 죽이려면 [1.0, 0.1, 1.0, 1.0, 1.0].
    low_band_item_weights = cfg.get("low_band_item_weights", None)
    seq_loss_weight = cfg.get("seq_loss_weight", 0.3)
    dur_loss_weight = cfg.get("dur_loss_weight", 0.3)
    paired_ratio    = cfg.get("paired_ratio", 0.7)
    # [v16-NEW] 페어드 항목(pi) 선택 방식. "uniform"(기본, 기존과 완전 동일 동작)
    # | "weighted"(실험적, opt-in): 아직 tolerance를 못 채운/오차가 큰 항목에
    # 상대적으로 더 많은 스텝을 배정. select_paired_index() 참고.
    paired_sampling_mode  = cfg.get("paired_sampling_mode", "uniform")
    sampling_explore_ratio = cfg.get("sampling_explore_ratio", 0.2)
    if paired_sampling_mode not in ("uniform", "weighted"):
        print(f"[경고] paired_sampling_mode='{paired_sampling_mode}'는 알 수 없는 값입니다. "
              f"'uniform'으로 폴백합니다.")
        paired_sampling_mode = "uniform"
    # [v4-NEW] 웜다운 진입 시 하향할 paired_ratio 상한 (문서에는 있었으나
    # 실제 루프에 미반영이었던 기능을 구현)
    warmdown_paired_ratio = cfg.get("warmdown_paired_ratio", 0.3)
    # [v4-NEW] 조기종료 판정용 EMA 평활 계수. 값이 작을수록 더 많은 스텝의
    # 평균을 반영해 "운 좋은 샘플 1개"로 인한 조기 락인을 방지한다.
    early_stop_ema_alpha = cfg.get("early_stop_ema_alpha", 0.08)
    # [v6-NEW] 웜다운 시 style_dp 그룹에 적용할 별도 LR 스케일.
    # 기존 버그: 웜다운이 모든 param_group(style_ttl, style_dp)의 LR을 동일하게
    # x0.05로 낮췄는데, style_dp는 이미 dp_lr_ratio로 낮게 잡혀 있어 웜다운
    # 진입 시점에 duration 매칭이 아직 안 끝났어도 그대로 사실상 동결되어버림
    # (합성이 원본보다 최대 25% 길어지는 원인이 된 사례가 있었음). 기본값 1.0 = 웜다운 때도
    # style_dp는 낮추지 않고 style_ttl만 낮춰서, duration이 웜다운 구간
    # (+최대 50스텝)에서도 계속 개선될 여지를 남긴다.
    warmdown_dp_lr_scale = cfg.get("warmdown_dp_lr_scale", 1.0)

    # [v5-NEW] 검토 후 조건부 적용
    # - style_reg_weight: std 0.06~0.08 경계 이탈 시 페널티
    # - style_reg_ref_anchor_weight: 초기 레퍼런스로의 L2 앵커. 기본 0(끔).
    #   최적점이 레퍼런스에서 먼 경우 매칭을 방해할 수 있어
    #   기본 비활성화 — 과적합/스타일 발산이 실제로 관측될 때만 켤 것.
    style_reg_weight            = cfg.get("style_reg_weight", 0.05)
    style_reg_ref_anchor_weight = cfg.get("style_reg_ref_anchor_weight", 0.0)
    # - mss_loss_weight: Multi-Scale Spectral Loss. 기본 0(끔). duration이
    #   mss_dur_tolerance 이내로 수렴한 페어드 스텝에서만 게이팅 적용(원래는
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

    # [v18-NEW] low_band_item_weights 길이/값 검증. 길이가 안 맞거나 미설정이면
    # 전부 1.0(기존과 100% 동일 동작)으로 폴백 — 조용히 틀린 인덱스로 적용되는
    # 사고를 막기 위해 불일치 시 명확히 경고하고 폴백한다.
    n_paired = len(paired_text_inputs)
    if low_band_item_weights is not None:
        if not paired_mode or len(low_band_item_weights) != n_paired:
            print(f"[경고] low_band_item_weights 길이({len(low_band_item_weights)})가 "
                  f"페어드 항목 수({n_paired})와 다르거나 페어드 모드가 아닙니다. "
                  f"전부 1.0(기존 동작)으로 폴백합니다.")
            low_band_item_weights = [1.0] * n_paired
        else:
            print(f"[알림] low_band_item_weights 적용됨: {low_band_item_weights} "
                  f"(1.0=정상 반영, 0에 가까울수록 그 항목의 저역 손실 기여도를 낮춤)")
    else:
        low_band_item_weights = [1.0] * n_paired

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
    # [v10-NEW] 재개 시 학습 상태(EMA/best_loss/paired_ratio)도 함께 복원 시도.
    # load_voice_style()은 helper.py의 외부 함수라 metadata까지 파싱해준다는
    # 보장이 없으므로, 체크포인트 JSON을 직접 한 번 더 읽어 train_state만 뽑는다.
    # 구버전 체크포인트(이 필드가 없던 시절 저장분)는 자동
    # 폴백 처리되어 None으로 남고, 아래 resume_grace_steps 안전장치가 대신 작동한다.
    resumed_train_state = None
    if latest_ckpt:
        resumed = load_voice_style(latest_ckpt)
        style_ttl = torch.tensor(resumed.ttl, dtype=torch.float32).to(DEVICE).clone().requires_grad_(True)
        style_dp  = torch.tensor(resumed.dp,  dtype=torch.float32).to(DEVICE).clone().requires_grad_(train_dp)
        print(f"[Resume] style 벡터를 체크포인트에서 복원했습니다.")
        try:
            with open(latest_ckpt, "r", encoding="utf-8") as f:
                _ckpt_raw = json.load(f)
            resumed_train_state = _ckpt_raw.get("metadata", {}).get("train_state")
        except Exception:
            resumed_train_state = None
        if resumed_train_state:
            print(f"[Resume] 학습 상태(EMA={resumed_train_state.get('smoothed_l3'):.4f}, "
                  f"best_loss={resumed_train_state.get('best_loss'):.4f})도 복원했습니다.")
        else:
            print(f"[Resume] 이 체크포인트엔 학습 상태 기록이 없습니다(구버전 저장분). "
                  f"EMA가 새로 형성될 때까지 웜다운 조기 발동 방지용 유예 구간을 적용합니다.")
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

    # [v9-FIX] v8-FIX가 KeyError는 없앴지만 불완전했음: CosineAnnealingLR을
    # last_epoch>=0으로 "생성"만 해서는 param_groups['lr']이 즉시 코사인 곡선값으로
    # 갱신되지 않는다(PyTorch 내부 동작 — 다음 .step() 호출 시에야 "직전 lr 기준
    # 재귀 공식"을 적용하는데, 이때 직전 lr이 initial_lr(=원본 시작값)로 잘못
    # 세팅되어 있으면 재귀 공식 자체가 완전히 틀린 값을 계산한다). 실측 결과 step
    # 5300 재개 시 실제로는 lr≈3.16e-5여야 하는데 v8-FIX는 lr=1.2e-4(원본 시작값
    # 그대로)로 잘못 시작해, 이미 수렴 근처인 스타일을 큰 lr로 갑자기 흔들 위험이
    # 있었다. scheduler._get_closed_form_lr()로 재개 시점의 정확한 코사인값을
    # 명시적으로 계산해 즉시 반영한다(처음부터 연속으로 돌린 경우와 값이 정확히
    # 일치함을 시뮬레이션으로 검증함).
    if start_step > 0:
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=lr * 0.01,
        last_epoch=(start_step - 1) if start_step > 0 else -1
    )

    if start_step > 0:
        for pg, closed_lr in zip(optimizer.param_groups, scheduler._get_closed_form_lr()):
            pg['lr'] = closed_lr

    best_loss      = float('inf')
    best_ttl       = None
    best_step      = None
    best_dp        = style_dp.detach().clone()
    warmdown_mode  = False
    warmdown_steps = 0
    MAX_WARMDOWN   = 50

    # [v4-NEW] EMA 기반 조기종료 지표 + 동적 paired_ratio
    smoothed_l3        = None
    # [v11-NEW] duration EMA — 아래 dur_gate_tolerance 설명 참고.
    # [v12-FIX] 실측 사례: 특정 항목이 45.9% 오차로 전혀 안 풀렸는데도 855스텝만에
    # 웜다운이 발동함. 원인 확인(몬테카를로 시뮬레이션): pi가 매 페어드 스텝마다
    # random.randrange(5)로 균등 샘플링되고 EMA alpha=0.08(유효 기억 폭 ~12.5
    # 페어드 스텝)이라, "최근에 우연히 쉬운 항목만 뽑히는 스트릭"만으로 pooled EMA가
    # tolerance 밑으로 떨어질 확률이 45%(!)에 달함 — 사실상 동전 던지기 수준으로
    # 게이트가 속을 수 있었음. 항목별로 따로 EMA를 추적해 "가장 안 풀린 항목"
    # 기준으로 게이트를 걸면 이 오탐이 시뮬레이션상 0%로 떨어짐(검증 완료).
    # smoothed_dur_err는 하위호환/로그용으로 유지하되, 이제부터 그 값은 "pooled
    # 평균"이 아니라 "5개 항목 중 가장 안 풀린 항목의 EMA"를 담는다.
    smoothed_dur_err   = None
    dur_err_ema_per_item = {}   # {pi: EMA} — 항목별 개별 추적
    # [v16-NEW] 저역(low-band) 비율 오차도 duration과 동일한 방식으로 항목별
    # EMA(절대 %p)를 추적. weighted 샘플링의 입력이자, 로그/최종 리포트에서
    # "학습 내내 특정 항목만 저역이 안 맞았는지"를 진단하는 데 쓰인다. 값 자체가
    # 웜다운 게이트 판정에는 아직 관여하지 않는다(기존 duration 게이트 동작을
    # 바꾸지 않기 위한 의도적 스코프 제한 — 필요해지면 별도 tolerance로 확장).
    low_band_err_ema_per_item = {}   # {pi: EMA(|gen-tgt|*100, %p 단위)}
    # [v13-NEW] 실측 사례: 가장 빠른 원본 발화에 해당하는 항목의 duration EMA가
    # 0.4653(step 10) -> 0.4306(step 7360)로, 7350스텝 동안 누적 개선률이 겨우
    # 7.46%에 불과했다(선형 외삽해도 0.19 도달까지 수만 스텝 이상 필요 — 사실상
    # 도달 불가능한 구조적 한계로 판단). "모든 항목이 tolerance를 만족해야 함"
    # 규칙은 이런 구조적 outlier 항목이 하나만 있어도 나머지 4개가 이미 다 풀렸는데도
    # num_steps 예산을 통째로 낭비하게 만든다. stall_patience_steps 이상 지났는데
    # stall_min_improvement 이상 개선이 없는 항목은 "정체(stalled)"로 판정해
    # 게이트 필수 조건에서 제외한다(단, 학습/보고에서는 계속 포함됨 — gate만 제외).
    dur_err_stall_baseline = {}   # {pi: (baseline_ema, baseline_step)}
    stalled_items = set()
    # [v14-NEW] 지금까지 best_dp는 best_ttl과 완전히 같은 조건(timbre primary_loss
    # 최저점)에서만 스냅샷됐다 — duration 품질은 dp 선택 기준에 전혀 반영 안 됐다.
    # 즉 timbre가 이른 스텝(예: 1500)에서 최저점을 찍고 그 뒤로 안 움직여도, dp는
    # num_steps까지 계속 개선되는 것과 무관하게 그 이른 시점의 dp가 저장됐을 것이다
    # (어떤 사례는 우연히 best_step이 전체 스텝의 끝자락이라 문제가 안 드러났을
    # 뿐, 일반적으로 보장되는 동작이 아니다). style_ttl과 style_dp는 저장 시
    # 완전히 독립된 필드이므로, 이제 dp는 "가장 안 풀린 항목의 duration EMA가
    # 최저였던 시점"을 별도로 추적해 그 스냅샷을 최종 저장에 사용한다.
    best_dur_err          = float('inf')
    best_dur_step         = None
    best_dp_for_duration  = None
    current_paired_ratio = paired_ratio

    # [v10-NEW] 재개 시 학습 상태 복원 (신버전 체크포인트만 해당).
    # 이걸로 EMA가 재개 직후 첫 몇 샘플만 보고 즉시 웜다운을 발동시키는 문제를
    # 근본적으로 막는다 — EMA가 "이어지는" 값이므로 재개해도 진짜 대표성 있는
    # 평균 상태에서 다시 시작한다.
    if resumed_train_state:
        smoothed_l3 = resumed_train_state.get("smoothed_l3", None)
        smoothed_dur_err = resumed_train_state.get("smoothed_dur_err", None)
        dur_err_ema_per_item = resumed_train_state.get("dur_err_ema_per_item", {}) or {}
        # JSON은 dict key를 문자열로 저장하므로 정수 pi로 되돌린다.
        dur_err_ema_per_item = {int(k): v for k, v in dur_err_ema_per_item.items()}
        # [v16-NEW] 구버전 체크포인트(이 필드 없음)로 재개해도 그냥 빈 dict로
        # 시작 — weighted 샘플링은 coverage-first 로직이 있어 안전하게 재형성됨.
        low_band_err_ema_per_item = resumed_train_state.get("low_band_err_ema_per_item", {}) or {}
        low_band_err_ema_per_item = {int(k): v for k, v in low_band_err_ema_per_item.items()}
        stalled_items = set(resumed_train_state.get("stalled_items", []) or [])
        _raw_baseline = resumed_train_state.get("dur_err_stall_baseline", {}) or {}
        dur_err_stall_baseline = {int(k): tuple(v) for k, v in _raw_baseline.items()}
        best_loss   = resumed_train_state.get("best_loss", best_loss)
        best_dur_err = resumed_train_state.get("best_dur_err", best_dur_err)
        best_dur_step = resumed_train_state.get("best_dur_step", best_dur_step)
        current_paired_ratio = resumed_train_state.get("current_paired_ratio", paired_ratio)

    # [v10-NEW] 구버전 체크포인트(train_state 없음)로 재개할 때의 안전장치.
    # EMA가 None에서 새로 형성되는 구간에는 웜다운이 발동하지 못하게 유예를 둔다.
    # 이 유예는 신버전 체크포인트(train_state 복원됨)에는 적용되지 않는다
    # (이미 정확한 EMA를 이어받았으므로 불필요).
    resume_grace_steps = cfg.get("resume_grace_steps", 200)
    needs_resume_grace  = (start_step > 0) and (resumed_train_state is None)

    initial_gap = None
    start_time  = time.time()

    print(f"\n[Dynamic Optimization 구동] 목표 임계치: {threshold}")
    print(f"  speed={speed} | dp_lr_ratio={dp_lr_ratio} | multi_wav_mode={multi_wav_mode}")
    print(f"  hf_weight={hf_weight} | ltas_weight={ltas_weight} | low_band_weight={low_band_weight} | "
          f"style_reg={style_reg_weight}(anchor={style_reg_ref_anchor_weight}) | "
          f"mss={mss_loss_weight}(tol={mss_dur_tolerance})"
          + (f" | seq={seq_loss_weight} | dur={dur_loss_weight} | paired_ratio={paired_ratio}"
             f" | paired_sampling={paired_sampling_mode}"
             + (f"(explore={sampling_explore_ratio})" if paired_sampling_mode == "weighted" else "")
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
            # [v16-NEW] paired_sampling_mode="uniform"(기본)이면
            # random.randrange(len(paired_text_inputs))와 완전히 동일하게 동작
            # (select_paired_index 내부에서 mode!="weighted"면 그대로 위임).
            pi = select_paired_index(
                len(paired_text_inputs), dur_err_ema_per_item, low_band_err_ema_per_item,
                stalled_items, mode=paired_sampling_mode,
                explore_ratio=sampling_explore_ratio
            )
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

        # [v16-NEW] LTAS 분포는 rfft가 들어가는 연산이라, ltas_loss/low_band_loss/
        # 항목별 저역 EMA 추적(weighted 샘플링용)이 각자 다시 계산하면 스텝당 최대
        # 2회 중복 호출됐다. 이번에 한 번만 계산해 전부 재사용하도록 통합.
        need_ltas_dist = (ltas_weight > 0 or low_band_weight > 0
                           or (use_paired and paired_sampling_mode == "weighted"))
        gen_ltas = compute_ltas_distribution(gen_wav) if need_ltas_dist else None

        # [NEW-1] LTAS 양방향 스펙트럼 손실
        if ltas_weight > 0:
            loss = loss + ltas_weight * ltas_loss_from_dist(gen_ltas, current_target_ltas)

        # [v7-NEW] 저역(100-500Hz) 비율 타겟 손실. LTAS 분포 계산을 재사용하므로
        # 추가 STFT 없이 거의 무비용. "가슴 공명대"만 별도로 붙잡아 LTAS의
        # 24밴드 평균화로 희석되는 저역 신호를 보강한다.
        gen_low_t, tgt_low_t = None, None
        if low_band_weight > 0:
            lb_loss, gen_low_t, tgt_low_t = low_band_loss_from_dist(gen_ltas, current_target_ltas)
            # [v18-NEW] 페어드 스텝이면 항목별 저역 손실 가중치를 곱한다 — 구조적으로
            # 다른 항목과 상충하는 항목(예: item2)의 그래디언트 간섭을 낮추기 위한
            # opt-in 레버. unpaired 스텝(평균 타겟이라 항목 구분 자체가 없음)은 항상 1.0.
            item_mult = low_band_item_weights[pi] if use_paired else 1.0
            loss = loss + low_band_weight * item_mult * lb_loss
        elif use_paired and paired_sampling_mode == "weighted" and gen_ltas is not None:
            # [v16-NEW] low_band_weight=0(손실로는 안 씀)이어도 weighted 샘플링에
            # 필요한 저역 오차 신호만 그래디언트 없이 계산해둔다.
            with torch.no_grad():
                gen_low_t = low_band_ratio(gen_ltas)
                tgt_low_t = low_band_ratio(current_target_ltas)

        # [v5-NEW] 페어드 스텝의 duration 오차율 계산 (MSS 게이팅 + dur_loss 공용)
        dur_ratio_err = None
        if use_paired:
            dur_natural   = dur.squeeze() * speed  # speed로 나눈 값을 원속도로 환원
            dur_ratio_err = (torch.abs(dur_natural - current_dur_target)
                              / max(current_dur_target, 1e-3)).item()
            # [v12-FIX] pooled EMA 대신 항목별(pi) EMA를 따로 갱신. dur_gate_tolerance가
            # 설정된 경우 "가장 안 풀린 항목" 기준으로 웜다운을 판정해, 특정 항목이
            # 계속 안 풀렸는데도 최근 샘플링 운으로 게이트가 통과되는 문제를 막는다.
            prev = dur_err_ema_per_item.get(pi)
            dur_err_ema_per_item[pi] = (dur_ratio_err if prev is None
                                         else early_stop_ema_alpha * dur_ratio_err
                                              + (1 - early_stop_ema_alpha) * prev)
            # smoothed_dur_err = 지금까지 관측된 항목들 중 가장 안 풀린(최댓값) EMA
            # (진단/로그/플롯 표시용 — stalled 항목도 포함해 "전체 중 최악"을 보여줌).
            smoothed_dur_err = max(dur_err_ema_per_item.values())

            # [v16-NEW] 저역(low-band) 비율 오차도 duration과 동일한 항목별 EMA로
            # 추적. gen_low_t/tgt_low_t는 위 loss 계산 구간에서 low_band_weight>0
            # 이거나 weighted 샘플링일 때만 채워진다 — 둘 다 아니면 None이라 스킵.
            if gen_low_t is not None:
                low_err_pp = abs((gen_low_t.detach() - tgt_low_t.detach()).item()) * 100.0
                prev_lb = low_band_err_ema_per_item.get(pi)
                low_band_err_ema_per_item[pi] = (low_err_pp if prev_lb is None
                                                  else early_stop_ema_alpha * low_err_pp
                                                       + (1 - early_stop_ema_alpha) * prev_lb)

            # [v14_2-FIX] v14 최초 버전은 위 smoothed_dur_err(전체 max, stalled 포함)를
            # best_dp 선정 기준으로 그대로 썼는데, item[1]처럼 영구 정체된 항목이 있으면
            # 이 max가 거의 항상 그 항목에 종속돼 실제로는 item2~5의 개선과 무관하게
            # (item[1] 자체의 미세한 노이즈성 등락만으로) best_dp가 갱신되는 문제가
            # 있었다. 시뮬레이션 검증: 이 상태에서 마지막 best_dp 갱신 시점이 전체 max
            # 기준 step 7887, stalled 제외 기준 step 6169로 서로 다르게 나옴 — 전체
            # max 기준은 "진짜 좋아진 시점"이 아니라 "item[1]이 우연히 살짝 내려간
            # 시점"을 고를 수 있다는 뜻. 게이트(dur_gate_ok)가 이미 stalled_items를
            # 제외하고 판정하는 것과 동일한 기준으로 best_dp도 선정하도록 통일한다.
            _non_stalled_vals = [v for p, v in dur_err_ema_per_item.items()
                                  if p not in stalled_items]
            required_dur_metric = max(_non_stalled_vals) if _non_stalled_vals else None

            # [v14-NEW] duration 전용 best dp 스냅샷. dur_gate 오탐 방지 때와 같은
            # 이유로, 전 항목이 최소 1번씩 관측되기 전엔 갱신하지 않는다(안 그러면
            # 초반에 우연히 쉬운 항목만 뽑혀 "가짜 best"가 찍힐 수 있음).
            n_paired_total = len(paired_text_inputs) if paired_text_inputs else 0
            if (len(dur_err_ema_per_item) >= n_paired_total > 0
                    and required_dur_metric is not None
                    and required_dur_metric < best_dur_err):
                best_dur_err = required_dur_metric
                best_dp_for_duration = style_dp.detach().clone()
                best_dur_step = step + 1

            # [v13-NEW] 정체 판정. patience 설정 시에만 동작(하위호환). 실측 사례에서
            # item[1] EMA가 7350스텝 동안 7.46%밖에 개선 안 된 걸 실측 — 구조적
            # outlier 항목 하나 때문에 나머지가 다 풀렸어도 num_steps 전체를
            # 낭비하는 걸 방지.
            if dur_gate_stall_patience_steps is not None and pi not in stalled_items:
                if pi not in dur_err_stall_baseline:
                    dur_err_stall_baseline[pi] = (dur_err_ema_per_item[pi], step)
                else:
                    base_val, base_step = dur_err_stall_baseline[pi]
                    if (step - base_step) >= dur_gate_stall_patience_steps:
                        rel_improve = ((base_val - dur_err_ema_per_item[pi])
                                        / max(base_val, 1e-6))
                        # 이미 tolerance 이내로 수렴한 항목은 "정체"라고 부를 필요가
                        # 없다 (그냥 다 풀려서 개선폭이 작아진 것뿐) — tolerance
                        # 초과 상태에서 개선이 없을 때만 진짜 "정체"로 판정한다.
                        still_over_tolerance = (dur_err_ema_per_item[pi] > dur_gate_tolerance)
                        if rel_improve < dur_gate_stall_min_improvement and still_over_tolerance:
                            stalled_items.add(pi)
                            print(f"    [정체 판정] 항목[{pi+1}] duration EMA가 "
                                  f"{dur_gate_stall_patience_steps}스텝 동안 "
                                  f"{rel_improve*100:.1f}%밖에 개선 안 됨(기준 "
                                  f"{dur_gate_stall_min_improvement*100:.0f}%). "
                                  f"웜다운 게이트 필수조건에서 제외(학습은 계속됨).")
                        else:
                            dur_err_stall_baseline[pi] = (dur_err_ema_per_item[pi], step)

        # [FIX-B] 페어드 dur 손실 → style_dp에 실제 gradient 공급
        if use_paired and train_dp and style_dp.requires_grad and dur_loss_weight > 0:
            dur_loss = ((dur_natural - current_dur_target)
                        / max(current_dur_target, 1e-3)) ** 2
            loss = loss + dur_loss_weight * dur_loss.mean()

        # [v5-NEW] Multi-Scale Spectral Loss — 게이팅 조건부 적용.
        # duration이 아직 크게 어긋난 상태(학습 초반/dp 미수렴)에서는 STFT 프레임
        # 정렬 자체가 부정확해 노이즈 그래디언트가 될 수 있어, 오차율이
        # mss_dur_tolerance 이내로 수렴했을 때만 활성화한다.
        if (use_paired and mss_loss_fn is not None
                and dur_ratio_err is not None and dur_ratio_err <= mss_dur_tolerance):
            loss = loss + mss_loss_weight * mss_loss_fn(gen_wav, target_wav_ts[pi])

        # [v5-NEW] Style 정규화 (조건부 적용)
        # - std 경계(0.06~0.08) 페널티: 가이드 문서 진단 기준을 실제 loss로 반영 (채택)
        # - 레퍼런스 앵커 L2: 기본 가중치 0으로 비활성화. 최적점이 초기 레퍼런스
        #   프리셋에서 먼 경우 매칭을 방해할 위험이 있어 필요할 때만
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
            # 웜다운이 조기 발동하는 것을 방지 (학습 초반 조기 언더핏 사례 대응)
            smoothed_l3 = (primary_loss if smoothed_l3 is None
                           else early_stop_ema_alpha * primary_loss
                                + (1 - early_stop_ema_alpha) * smoothed_l3)
        if (not use_paired) and primary_loss < best_loss:
            best_loss = primary_loss
            best_ttl  = style_ttl.detach().clone()
            best_dp   = style_dp.detach().clone()
            best_step = step + 1
        elif use_paired and best_ttl is None:
            # 초반 전부 페어드일 때 best 미설정 방지
            best_ttl = style_ttl.detach().clone()
            best_dp  = style_dp.detach().clone()
            best_step = step + 1

        # ── 로그 출력 (Cell 10 파싱 정규식 호환 포맷 고정) ────────────────────
        if (step + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            ttl_std    = style_ttl.detach().std().item()
            ema_str    = f"{smoothed_l3:.4f}" if smoothed_l3 is not None else "N/A"
            # [v11-NEW] EMA(dur_err)는 기존 정규식이 찾는 마지막 캡처그룹(EMA(L3))
            # 뒤에 덧붙인다. 정규식이 문자열 끝(`$`)에 고정되어 있지 않고
            # `re.search`로 부분 매치하므로, 뒤에 필드를 추가해도 기존 Cell 10
            # 파싱은 영향받지 않는다(검증: 아래 self-test 참고).
            dur_ema_str = (f"{smoothed_dur_err:.4f}" if smoothed_dur_err is not None else "N/A")
            # [v16-NEW] EMA(dur_err)와 같은 방식으로 뒤에 덧붙인다(re.search 부분
            # 매치라 기존 Cell 10 정규식엔 영향 없음). "가장 안 풀린 항목"의 저역
            # EMA를 진단용으로 노출 — 이 값이 훈련 내내 특정 항목(예: item2)에서만
            # 안 줄어드는지 실시간으로 확인 가능해진다.
            low_ema_str = (f"{max(low_band_err_ema_per_item.values()):.2f}"
                            if low_band_err_ema_per_item else "N/A")
            print(f"  Step {step+1}/{num_steps} | Loss(total): {loss.item():.4f} | "
                  f"Loss(L3): {primary_loss:.4f} | LR: {current_lr:.5f} | "
                  f"Best(L3): {best_loss:.4f} | EMA(L3): {ema_str} | "
                  f"ttl_std: {ttl_std:.4f} | EMA(dur_err): {dur_ema_str} | "
                  f"EMA(low_err): {low_ema_str}")
            if ttl_std > 0.09 or ttl_std < 0.05:
                print(f"    [주의] style_ttl std({ttl_std:.4f})가 정상 범위(0.06~0.08) 밖입니다. "
                      f"과적합/붕괴 가능성 — lr 또는 threshold 재검토 권장.")

            # [FIX-G] 진행률 바를 best_loss(순간 최저치)가 아니라 웜다운 실제
            # 판정 지표(EMA, 형성 전엔 best_loss 폴백)로 계산하도록 통일.
            # 기존 버그: best_loss만 보고 100%를 찍어버려서, EMA가 threshold에
            # 못 미쳐 웜다운이 전혀 시작 안 된 상태에서도 "100%인데 안 끝난다"는
            # 혼란을 유발했음. 이제 progress_gate가 warmdown 판정과 동일한 값이므로
            # 100%가 뜨면 실제로 그 다음 스텝에 웜다운이 시작된다.
            progress_gate = smoothed_l3 if smoothed_l3 is not None else best_loss
            if initial_gap is None and progress_gate < float('inf'):
                initial_gap = max(0.001, progress_gate - threshold)
            if initial_gap is not None:
                current_gap = max(0.0, progress_gate - threshold)
                percentage  = (max(0.0, (1.0 - (current_gap / initial_gap)) * 100.0)
                               if current_gap > 0 else 100.0)
                num_blocks  = min(20, max(0, int(percentage / 5)))
                bar_str     = "█" * num_blocks + "-" * (20 - num_blocks)
                gate_label  = "EMA" if smoothed_l3 is not None else "best(EMA 형성전 임시)"
                grace_note  = ""
                if needs_resume_grace and (step - start_step < resume_grace_steps):
                    remaining = resume_grace_steps - (step - start_step)
                    grace_note = f"  [재개 유예 중: 웜다운 {remaining}스텝 후 판정 시작]"
                print(f"    [{bar_str}]  {percentage:.1f}%  "
                      f"({gate_label} {progress_gate:.4f} -> 목표 {threshold:.4f}, "
                      f"gap +{current_gap:.4f})  [참고: best(순간최저) {best_loss:.4f}]{grace_note}")

        # ── 체크포인트 저장 ───────────────────────────────────────────────────
        if (step + 1) % save_every == 0 and best_ttl is not None:
            ckpt_path = f"{log_dir}/{name}_{step+1:04d}.json"
            # [v10-NEW] EMA/best_loss/paired_ratio를 체크포인트에 함께 저장해
            # 다음 재개 시 정확히 이어받도록 함 (재개 즉시 조기 웜다운 방지).
            train_state = {
                "smoothed_l3": smoothed_l3,
                "smoothed_dur_err": smoothed_dur_err,
                "dur_err_ema_per_item": dur_err_ema_per_item,
                "low_band_err_ema_per_item": low_band_err_ema_per_item,  # [v16-NEW]
                "stalled_items": sorted(stalled_items),
                "dur_err_stall_baseline": dur_err_stall_baseline,
                "best_loss": best_loss,
                "best_dur_err": best_dur_err,
                "best_dur_step": best_dur_step,
                "current_paired_ratio": current_paired_ratio,
            }
            # [v14-NEW] duration 전용 best dp가 있으면 그걸 저장, 없으면(dur_gate
            # 비활성 등) 기존처럼 timbre 기준 best_dp로 폴백 — 하위호환 유지.
            dp_for_ckpt = best_dp_for_duration if best_dp_for_duration is not None else best_dp
            save_style(ckpt_path, best_ttl, dp_for_ckpt, target_wav_paths, train_state=train_state)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        # ── Warmdown ─────────────────────────────────────────────────────────
        # [v4] 단일 샘플 best_loss 대신 EMA(smoothed_l3)로 판정 → 조기 락인 방지.
        # EMA가 아직 형성되지 않은 초반 스텝에는 종전처럼 best_loss로 폴백.
        # [FIX-G] 이 값은 위 progress bar의 progress_gate와 완전히 동일한 계산식이다
        # (progress bar는 10스텝마다만 찍히므로 별도 변수로 매 스텝 재계산).
        warmdown_gate = smoothed_l3 if smoothed_l3 is not None else best_loss
        # [v12-FIX] 실측 사례: 특정 항목이 45.9% 오차로 안 풀렸는데도 855스텝만에
        # 웜다운 발동(원인: pooled EMA가 최근 샘플링 운으로 왜곡, 몬테카를로 검증상
        # 45% 확률로 오탐 가능했음). 이제 (a) 5개 항목 전부 최소 1번씩은 샘플링돼
        # 있어야 하고 (b) 그중 가장 안 풀린 항목의 EMA도 tolerance 이내여야 통과.
        # [v13-FIX] 실측 사례: 특정 항목이 7350스텝 동안 7.46%밖에 개선 안 됨
        # (구조적 outlier). "전부 만족" 규칙 그대로면 나머지 4개가 다 풀려도
        # num_steps 전체를 낭비함. stalled_items로 판정된 항목은 이 요구조건에서
        # 제외(단, 정체 판정 자체가 꺼져 있으면 stalled_items는 항상 비어 있어
        # v12와 동일하게 동작 — 하위호환).
        n_total_items = len(paired_text_inputs) if paired_text_inputs else 0
        required_items_ok = all(
            (pi in stalled_items)
            or (pi in dur_err_ema_per_item and dur_err_ema_per_item[pi] <= dur_gate_tolerance)
            for pi in range(n_total_items)
        ) if (dur_gate_tolerance is not None and n_total_items > 0) else True
        dur_gate_ok = (
            dur_gate_tolerance is None
            or not train_dp
            or smoothed_dur_err is None
            or required_items_ok
        )
        # [v10-NEW] 구버전 체크포인트로 재개해 EMA가 새로 형성 중인 동안은
        # resume_grace_steps만큼 웜다운 발동을 유예. 재개 직후 9스텝만에
        # (EMA가 첫 몇 샘플만으로 형성되어) 웜다운이 발동, 이후 학습이 사실상
        # 진행되지 않아 duration 오차가 21.5%까지 방치된 사례의 재발 방지.
        in_resume_grace = needs_resume_grace and (step - start_step < resume_grace_steps)
        if not warmdown_mode and warmdown_gate <= threshold and not in_resume_grace:
            if not dur_gate_ok:
                if (step + 1) % 10 == 0:
                    n_total = len(paired_text_inputs) if paired_text_inputs else 0
                    uncovered = [pi for pi in range(n_total)
                                 if pi not in stalled_items and pi not in dur_err_ema_per_item]
                    if uncovered:
                        print(f"    [duration 게이트 대기] 음색 EMA는 threshold 도달했지만 "
                              f"아직 페어드 항목 {n_total - len(uncovered)}/{n_total}개만 "
                              f"관측돼 웜다운을 유예합니다.")
                    else:
                        # 필수 항목(정체 판정 안 된 것들) 중 가장 안 풀린 것을 보고
                        required = {pi: v for pi, v in dur_err_ema_per_item.items()
                                    if pi not in stalled_items}
                        worst_pi = max(required, key=required.get)
                        stall_note = (f" (정체 제외된 항목: "
                                      f"{sorted(p+1 for p in stalled_items)})"
                                      if stalled_items else "")
                        print(f"    [duration 게이트 대기] 음색 EMA는 threshold 도달했지만 "
                              f"가장 안 풀린 항목[{worst_pi+1}]의 duration EMA"
                              f"({required[worst_pi]:.3f})가 목표({dur_gate_tolerance:.3f}) "
                              f"이내가 아니라 웜다운을 유예합니다.{stall_note}")
            else:
                print(f"\n>>> [Phase Transition] EMA(L3) Threshold({threshold}) 도달! "
                      f"(EMA={warmdown_gate:.4f})")
                print(f">>> Dynamic Warm-down 모드 활성화 "
                      f"(style_ttl LR x0.05, style_dp LR x{warmdown_dp_lr_scale}, "
                      f"최대 50스텝 추가 정렬)")
                warmdown_mode = True
                # [v6-FIX] style_ttl(index 0)과 style_dp(index 1, train_dp일 때만 존재)를
                # 구분해서 스케일 적용. style_dp는 warmdown_dp_lr_scale(기본 1.0=유지)로
                # 별도 조정해, 웜다운 진입 시점에 duration 매칭이 덜 끝났어도 이후
                # 최대 50스텝 동안 계속 개선될 여지를 남긴다.
                for gi, pg in enumerate(optimizer.param_groups):
                    if train_dp and gi == 1:
                        pg['lr'] = pg['lr'] * warmdown_dp_lr_scale
                    else:
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
    final_ttl_for_save = best_ttl if best_ttl is not None else style_ttl
    # [v14-NEW] duration 전용 best dp가 있으면 그걸 최종 저장에 쓴다. 아래 검증
    # 리포트(다차원 평가 리포트)도 반드시 같은 dp로 합성해야, 리포트에 찍히는
    # duration/저역 숫자가 "실제로 저장된 체크포인트"와 일치한다 — 그렇지 않으면
    # 리포트가 저장본과 다른 파라미터를 보여주는 오도(misleading) 상황이 된다.
    final_dp_for_save = best_dp_for_duration if best_dp_for_duration is not None else best_dp
    final_train_state = {
        "smoothed_l3": smoothed_l3,
        "smoothed_dur_err": smoothed_dur_err,
        "dur_err_ema_per_item": dur_err_ema_per_item,
        "low_band_err_ema_per_item": low_band_err_ema_per_item,  # [v16-NEW]
        "stalled_items": sorted(stalled_items),
        "dur_err_stall_baseline": dur_err_stall_baseline,
        "best_loss": best_loss,
        "best_dur_err": best_dur_err,
        "best_dur_step": best_dur_step,
        "current_paired_ratio": current_paired_ratio,
    }
    save_style(final_path, final_ttl_for_save, final_dp_for_save, target_wav_paths,
               train_state=final_train_state)

    # [v6-NEW] Duration 매칭 사후 검증. 합성 음성이 원본보다 최대 25% 길게
    # 나오는 문제가 dp_lr_ratio 과소 설정으로 발생했음이 사후(음원 직접 청취)에야
    # 발견되었음. 학습 종료 시점에 각 페어드 타겟에 대해 실제로 합성해 dur 오차를
    # 직접 측정/리포트하여, 다음부터는 음원을 듣기 전에 로그만으로 이 문제를
    # 조기에 발견할 수 있게 한다.
    # [v7-NEW] 같은 forward에서 나온 wav를 재사용해 저역(100-500Hz) 비율 오차도
    # 함께 실측(추가 forward 없음). "청감상 가볍다"는 문제를 사전에
    # 로그로 잡기 위함.
    dur_report = []
    low_band_report = []
    if paired_mode:
        with torch.no_grad():
            for pi, ((ids, mask), tgt_dur) in enumerate(zip(paired_text_inputs, target_dur_secs)):
                wav_final, dur_final = tts_forward(
                    ids, mask, final_ttl_for_save, final_dp_for_save,
                    dp_model, te_model, ve_model, voc_model,
                    total_step, speed, pr_latents[pi], pr_masks[pi]
                )
                dur_natural = (dur_final.squeeze() * speed).item()
                err_pct = (dur_natural - tgt_dur) / max(tgt_dur, 1e-3) * 100.0
                dur_report.append((pi, tgt_dur, dur_natural, err_pct))

                gen_ltas_final = compute_ltas_distribution(wav_final.squeeze())
                gen_low  = low_band_ratio(gen_ltas_final).item()
                tgt_low  = low_band_ratio(target_ltas_list[pi]).item()
                low_band_report.append((pi, tgt_low, gen_low))

    elapsed = time.time() - start_time
    final_ttl_std = final_ttl_for_save.detach().std().item()
    print("\n" + "="*60)
    print("[학습 완료 결과 다차원 평가 리포트]")
    print(f"- 총 학습 소요 시간  : {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"- 최종 학습 진행 스텝: {step + 1} / {num_steps}")
    print(f"- 최소 음색 손실값   : {best_loss:.4f} (step {best_step} 스냅샷 — style_ttl은 이 시점 파라미터로 저장됩니다)")
    # [v14-NEW] style_dp는 v14부터 style_ttl과 독립적으로 "duration이 가장 잘 맞았던
    # 시점"을 따로 추적해 저장한다 — 같은 step이 아닐 수 있음을 명시.
    if best_dp_for_duration is not None:
        print(f"- 최소 duration 오차 : EMA {best_dur_err:.4f} (step {best_dur_step} 스냅샷) — style_dp는 "
              f"이 시점 파라미터로 저장됩니다(style_ttl과 다른 step일 수 있음)")
    else:
        print(f"- 최소 duration 오차 : 추적 안 됨(페어드 텍스트 없음/dp 미학습) — style_dp는 "
              f"style_ttl과 같은 시점(음색 최저점) 파라미터로 저장됩니다(v13 이전과 동일)")
    print(f"- EMA 음색 손실값    : {smoothed_l3:.4f}" if smoothed_l3 is not None else "- EMA 음색 손실값    : N/A")
    # [FIX-J] EMA(smoothed_l3)는 "현재(마지막) style_ttl" 기준 실시간 모니터링 값이고,
    # 저장되는 체크포인트는 best_ttl(=best_loss가 나온 step의 스냅샷)이다. 웜다운 꼬리
    # 구간에서 unpaired 텍스트 난이도 편차로 EMA가 threshold 위로 다시 튀는 경우,
    # "성공" 판정과 "EMA가 threshold를 넘었다"는 로그가 동시에 찍혀 혼란을 준다.
    # → 두 값이 서로 다른 파라미터 상태를 가리킨다는 점을 명시적으로 경고.
    ema_drifted_above_threshold = (
        warmdown_mode and smoothed_l3 is not None and smoothed_l3 > threshold
    )
    if ema_drifted_above_threshold:
        print(f"  [참고] 최종 EMA({smoothed_l3:.4f})가 threshold({threshold:.4f})보다 높습니다. "
              f"이는 웜다운 꼬리 구간(unpaired 샘플 편차)에서 '현재' 파라미터 기준 "
              f"모니터링 값이 흔들린 것으로, 실제 저장된 체크포인트(best_ttl, step "
              f"{best_step}, loss {best_loss:.4f})와는 다른 상태를 가리킵니다. "
              f"음원 품질 자체보다는 리포트 해석 문제일 가능성이 높습니다.")
    print(f"- 최종 style_ttl std : {final_ttl_std:.4f} (정상범위 0.06~0.08)")
    print(f"- 학습 모드          : {'PAIRED' if paired_mode else 'UNPAIRED'}"
          f" | hf_weight={hf_weight} | ltas_weight={ltas_weight} | low_band_weight={low_band_weight}")
    if dur_report:
        print("-" * 60)
        print("- Duration 매칭 (원본 발화 길이 대비 오차, 0%에 가까울수록 좋음):")
        abs_errs = []
        for pi, tgt_dur, dur_natural, err_pct in dur_report:
            print(f"    [{pi+1}] 목표 {tgt_dur:.2f}s → 합성 {dur_natural:.2f}s "
                  f"(오차 {err_pct:+.1f}%)")
            abs_errs.append(abs(err_pct))
        mean_abs_err = sum(abs_errs) / len(abs_errs)
        print(f"  평균 절대 오차: {mean_abs_err:.1f}%")
        if mean_abs_err > 15.0:
            print(f"  [경고] duration 오차가 큽니다(>15%). dp_lr_ratio를 올리거나 "
                  f"(예: 0.01~0.03) warmdown_dp_lr_scale을 1.0 근처로 유지해보세요.")
        if stalled_items:
            print(f"  [참고] 항목 {sorted(p+1 for p in stalled_items)}은(는) duration 게이트 "
                  f"진행 중 '정체'로 판정되어 웜다운 필수조건에서 제외됐습니다 — "
                  f"구조적으로 더 학습해도 잘 안 줄어드는 항목일 가능성이 있습니다.")
    if low_band_report:
        print("-" * 60)
        print("- 저역(100-500Hz, 중후함/가슴공명대) 비율 매칭 (원본 대비, %p 단위 오차):")
        abs_band_errs = []
        for pi, tgt_low, gen_low in low_band_report:
            diff_pp = (gen_low - tgt_low) * 100.0
            print(f"    [{pi+1}] 목표 {tgt_low*100:.1f}% → 합성 {gen_low*100:.1f}% "
                  f"(오차 {diff_pp:+.1f}%p)")
            abs_band_errs.append(abs(diff_pp))
        mean_band_err = sum(abs_band_errs) / len(abs_band_errs)
        print(f"  평균 절대 오차: {mean_band_err:.1f}%p")
        if mean_band_err > 5.0:
            print(f"  [참고] 저역 비율 오차가 5%p를 넘습니다. low_band_weight를 "
                  f"높여보거나(현재 {low_band_weight}), ltas_weight와의 균형을 재검토해보세요.")
        # [v16-NEW] 학습 내내 관측된 항목별 저역 EMA(진단용, low_band_weight>0이거나
        # paired_sampling_mode="weighted"일 때만 채워짐). duration의 "정체 판정"과
        # 같은 취지로, 특정 항목만 학습 시작부터 끝까지 안 풀렸는지 한눈에 보여준다
        # — 단, 이 값은 게이트/저장 시점 선택에는 관여하지 않는 순수 진단 정보다.
        if low_band_err_ema_per_item:
            worst_pi, worst_ema = max(low_band_err_ema_per_item.items(), key=lambda kv: kv[1])
            print(f"  [진단] 학습 중 항목별 저역 오차 EMA(최종): "
                  + ", ".join(f"[{p+1}] {v:.1f}%p"
                              for p, v in sorted(low_band_err_ema_per_item.items())))
            if worst_ema > 8.0:
                print(f"  [참고] 항목[{worst_pi+1}]은 학습 내내 저역 오차 EMA가 가장 컸습니다"
                      f"({worst_ema:.1f}%p). item[1]의 duration처럼 구조적 outlier일 가능성이 "
                      f"있습니다 — low_band_weight를 더 올려도 안 줄어든다면(이미 0.1→0.12 "
                      f"테스트에서 확인됨), 사후 EQ 보정 대상으로 검토해보세요.")
    print("-" * 60)
    if warmdown_mode and ema_drifted_above_threshold:
        print("▶ 최종 판정: [⚠️ 웜다운 진입했으나 종료 시점 EMA가 threshold 재초과 — "
              "저장본은 best 스냅샷 사용됨, 음원 직접 청취로 재확인 권장]")
    elif warmdown_mode:
        print("▶ 최종 판정: [✨ 웜다운 방어 수렴 성공]")
    else:
        print("▶ 최종 판정: [✅ 만기 완주 수렴]")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# [v16-NEW] self-test — `python optimize_style.py --selftest`로 실행.
# GPU/ONNX 모델/오디오 파이프라인을 전혀 쓰지 않고 select_paired_index()의
# 샘플링 로직만 검증한다. "코드 수정 시 실행/시뮬레이션으로 검증" 원칙에 따라
# 추가됨 — 단, 이 self-test가 통과해도 실제 학습 결과(음색/duration/저역 수렴)를
# 보장하지 않는다. 그건 반드시 실제 학습 1회로 확인해야 한다.
# ─────────────────────────────────────────────────────────────────────────────
def _selftest_select_paired_index():
    n_items = 5
    N = 20000

    # 1) uniform 모드는 순수 random.randrange와 통계적으로 동일해야 한다
    #    (하위 호환 검증 — 기존 config로 돌리는 사람에게 아무 영향 없어야 함).
    rng = random.Random(0)
    counts = [0] * n_items
    for _ in range(N):
        pi = select_paired_index(n_items, {}, {}, set(), mode="uniform", rng=rng)
        counts[pi] += 1
    expected = N / n_items
    max_dev = max(abs(c - expected) / expected for c in counts)
    assert max_dev < 0.05, f"[FAIL] uniform 모드 분포 편차 과다: {counts}"
    print(f"[PASS] uniform 모드 == 균등분포 (편차 {max_dev*100:.1f}%): {counts}")

    # 2) coverage-first: EMA가 하나도 없는 초반엔 전 항목이 최소 1번씩 관측될
    #    때까지 반드시 미관측 항목 중에서만 뽑혀야 한다 (weighted 계산이 빈 평균으로
    #    왜곡되는 것 방지).
    rng = random.Random(1)
    dur_ema, low_ema = {}, {}
    picked = []
    for _ in range(n_items):
        pi = select_paired_index(n_items, dur_ema, low_ema, set(), mode="weighted", rng=rng)
        picked.append(pi)
        dur_ema[pi], low_ema[pi] = 0.1, 1.0
    assert sorted(picked) == list(range(n_items)), f"[FAIL] coverage-first 위반: {picked}"
    print(f"[PASS] coverage-first: 초반 {n_items}스텝에 전 항목 1회씩 관측됨 {picked}")

    # 3) welt14 실측 패턴 재현: item1(인덱스0)은 duration 정체(stalled),
    #    item2(인덱스1)는 저역 오차가 4번의 학습 내내 가장 컸음(+11.9~+14.3%p).
    dur_ema = {0: 0.427, 1: 0.119, 2: 0.139, 3: 0.058, 4: 0.142}
    low_ema = {0: 7.0,   1: 13.6,  2: 5.1,   3: 1.7,   4: 4.3}
    stalled = {0}
    rng = random.Random(2)
    counts = [0] * n_items
    for _ in range(N):
        pi = select_paired_index(n_items, dur_ema, low_ema, stalled,
                                  mode="weighted", explore_ratio=0.2, rng=rng)
        counts[pi] += 1
    freqs = [c / N for c in counts]
    print("[INFO] weighted 샘플링 빈도(welt14 실측 패턴 재현): "
          + ", ".join(f"item{i+1}={f:.3f}" for i, f in enumerate(freqs)))
    assert freqs[1] == max(freqs), \
        f"[FAIL] 가장 안 풀린 비정체 항목(item2)이 최다 샘플링돼야 함: {freqs}"
    assert 0 < freqs[0] < freqs[1], \
        f"[FAIL] 정체 항목(item1)은 낮되 0은 아니어야 함: {freqs}"
    assert freqs[3] > 0.02, \
        f"[FAIL] explore_ratio가 최소 커버리지를 보장하지 못함(item4): {freqs}"
    print("[PASS] weighted 샘플링이 welt14 실측 패턴에서 기대한 방향으로 동작함 "
          "(item2 최다 샘플링, item1은 낮지만 0은 아님, explore_ratio로 전 항목 최소 커버 유지)")

    print("\n[v16 self-test] 전부 통과. ※ 이 self-test는 샘플링 로직만 검증합니다 — "
          "GPU/ONNX 모델/실제 오디오를 쓰는 전체 학습 결과(음색·duration·저역 수렴)는 "
          "보장하지 않으며, 반드시 실제 학습 1회로 최종 확인해야 합니다.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest_select_paired_index()
    else:
        main()
