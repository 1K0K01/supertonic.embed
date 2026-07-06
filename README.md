물론이죠! 바로 다운로드 가능한 MD 파일로 만들어드리겠습니다. 아래 내용을 복사하셔서 README_kor.md 파일로 저장하시면 됩니다.

---

```markdown
# SupertonicTTS 보이스 스타일 추출기 (한국어 남성 저음 특화 개조판)

이 저장소는 SupertonicTTS의 공식 스타일 인코더(미공개) 없이, **어떤 WAV 파일에서도 음성 스타일 임베딩을 추출**할 수 있게 안내합니다.

3~10초 분량의 음성 샘플을 넣으면, SupertonicTTS가 그 목소리로 말하게 하는 **음성 스타일 JSON**을 얻을 수 있습니다.

**요구 사항:** NVIDIA GPU (VRAM 4GB 이상), CUDA 지원.

---

## 책임 있는 사용 (Responsible Use)

**이 코드는 오직 학술 연구 목적으로 공개되었습니다.** 음성 복제 기술은 심각한 피해를 초래할 수 있습니다. 이 저장소를 사용함으로써 귀하는 다음 사항에 동의합니다.

- **음성을 복제하려는 화자로부터 명시적인 동의**를 받아야 합니다. 허가 없이 실제 사람의 목소리를 복제하는 것은 귀하의 관할권에서 불법일 수 있습니다.
- 이 도구를 **비동의 음성 사칭, 보이스 피싱, 사기, 괴롭힘, 명예 훼손, 오도하는 정치·상업 콘텐츠 생성, 또는 음성 기반 인증 우회**에 사용하지 마십시오.
- **명시적 허가 없이 식별 가능한 개인(공인 포함)**을 타겟으로 삼지 마십시오.
- 합성 오디오를 배포할 때는 **AI 생성임을 밝히고**, 가능하면 워터마크나 출처 메타데이터(C2PA 등)를 포함하십시오.

저자는 모든 오용에 대한 책임을 부인합니다. 첨부된 논문의 평가는 Common Voice(CC0 라이선스)를 사용하며 식별 가능한 개인을 타겟으로 하지 않습니다. 이 코드의 오용을 발견하면 이슈를 열어 주십시오.

---

## 동작 원리

```

┌───────────┐ │ ┌────────────┐  ┌─────────────┐  ┌───────────┐ │ ┌─────────┐
│   style   │→│ │  Text      │  │  Vector     │  │  Vocoder  │ │→│ 생성 WAV │
│  vector   │ │ │ Encoder    │  │  Estimator  │  │           │ │ └────┬────┘
└─────┬─────┘ │ └────────────┘  └─────────────┘  └───────────┘ │      │
│       └────────────────────────────────────────────────┘      │
│                                                               │
│                      ┌───────────────┐                       │
│                      │    WavLM      │◄──────────────────────┘
│                      │   Layer 3     │
│                      │ (화자 정체성)  │◄── 타겟 WAV
│                      └───────┬───────┘
│                              │
│        gradient              │ loss
└──────────────────────────────┘
"스타일을 타겟과 더 유사하게 업데이트"

```

1. WavLM Layer 3 거리를 기준으로 **가장 가까운 프리셋 스타일(F1~F5, M1~M5) 자동 선택**
2. TTS로 WAV를 합성하고, **WavLM Layer 3 특징**으로 타겟 WAV와 비교
3. 손실이 수렴할 때까지 **스타일 벡터를 경사 하강법으로 업데이트** (임계값 0.24에서 조기 종료)

### 수렴 가이드
동일 화자 기준 손실은 **0.15~0.24**입니다. 최적화는 이 임계값에 도달하면 자동으로 중지됩니다.

---

## 주요 개조 사항 (본 저장소)

이 저장소는 원본 레포지토리를 **한국어 남성 저음 TTS**에 최적화하도록 개조했습니다.

- **`hf_weight` Config 연동** – 고주파 억제 강도를 config에서 조절 가능 (기존 하드코딩 제거)
- **페어드 텍스트 모드 (`target_texts`)** – 각 WAV에 실제 대사를 제공하면, 동일 문장끼리 비교하여 **운율(prosody)과 발화 길이**를 감독
- **LTAS (장기 평균 스펙트럼) 손실** – 주파수 대역별 에너지 분포를 매칭하여 **밝음/탁함 양방향 교정**
- **시퀀스 손실 (`seq_loss_weight`)** – WavLM Layer 6의 시간축 운율 패턴을 정렬하여 **억양/리듬 학습**
- **발화 길이 손실 (`dur_loss_weight`)** – 페어드 모드에서 style_dp에 실제 gradient를 공급
- **동적 `paired_ratio`** – 웜다운(미세 조정) 단계에서 페어드 비율을 낮춰 **과적합 방지**
- **텍스트별 고정 latent** – 텍스트 길이에 따라 독립적인 노이즈 latent를 생성하여 **길이 의존 왜곡 제거**

> **⚠️ 현재 이 저장소는 아직 완성된 버전이 아닙니다.** 계속해서 실험과 개선이 진행 중이며, 공식 릴리스는 추후 예정입니다.  
> 또한, **이 레포지토리를 이용한 학습용 Colab 노트북은 현재 미공개 상태**입니다.

---

## 빠른 시작 (Quick Start)

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. SupertonicTTS 모델 다운로드

Supertone/supertonic-2에서 onnx/와 voice_styles/ 폴더를 다운로드하여 프로젝트 루트에 배치하세요.

3. 음성 샘플 준비

WAV 파일(3~10초, 단일 화자)을 wavs/ 폴더에 넣으세요. 샘플레이트는 자동으로 44.1kHz로 리샘플링됩니다.

4. Config 생성

configs/my_voice.json 예시:

```json
{
  "name": "my_voice",
  "target_wavs": ["wavs/sample1.wav", "wavs/sample2.wav"],
  "multi_wav_mode": "stochastic",
  "reference_style": "auto",
  "seed": 777,
  "lr": 1.2e-4,
  "num_steps": 10000,
  "total_step": 5,
  "speed": 1.05,
  "save_every": 100,
  "early_stop_loss_threshold": 0.13,
  "train_style_dp": true,
  "dp_lr_ratio": 0.001,
  "use_ecapa_loss": true,
  "ecapa_loss_weight": 0.35,
  "hf_weight": 0.10,
  "ltas_weight": 0.5,
  "target_texts": ["대사1", "대사2", ...],
  "seq_loss_weight": 0.3,
  "dur_loss_weight": 0.3,
  "paired_ratio": 0.7
}
```

파라미터 설명
name 체크포인트 및 결과 파일 저장 이름
target_wavs 음성 클로닝에 사용할 WAV 파일 경로 리스트
multi_wav_mode "stochastic" (랜덤 샘플링) 또는 "average" (평균 고정)
reference_style "auto" (자동 선택) 또는 "voice_styles/F1.json" (수동)
seed 재현성을 위한 랜덤 시드
lr 학습률. 1.2e-4 ~ 1.5e-4 권장. 너무 높으면 발음이 깨짐
num_steps 최대 최적화 스텝 (조기 종료가 더 빠를 수 있음)
total_step Flow Matching 반복 횟수 (5=학습용, 10~15=합성 품질용)
speed 발화 속도 (1.00~1.05 권장)
save_every 체크포인트 저장 간격
early_stop_loss_threshold Layer 3 손실 조기 종료 임계값 (0.13~0.17)
train_style_dp style_dp 학습 여부 (페어드 모드에서만 의미)
dp_lr_ratio style_dp 학습률 비율 (0.001~0.005)
use_ecapa_loss ECAPA 화자 정체성 손실 사용 여부
ecapa_loss_weight ECAPA 손실 가중치 (0.35~0.40)
hf_weight 4kHz 이상 고주파 억제 강도 (0.08~0.15)
ltas_weight LTAS 스펙트럼 매칭 손실 가중치 (0.2~0.5, 0=끔)
target_texts (페어드 모드) 각 WAV에 대응하는 실제 대사 리스트
seq_loss_weight 페어드 시퀀스(운율) 손실 가중치 (0.1~0.3)
dur_loss_weight 페어드 발화 길이 손실 가중치 (0.1~0.3)
paired_ratio 페어드 샘플 선택 확률 (0.4~0.7)

5. 최적화 실행

```bash
python optimize_style.py my_voice
```

중단된 경우 최신 체크포인트에서 자동으로 재개됩니다. 손실이 0.24 이하로 떨어지면 조기 종료됩니다.

6. 추출된 스타일 사용

main.py에서 추론 예제를 확인하세요.

---

소요 시간

· 모델 로딩 및 변환: ~30초
· 자동 프리셋 선택: ~1분 (10개 스타일 비교)
· 최적화: RTX 3090 기준 평균 503스텝, 약 5~6분

---

성능 (원본 논문 기준)

44명 화자 × 5 발화 = 220개 샘플 평가:

 SIM (WavLM) ↑ SIM (ECAPA) ↑ SIM (ResNet) ↑ WER ↓
최근접 프리셋 (최적화 없음) 0.758 0.129 0.112 4.80%
제안 방법 0.867 0.452 0.446 2.70%

---

파일 구조

```
configs/                  # 학습 설정
├── my_voice.json
└── ...

wavs/                     # 참조 WAV 파일
├── sample1.wav
└── ...

onnx/                     # SupertonicTTS ONNX 모델
├── duration_predictor.onnx
├── text_encoder.onnx
├── vector_estimator.onnx
├── vocoder.onnx
├── tts.json
└── unicode_indexer.json

voice_styles/             # 음성 스타일 JSON
├── M1~M5.json, F1~F5.json  (프리셋)
└── my_voice.json           (추출 결과)

logs/                     # 체크포인트
└── my_voice/
    ├── train_config.json
    ├── my_voice_00000100.json
    └── ...

results/                  # 테스트 출력
└── my_voice_optimized.wav
```

---

사용 모델

모델 역할
duration_predictor 지속 시간 예측 (SupertonicTTS)
text_encoder 텍스트 인코딩 (SupertonicTTS)
vector_estimator Flow Matching 노이즈 제거 (SupertonicTTS)
vocoder 잠재 벡터 → WAV (SupertonicTTS)
WavLM-Large 지각 손실, Layer 3 (microsoft/wavlm-large)

---

기술적 상세

ONNX → PyTorch 변환

경사 역전파를 위해 ONNX 모델을 PyTorch로 변환합니다:

· onnxslim으로 모델 정리
· opset 17 강제 지정 (onnx2torch 호환성)
· Clip 노드의 빈 입력 수정

WavLM Layer 3 특징 매칭

Chiu et al. (2025)의 연구에 따르면, WavLM Layer 3가 화자 정체성을 가장 잘 인코딩합니다. 생성 오디오와 타겟 오디오의 시간 평균 특징 통계(평균, 표준편차)를 비교합니다. 시간축 평균화는 콘텐츠 의존성을 줄입니다.

스타일 공간

· style_ttl [1, 50, 256] = 12,800개 파라미터 (음색, 최적화 대상)
· style_dp [1, 8, 16] = 128개 파라미터 (리듬/지속 시간, 선택적 학습)

남성과 여성 음성은 스타일 공간에서 서로 다른 영역을 차지합니다. 가장 가까운 프리셋 스타일에서 시작하는 것이 중요합니다.

---

인용 (Citation)

이 작업을 사용한다면 다음을 인용해 주세요:

```bibtex
@misc{kim2026supertonicembed,
  author       = {Gyeongmin Kim},
  title        = {Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19646514},
  url          = {https://doi.org/10.5281/zenodo.19646514}
}
```

Zenodo 프리프린트: https://doi.org/10.5281/zenodo.19646514

---

도움말 (Looking for help?)

질문이 있으시면 자유롭게 이슈를 열어 주세요.

```
