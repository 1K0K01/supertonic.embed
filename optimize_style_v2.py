import os
import json
import argparse
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torch.fft
from transformers import WavLMModel

# 기존 레포지토리의 이식용 헬퍼 함수 임포트 (SupertonicTTS inference 모듈)
# 주의: 환경에 맞게 helper 모듈의 함수명은 실제 코드에 맞춰 사용하십시오.
try:
    from helper import load_tts_models, generate_wav_from_style
except ImportError:
    print("[경고] helper.py를 찾을 수 없습니다. 코랩 환경에 맞게 함수를 매핑하세요.")

# =====================================================================
# 1. F0 프로파일링 및 자동 프리셋 매칭 로직
# =====================================================================
def analyze_f0_profile(wav_path):
    """
    [구조적 변경 사항 3 반영] 성우별 F0 연동 베이스 프리셋(M1~M5) 자동 전환
    librosa의 pYIN 알고리즘을 사용하여 음원의 F0(기본 주파수)를 측정하고 중앙값을 반환합니다.
    """
    print(f"[*] 오디오 F0 프로파일링 분석 중... ({wav_path})")
    y, sr = librosa.load(wav_path, sr=16000)
    
    # pYIN은 음성 및 음악의 F0 추정에 매우 정밀한 확률론적 알고리즘입니다.
    f0, voiced_flag, _ = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'), # 약 65Hz (남성 극저음 하한)
        fmax=librosa.note_to_hz('C7'), 
        sr=sr
    )
    
    valid_f0 = f0[voiced_flag]
    if len(valid_f0) == 0:
        print("[경고] 유효한 F0를 찾지 못했습니다. 기본값(120Hz)을 적용합니다.")
        return 120.0
        
    median_f0 = float(np.median(valid_f0))
    return median_f0

def auto_select_preset(median_f0):
    """
    F0 중앙값을 기준으로 최적의 베이스 프리셋을 자동으로 매칭합니다.
    """
    if median_f0 < 115.0:
        preset = "voice_styles/M5.json"
        desc = "남성 극저음역대"
    elif 115.0 <= median_f0 < 130.0:
        preset = "voice_styles/M4.json"  # M2 또는 M4 중 안정적인 M4 채택
        desc = "남성 저음역대"
    else:
        preset = "voice_styles/M1.json"  # 130Hz 이상
        desc = "남성 일반/고음역대"
        
    print(f"[*] F0 중앙값: {median_f0:.1f}Hz -> {desc} 감지. '{preset}' 프리셋을 자동 선택합니다.")
    return preset


# =====================================================================
# 2. 하이브리드 손실 연산 (F0 & 고주파 마스킹 보강)
# =====================================================================
def compute_hybrid_loss(wavlm_synth_hidden, wavlm_target_hidden, synth_audio, target_audio, f0_val):
    """
    [구조적 변경 사항 2 반영] F0 기반 dynamic_weights 연산 및 고주파 노이즈 억제
    수학적 처리 구간: WavLM 다중 레이어 MSE와 RFFT(Real Fast Fourier Transform) 마스킹을 결합합니다.
    """
    # 1. WavLM 다중 레이어 가중치 연산 (WAVLM_LAYER_WEIGHTS)
    # L3: 화자의 기본 음색(Timbre) 아이덴티티 / L6, L9: 운율 및 발음(Prosody)
    layer_weights = {3: 0.7, 6: 0.2, 9: 0.1}
    wavlm_loss = 0.0
    for layer_idx, weight in layer_weights.items():
        # 각 레이어별 시간축(Time-axis) 특성을 반영하기 위해 통계량(평균) 기준으로 비교 (Content dependency 최소화)
        synth_mean = wavlm_synth_hidden[layer_idx].mean(dim=1)
        target_mean = wavlm_target_hidden[layer_idx].mean(dim=1)
        wavlm_loss += weight * F.mse_loss(synth_mean, target_mean)

    # 2. 고주파 디지털 노이즈(쇳소리) 억제 하이브리드 손실 (F0 연동)
    # RFFT 연산을 통해 오디오 신호를 주파수 도메인으로 변환
    synth_fft = torch.fft.rfft(synth_audio)
    target_fft = torch.fft.rfft(target_audio)
    freqs = torch.fft.rfftfreq(synth_audio.shape[-1], d=1/16000)

    # 4000Hz 이상의 고주파수 영역에 대한 L1 Loss 페널티 부여
    # 남성 저음의 경우 해당 대역에서 발생하는 불필요한 배음 증폭이 로봇음/쇳소리를 유발함
    hf_mask = (freqs > 4000.0).float().to(synth_audio.device)
    hf_loss = F.l1_loss(
        torch.abs(synth_fft) * hf_mask,
        torch.abs(target_fft) * hf_mask
    )

    # F0가 낮을수록(저음일수록) 고주파 페널티의 가중치를 증폭시킴 (역비례 스케일링)
    hf_penalty_weight = max(0.01, 0.05 * (115.0 / max(f0_val, 1.0)))
    
    total_loss = wavlm_loss + (hf_penalty_weight * hf_loss)
    return total_loss, wavlm_loss.item(), hf_loss.item()


# =====================================================================
# 3. 메인 최적화 파이프라인
# =====================================================================
def optimize_voice_style(target_wav_path, output_name, initial_lr=2e-4, max_steps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 대상 오디오 F0 프로파일링 및 프리셋 선정
    median_f0 = analyze_f0_profile(target_wav_path)
    base_preset_path = auto_select_preset(median_f0)
    
    # 2. 타겟 오디오 로드 및 WavLM 준비
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    wavlm_model.eval()
    
    target_audio, sr = torchaudio.load(target_wav_path)
    if sr != 16000:
        target_audio = torchaudio.transforms.Resample(sr, 16000)(target_audio)
    target_audio = target_audio.to(device)
    
    with torch.no_grad():
        target_outputs = wavlm_model(target_audio, output_hidden_states=True)
        wavlm_target_hidden = target_outputs.hidden_states

    # 3. 베이스 스타일 로드 및 학습 텐서 설정 (Supertonic3 명세: style_ttl 최적화)
    with open(base_preset_path, "r") as f:
        base_style = json.load(f)
    
    style_ttl = torch.tensor(base_style["style_ttl"], device=device, requires_grad=True)
    # style_dp(리듬 벡터)는 원래 Frozen이나, Warm-down 시 미세 정렬의 영향을 받도록 구성할 수 있음
    # 본 로직에서는 논문의 기본 방침을 따라 style_ttl만 업데이트하나, loss 표면이 평탄해져 전체 생성 품질이 올라갑니다.
    
    optimizer = torch.optim.Adam([style_ttl], lr=initial_lr)
    
    # [구조적 변경 사항 1 반영] Dynamic Warm-down 제어 변수
    warmdown_mode = False
    warmdown_steps_remaining = 50
    loss_threshold = 0.18

    print("\n=========================================================")
    print("[*] Supertonic3 스타일 임베딩 최적화를 시작합니다.")
    print("=========================================================\n")

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        
        # 가상의 TTS 합성 호출 (기존 helper.py의 함수 사용)
        # 텍스트는 음소 커버리지가 높은 pangram을 사용하는 것이 일반적입니다.
        synth_audio = generate_wav_from_style(style_ttl, "기본 합성용 텍스트입니다.")
        synth_audio = synth_audio.to(device)
        
        # 생성된 오디오의 WavLM 추출
        synth_outputs = wavlm_model(synth_audio, output_hidden_states=True)
        wavlm_synth_hidden = synth_outputs.hidden_states
        
        # 하이브리드 손실 연산
        loss, l3_loss_val, hf_loss_val = compute_hybrid_loss(
            wavlm_synth_hidden, wavlm_target_hidden, 
            synth_audio, target_audio, median_f0
        )
        
        loss.backward()
        optimizer.step()
        
        current_lr = optimizer.param_groups['lr']
        
        # [구조적 변경 사항 4 반영] 코랩 정규식 파싱 호환을 위한 고정 출력 포맷
        print(f"Step {step:04d} | LR: {current_lr:.6f} | Loss: {loss.item():.4f} | L3_MSE: {l3_loss_val:.4f} | HF_Pen: {hf_loss_val:.4f}")

        # [구조적 변경 사항 1 반영] Dynamic Warm-down (동적 위상 스무딩) 진입 판별
        if not warmdown_mode and l3_loss_val <= loss_threshold:
            warmdown_mode = True
            print("\n>>> [Phase Transition] Best L3 Loss Threshold(0.18) 도달!")
            print(">>> 즉시 종료하지 않고 Dynamic Warm-down 모드에 진입합니다.")
            print(">>> Learning Rate를 기존의 5% 수준으로 강제 드롭하여 50스텝 동안 가중치 미세 정렬(Smoothing)을 수행합니다.\n")
            
            # LR 5% 강제 드롭 연산 (예: 2e-4 -> 1e-5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.05
                
        # Warm-down 모드 카운트다운 처리
        if warmdown_mode:
            warmdown_steps_remaining -= 1
            if warmdown_steps_remaining <= 0:
                print("\n>>> [Early Stop] Warm-down 50스텝 미세 정렬 완료. 고주파 위상 왜곡을 최소화하고 안전하게 종료합니다.")
                break

    # 최종 결과 저장
    base_style["style_ttl"] = style_ttl.detach().cpu().numpy().tolist()
    os.makedirs("output_styles", exist_ok=True)
    out_path = f"output_styles/{output_name}.json"
    
    with open(out_path, "w") as f:
        json.load(base_style, f)
    print(f"\n[*] 최종 화자 임베딩 추출 완료: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_wav", type=str, required=True, help="클로닝할 대상 WAV 파일 경로")
    parser.add_argument("--name", type=str, required=True, help="출력될 스타일 파일 이름")
    parser.add_argument("--lr", type=float, default=2e-4, help="초기 학습률")
    parser.add_argument("--steps", type=int, default=1000, help="최대 최적화 스텝")
    args = parser.parse_args()
    
    optimize_voice_style(args.target_wav, args.name, args.lr, args.steps)
