# Resource Test DDP TensorFlow PyTorch

이 저장소는 PyTorch와 TensorFlow를 사용하여 **리소스 모니터링 및 워크로드 테스트**를 위한 코드를 포함합니다. 각 리소스별(CPU, Memory, GPU Memory, GPU Utilization)로 집중적인 테스트를 수행하고, **DDP 분산 학습**을 통해 시스템의 리소스 사용량을 모니터링할 수 있습니다.

## 📋 목적

- **CPU 사용률 테스트**: 행렬 연산, 병렬 처리를 통한 CPU 집약적 작업
- **메모리 사용량 테스트**: 대용량 텐서 할당을 통한 메모리 집약적 작업  
- **GPU 메모리 테스트**: GPU 메모리 할당 및 대용량 모델 로딩
- **GPU 사용률 테스트**: 계산 집약적 GPU 연산을 통한 GPU 활용도 최대화
- **DDP 분산 학습 테스트**: 멀티 GPU 환경에서 분산 학습을 통한 종합적 리소스 테스트

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 개별 테스트 실행

```bash
# PyTorch CPU 테스트 (60초간)
python run_tests.py pytorch cpu --duration 60

# TensorFlow 메모리 테스트 (120초간, 텐서 30개)
python run_tests.py tensorflow memory --duration 120 --num-tensors 30

# PyTorch GPU 메모리 테스트
python run_tests.py pytorch gpu_memory --tensor-size 3000

# TensorFlow GPU 사용률 테스트
python run_tests.py tensorflow gpu_utilization --matrix-ops 500

# PyTorch DDP 분산 학습 테스트 (NEW!)
python run_tests.py pytorch ddp_training --epochs 50 --batch-size 32
```

### 3. 전체 테스트 실행

```bash
# 모든 PyTorch 테스트 순차 실행 (DDP 포함)
python run_tests.py --all-pytorch --duration 90

# 모든 TensorFlow 테스트 순차 실행
python run_tests.py --all-tensorflow --duration 90
```

### 4. 테스트 목록 확인

```bash
python run_tests.py --list
```

## 📁 프로젝트 구조

```
resource_test_ddp_tf_pytorch/
├── README.md                              # 프로젝트 문서
├── requirements.txt                       # 의존성 목록
├── run_tests.py                          # 메인 테스트 실행 스크립트
├── utils/
│   └── resource_monitor.py               # 리소스 모니터링 유틸리티
├── pytorch_tests/
│   ├── cpu_test.py                       # PyTorch CPU 집약적 테스트
│   ├── memory_test.py                    # PyTorch 메모리 집약적 테스트
│   ├── gpu_memory_test.py               # PyTorch GPU 메모리 테스트
│   ├── gpu_utilization_test.py          # PyTorch GPU 사용률 테스트
│   └── ddp_training_test.py             # PyTorch DDP 분산 학습 테스트 (NEW!)
└── tensorflow_tests/
    ├── cpu_test.py                       # TensorFlow CPU 집약적 테스트
    ├── memory_test.py                    # TensorFlow 메모리 집약적 테스트
    ├── gpu_memory_test.py               # TensorFlow GPU 메모리 테스트
    └── gpu_utilization_test.py          # TensorFlow GPU 사용률 테스트
```

## 🧪 테스트 상세 설명

### CPU 테스트
- **목적**: CPU 사용률 최대화
- **작업**: 대용량 행렬 연산, 병렬 처리, 수학적 집약적 연산
- **모니터링**: CPU 사용률, 코어별 사용량

### 메모리 테스트
- **목적**: 메모리 사용량 최대화
- **작업**: 대용량 텐서 할당, 메모리 누수 시뮬레이션, 점진적 할당
- **모니터링**: 메모리 사용률, 메모리 사용량(GB)

### GPU 메모리 테스트
- **목적**: GPU 메모리 사용량 최대화
- **작업**: GPU 텐서 할당, 대용량 모델 로딩, 스트레스 테스트
- **모니터링**: GPU 메모리 사용량(GB), 다중 GPU 지원

### GPU 사용률 테스트
- **목적**: GPU 계산 사용률 최대화
- **작업**: 병렬 행렬 연산, 컨볼루션, 트랜스포머, FFT 연산
- **모니터링**: GPU 사용률(%), GPU 활용도

### DDP 분산 학습 테스트 (NEW!)
- **목적**: 멀티 GPU 환경에서 분산 학습을 통한 종합적 리소스 테스트
- **작업**: 
  - LazyFakeDataset 사용 (실제 이미지 대신 무작위 텐서 생성)
  - ResNet-50 모델 기반 분산 학습
  - 100,000개 샘플, 100 에폭 학습 (설정 가능)
  - 자동 모델 저장 (model_final.pth)
- **특징**:
  - 멀티 GPU 자동 분산 학습
  - GPU 없을 시 CPU 모드 자동 전환
  - 실시간 리소스 모니터링
  - 중간 체크포인트 자동 저장
- **모니터링**: CPU 코어 사용량, 메모리(GB), GPU 메모리(GB), GPU 사용률(%)

## ⚙️ 명령줄 옵션

### 공통 옵션
- `--duration N`: 테스트 지속 시간 (초, 기본값: 60)
- `--list`: 사용 가능한 테스트 목록 출력

### CPU 테스트 옵션
- `--matrix-iterations N`: 행렬 연산 반복 횟수
- `--matrix-size N`: 행렬 크기
- `--skip-parallel`: 병렬 처리 테스트 건너뛰기

### 메모리 테스트 옵션
- `--num-tensors N`: 할당할 텐서 수
- `--tensor-size N`: 텐서 크기
- `--skip-progressive`: 점진적 할당 테스트 건너뛰기

### GPU 테스트 옵션
- `--stress-level N`: 스트레스 테스트 레벨 (1-5)
- `--skip-conv`: 컨볼루션 테스트 건너뛰기
- `--skip-training`: 학습 테스트 건너뛰기

### ⚡ 집약적 GPU 테스트 (DCGM 메트릭 대응) - NEW!
DCGM (Data Center GPU Manager) 메트릭에 나타나도록 GPU 사용률을 지속적으로 높게 유지하는 집약적 테스트입니다.
- `--duration N`: 테스트 지속 시간 (초, 기본값: 600초/10분)
- **최소 권장 시간**: 10분 이상 (DCGM 메트릭 수집 주기 고려)
- **지원 메트릭**: `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_DEV_GPU_UTIL`
- **멀티 GPU 지원**: 모든 GPU에서 동시에 워크로드 실행

#### 집약적 GPU 테스트 사용법
```bash
# 간편한 집약적 GPU 테스트 (10분 이상)
python run_tests.py --intensive-gpu

# 개별 프레임워크 집약적 테스트 (10분)
python run_tests.py pytorch gpu_utilization --duration 600
python run_tests.py tensorflow gpu_utilization --duration 600

# 장시간 집약적 테스트 (30분)
python run_tests.py pytorch gpu_utilization --duration 1800
```

#### 집약적 GPU 테스트 특징
- **지속적 높은 사용률**: GPU 사용률을 90% 이상 유지
- **멀티 GPU 지원**: 각 GPU에서 2개 워크로드 스레드 실행
- **다양한 연산**: 컨볼루션, 트랜스포머, 행렬 연산, FFT 등 혼합
- **메모리 관리**: 자동 메모리 압박 방지 및 배치 크기 조정
- **실시간 모니터링**: 남은 시간 및 진행 상황 표시
- **안전한 중단**: Ctrl+C로 언제든지 중단 가능

### DDP 분산 학습 테스트 옵션 (NEW!)
- `--num-samples N`: 데이터셋 샘플 수 (기본값: 100,000)
- `--epochs N`: 학습 에폭 수 (기본값: 100)
- `--batch-size N`: 배치 크기 (기본값: 32)
- `--lr FLOAT`: 학습률 (기본값: 0.01)
- `--num-classes N`: 클래스 수 (기본값: 100)
- `--save-dir PATH`: 모델 저장 디렉토리 (기본값: ./saved_models)
- `--force-single-gpu`: 단일 GPU 강제 사용
- `--force-cpu`: CPU 강제 사용

## 📊 모니터링 기능

각 테스트는 실시간으로 다음 리소스를 모니터링합니다:

- **CPU**: 사용률(%), **코어 사용량** (예: 6.33 cores)
- **메모리**: 사용률(%), **사용량(GB)** (예: 5.38 GB)
- **GPU**: **메모리 사용량(GB)** (예: 2.15 GB), **사용률(%)** (예: 85.3%)
- **실시간 그래프**: 테스트 진행 중 리소스 사용량 변화

### 모니터링 출력 예시
```
==================================================
RESOURCE MONITORING SUMMARY
==================================================
Duration: 172.5 seconds
CPU Usage - Avg: 6.33 cores, Max: 9.32 cores
Memory Usage - Avg: 78.7%, Max: 80.5%
Memory Usage - Avg: 5.38GB, Max: 6.14GB
GPU Usage:
  GPU 0 - Util Avg: 85.3%, Max: 98.1%
  GPU 0 - Memory Avg: 2.15GB, Max: 3.42GB
==================================================
```

## 📋 시스템 요구사항

### 기본 요구사항
- Python 3.8+
- NumPy
- psutil

### PyTorch 테스트
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

### TensorFlow 테스트  
- TensorFlow >= 2.12.0

### GPU 테스트 (선택사항)
- NVIDIA GPU
- CUDA Toolkit
- nvidia-ml-py3
- GPUtil

### DDP 분산 학습 테스트
- 멀티 GPU 환경 (2개 이상 권장)
- NCCL (GPU 간 통신)
- 단일 GPU 또는 CPU 모드도 지원

## 🎯 사용 시나리오

### 1. 시스템 성능 벤치마킹
```bash
# 전체 시스템 리소스 테스트
python run_tests.py --all-pytorch --duration 300
python run_tests.py --all-tensorflow --duration 300
```

### 2. 특정 리소스 스트레스 테스트
```bash
# CPU 집약적 작업 테스트
python run_tests.py pytorch cpu --duration 600 --matrix-size 3000

# GPU 메모리 한계 테스트
python run_tests.py pytorch gpu_memory --stress-level 5

# DCGM 메트릭 대응 집약적 GPU 테스트
python run_tests.py --intensive-gpu
```

### 3. 분산 학습 워크로드 테스트
```bash
# 멀티 GPU 분산 학습 테스트
python run_tests.py pytorch ddp_training --epochs 100 --batch-size 64

# CPU 모드 분산 학습 테스트
python pytorch_tests/ddp_training_test.py --force-cpu --epochs 10

# 단일 GPU 학습 테스트
python pytorch_tests/ddp_training_test.py --force-single-gpu --epochs 50
```

### 4. 워크로드 모니터링
```bash
# 실제 워크로드와 유사한 패턴으로 테스트
python run_tests.py tensorflow gpu_utilization --conv-iterations 1000

# DCGM 메트릭 수집을 위한 장시간 집약적 테스트
python run_tests.py pytorch gpu_utilization --duration 1800  # 30분
```

## 🔧 개발자 가이드

### 새로운 테스트 추가
1. `pytorch_tests/` 또는 `tensorflow_tests/` 디렉토리에 새 테스트 파일 생성
2. `utils/resource_monitor.py`를 사용하여 리소스 모니터링 구현
3. `run_tests.py`의 테스트 목록에 새 테스트 추가

### 커스텀 모니터링
`ResourceMonitor` 클래스를 확장하여 추가 메트릭 모니터링 가능:

```python
from utils.resource_monitor import ResourceMonitor

monitor = ResourceMonitor(interval=0.5)
monitor.start_monitoring()
# ... 테스트 코드 ...
monitor.stop_monitoring()
monitor.print_summary()
```

### LazyFakeDataset 사용법
```python
from pytorch_tests.ddp_training_test import LazyFakeDataset

# 가짜 데이터셋 생성 (메모리 효율적)
dataset = LazyFakeDataset(
    num_samples=10000,
    image_size=(3, 224, 224),
    num_classes=100
)
```

## 📈 출력 예시

### 일반 테스트 실행
```
🧪 리소스 테스트 실행기
PyTorch와 TensorFlow를 사용한 CPU, 메모리, GPU 리소스 테스트
============================================================

🚀 실행 중: pytorch_tests/cpu_test.py
   명령어: python pytorch_tests/cpu_test.py --duration 60

💻 시스템 정보:
  OS: macOS 14.0
  CPU: Apple M2 Pro (12 코어)
  Memory: 32.0 GB
  GPU: Apple M2 Pro

🔥 PyTorch CPU 집약적 테스트 시작 (지속 시간: 60초)
============================================================

1. 행렬 연산 테스트
  반복 0/100 완료
  반복 20/100 완료
  ...

리소스 모니터링 요약:
  평균 CPU 사용률: 87.3%
  최대 CPU 사용률: 95.2%
  평균 메모리 사용률: 45.1%
  테스트 지속 시간: 60.2초

✅ 테스트가 성공적으로 완료되었습니다!
```

### 집약적 GPU 테스트 실행 (DCGM 메트릭 대응)
```
🧪 리소스 테스트 실행기
PyTorch와 TensorFlow를 사용한 CPU, 메모리, GPU 리소스 테스트
============================================================

🔥 집약적 GPU 테스트를 실행합니다...
   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타나도록 10분 이상 실행됩니다.
   Ctrl+C로 중단할 수 있습니다.

============================================================
PyTorch GPU 집약적 테스트 시작
============================================================

🚀 실행 중: pytorch_tests/gpu_utilization_test.py
   명령어: python pytorch_tests/gpu_utilization_test.py --duration 600

💻 시스템 정보:
  OS: Linux 5.4.0
  CPU: Intel Xeon Gold 6230 (40 코어)
  Memory: 512.0 GB
  GPU: NVIDIA A100 80GB (4개)

🎮 GPU 정보:
  GPU 0: NVIDIA A100 80GB PCIe
    메모리: 81.0 GB
    멀티프로세서: 108
  GPU 1: NVIDIA A100 80GB PCIe
    메모리: 81.0 GB
    멀티프로세서: 108
  [...]

🎯 4개 GPU에서 10분 동안 집약적 워크로드 시작
   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...
============================================================

🚀 GPU 0 워크로드 0 시작
🚀 GPU 0 워크로드 1 시작
🚀 GPU 1 워크로드 0 시작
🚀 GPU 1 워크로드 1 시작
[...]

  GPU 0 워크로드 0: 20 반복 완료, 남은 시간: 538.2초, Loss: 4.6823
  GPU 1 워크로드 0: 18 반복 완료, 남은 시간: 541.8초, Loss: 4.7291
  GPU 0 워크로드 1: 19 반복 완료, 남은 시간: 540.1초, Loss: 4.6547
  [...]

⏱️  진행 중... 남은 시간: 480.3초
⏱️  진행 중... 남은 시간: 470.1초
[...]

✅ GPU 0 워크로드 0 완료 (156 반복)
✅ GPU 0 워크로드 1 완료 (148 반복)
✅ GPU 1 워크로드 0 완료 (152 반복)
✅ GPU 1 워크로드 1 완료 (145 반복)
[...]

🏁 모든 GPU 워크로드 완료

==================================================
RESOURCE MONITORING SUMMARY
==================================================
Duration: 600.2 seconds
CPU Usage - Avg: 18.7 cores, Max: 28.4 cores
Memory Usage - Avg: 87.3%, Max: 91.2%
Memory Usage - Avg: 448.2GB, Max: 467.1GB
GPU Usage:
  GPU 0 - Util Avg: 94.8%, Max: 99.2%
  GPU 0 - Memory Avg: 72.1GB, Max: 78.9GB
  GPU 1 - Util Avg: 93.1%, Max: 98.7%
  GPU 1 - Memory Avg: 71.8GB, Max: 78.4GB
  [...]
==================================================

🎯 PyTorch GPU 사용률 집약적 테스트 완료!
   총 실행 시간: 600.2초
============================================================

✅ 모든 집약적 GPU 테스트가 성공적으로 완료되었습니다!
```

### DDP 분산 학습 테스트 실행
```
🚀 2개 GPU로 DDP 학습 시작
============================================================
Running DDP training on rank 0
Running DDP training on rank 1

🔥 PyTorch DDP 학습 시작 (GPU 2개 사용)
   샘플 수: 100,000
   에폭 수: 100
   배치 크기: 32
   클래스 수: 100
   디바이스: cuda:0
============================================================

✅ Epoch 1/100 완료 - 평균 Loss: 4.8756, 시간: 45.32초
🎯 모델 저장 완료: ./saved_models/model_final.pth

✅ DDP 학습이 성공적으로 완료되었습니다!
```

## 📦 저장된 파일

DDP 분산 학습 테스트 실행 시 다음 파일들이 생성됩니다:

- `saved_models/model_final.pth`: 최종 학습된 모델
- `saved_models/checkpoint_epoch_*.pth`: 중간 체크포인트들 (10 에폭마다)
