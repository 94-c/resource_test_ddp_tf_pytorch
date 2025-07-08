# Resource Test DDP TensorFlow PyTorch

PyTorch와 TensorFlow를 사용한 **리소스 모니터링 및 성능 테스트** 도구입니다. CPU, 메모리, GPU 자원의 집약적 사용량을 측정하고 분산 학습 환경에서의 리소스 활용도를 분석할 수 있습니다.

## 🎯 주요 기능

- **CPU 집약적 테스트**: 행렬 연산, 병렬 처리를 통한 CPU 사용률 최대화
- **메모리 집약적 테스트**: 대용량 텐서 할당을 통한 메모리 사용량 최대화
- **GPU 메모리 테스트**: GPU 메모리 할당 및 사용량 측정
- **GPU 사용률 테스트**: 계산 집약적 GPU 연산을 통한 GPU 활용도 최대화
- **DDP 분산 학습 테스트**: 멀티 GPU 환경에서 분산 학습 성능 테스트
- **환경 진단**: GPU 환경 문제 해결을 위한 진단 도구

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 진단 (GPU 문제 해결용)
python check_env.py
```

### 2. 기본 테스트 실행

```bash
# PyTorch CPU 테스트 (60초)
python run_tests.py pytorch cpu --duration 60

# TensorFlow 메모리 테스트 (120초)
python run_tests.py tensorflow memory --duration 120

# PyTorch GPU 사용률 테스트 (600초 - DCGM 메트릭 대응)
python run_tests.py pytorch gpu_utilization --duration 600

# PyTorch DDP 분산 학습 테스트
python run_tests.py pytorch ddp_training --epochs 10 --batch-size 32
```

### 3. 집약적 GPU 테스트 (DCGM 메트릭용)

```bash
# DCGM 메트릭에 나타나도록 하는 집약적 GPU 테스트 (10분 이상)
python run_tests.py --intensive-gpu

# 개별 프레임워크 집약적 테스트
python run_tests.py pytorch gpu_utilization --duration 600
python run_tests.py tensorflow gpu_utilization --duration 600
```

### 4. 전체 테스트 실행

```bash
# 모든 PyTorch 테스트 순차 실행
python run_tests.py --all-pytorch --duration 90

# 모든 TensorFlow 테스트 순차 실행
python run_tests.py --all-tensorflow --duration 90

# 테스트 목록 확인
python run_tests.py --list
```

## 📁 프로젝트 구조

```
resource_test_ddp_tf_pytorch/
├── README.md                   # 프로젝트 문서
├── requirements.txt            # 의존성 목록
├── run_tests.py               # 메인 테스트 실행기
├── check_env.py              # 환경 진단 도구
├── utils/
│   └── resource_monitor.py   # 리소스 모니터링 유틸리티
├── docs/
│   └── grafana_dashboard.json # Grafana 대시보드 설정
├── pytorch_tests/
│   ├── cpu_test.py           # PyTorch CPU 테스트
│   ├── memory_test.py        # PyTorch 메모리 테스트
│   ├── gpu_memory_test.py    # PyTorch GPU 메모리 테스트
│   ├── gpu_utilization_test.py # PyTorch GPU 사용률 테스트
│   └── ddp_training_test.py  # PyTorch DDP 분산 학습 테스트
└── tensorflow_tests/
    ├── cpu_test.py           # TensorFlow CPU 테스트
    ├── memory_test.py        # TensorFlow 메모리 테스트
    ├── gpu_memory_test.py    # TensorFlow GPU 메모리 테스트
    └── gpu_utilization_test.py # TensorFlow GPU 사용률 테스트
```

## 🧪 테스트 타입별 상세 설명

### CPU 테스트
- **목적**: CPU 사용률 최대화 및 멀티코어 활용도 측정
- **작업**: 대용량 행렬 연산, 병렬 처리, 수학적 집약적 연산
- **모니터링**: CPU 사용률(%), 코어별 사용량, 평균/최대 코어 사용량

### 메모리 테스트
- **목적**: 메모리 사용량 최대화 및 메모리 압박 상황 시뮬레이션
- **작업**: 대용량 텐서 할당, 점진적 메모리 할당, 메모리 압박 테스트
- **모니터링**: 메모리 사용률(%), 메모리 사용량(GB), 평균/최대 메모리 사용량

### GPU 메모리 테스트
- **목적**: GPU 메모리 사용량 최대화 및 GPU 메모리 한계 측정
- **작업**: 대용량 GPU 텐서 할당, 모델 로딩, GPU 메모리 스트레스 테스트
- **모니터링**: GPU 메모리 사용량(GB), 다중 GPU 지원

### GPU 사용률 테스트
- **목적**: GPU 계산 사용률 최대화 및 GPU 성능 벤치마킹
- **작업**: 
  - 병렬 행렬 연산 (GEMM)
  - 컨볼루션 연산 (CNN 워크로드)
  - 트랜스포머 연산 (어텐션 메커니즘)
  - FFT 연산 (고속 푸리에 변환)
- **특징**: 
  - DCGM 메트릭 (`DCGM_FI_PROF_GR_ENGINE_ACTIVE`) 대응
  - 멀티 GPU 지원 (각 GPU에서 2개 워크로드 스레드)
  - 지속적 높은 사용률 유지 (90% 이상)
- **모니터링**: GPU 사용률(%), 멀티 GPU 개별 모니터링

### DDP 분산 학습 테스트
- **목적**: 멀티 GPU 환경에서 분산 학습을 통한 종합적 리소스 테스트
- **작업**:
  - LazyFakeDataset 사용 (실제 데이터 대신 무작위 텐서 생성)
  - ResNet-50 모델 기반 분산 학습
  - 자동 모델 저장 및 체크포인트 생성
- **특징**:
  - 멀티 GPU 자동 분산 학습
  - GPU 없을 시 CPU 모드 자동 전환
  - 중간 체크포인트 자동 저장
- **모니터링**: CPU 코어 사용량, 메모리(GB), GPU 메모리(GB), GPU 사용률(%)

## ⚙️ 명령줄 옵션

### 공통 옵션
```bash
--duration N          # 테스트 지속 시간 (초, 기본값: 60)
--list               # 사용 가능한 테스트 목록 출력
--all-pytorch        # 모든 PyTorch 테스트 순차 실행
--all-tensorflow     # 모든 TensorFlow 테스트 순차 실행
--intensive-gpu      # 집약적 GPU 테스트 (DCGM 메트릭용, 10분 이상)
```

### CPU 테스트 옵션
```bash
--matrix-iterations N # 행렬 연산 반복 횟수
--matrix-size N       # 행렬 크기
--skip-parallel       # 병렬 처리 테스트 건너뛰기
```

### 메모리 테스트 옵션
```bash
--num-tensors N       # 할당할 텐서 수
--tensor-size N       # 텐서 크기
--skip-progressive    # 점진적 할당 테스트 건너뛰기
```

### GPU 테스트 옵션
```bash
--stress-level N      # 스트레스 테스트 레벨 (1-5)
--skip-conv          # 컨볼루션 테스트 건너뛰기
--skip-training      # 학습 테스트 건너뛰기
```

### DDP 분산 학습 테스트 옵션
```bash
--num-samples N       # 데이터셋 샘플 수 (기본값: 100,000)
--epochs N           # 학습 에폭 수 (기본값: 100)
--batch-size N       # 배치 크기 (기본값: 32)
--lr FLOAT           # 학습률 (기본값: 0.01)
--num-classes N      # 클래스 수 (기본값: 100)
--save-dir PATH      # 모델 저장 디렉토리
--force-single-gpu   # 단일 GPU 강제 사용
--force-cpu          # CPU 강제 사용
```

## 📊 실시간 모니터링

각 테스트는 실시간으로 다음 리소스를 모니터링합니다:

- **CPU**: 사용률(%), 코어 사용량 (예: 6.33 cores)
- **메모리**: 사용률(%), 사용량(GB) (예: 5.38 GB)
- **GPU**: 메모리 사용량(GB) (예: 2.15 GB), 사용률(%) (예: 85.3%)
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

## 🔧 환경 진단 및 문제 해결

### GPU 환경 문제 진단
```bash
# 환경 진단 실행
python check_env.py
```

### 일반적인 GPU 문제 해결
```bash
# 1. 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=1

# 2. NVML 문제 해결
export NVIDIA_DISABLE_REQUIRE=1

# 3. 안전한 배치 크기로 테스트
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python run_tests.py pytorch gpu_utilization --duration 600
```

### DCGM 메트릭 관련 팁
- **최소 실행 시간**: 10분 이상 권장 (DCGM 메트릭 수집 주기 고려)
- **메트릭 타입**: `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_DEV_GPU_UTIL`
- **집약적 테스트**: `--intensive-gpu` 옵션 사용

## 💡 사용 예시

### 개발 환경 성능 테스트
```bash
# 빠른 성능 체크 (각 테스트 60초)
python run_tests.py pytorch cpu --duration 60
python run_tests.py pytorch memory --duration 60
python run_tests.py pytorch gpu_utilization --duration 60
```

### 프로덕션 환경 스트레스 테스트
```bash
# 장시간 스트레스 테스트 (각 테스트 10분)
python run_tests.py --intensive-gpu
python run_tests.py pytorch ddp_training --epochs 50 --batch-size 64
```

### 리소스 벤치마킹
```bash
# 전체 PyTorch 테스트 스위트 실행
python run_tests.py --all-pytorch --duration 300

# 전체 TensorFlow 테스트 스위트 실행
python run_tests.py --all-tensorflow --duration 300
```

## 🛠️ 기술 스택

- **딥러닝 프레임워크**: PyTorch 2.0+, TensorFlow 2.12+
- **리소스 모니터링**: psutil, nvidia-ml-py3, GPUtil
- **분산 학습**: PyTorch DDP (DistributedDataParallel)
- **GPU 메트릭**: DCGM 호환 메트릭 지원
- **시각화**: matplotlib, tqdm

## 🎯 사용 사례

1. **GPU 서버 성능 검증**: 새로운 GPU 서버의 성능 및 안정성 검증
2. **리소스 모니터링**: 실제 워크로드 실행 시 리소스 사용량 패턴 분석
3. **분산 학습 환경 테스트**: 멀티 GPU 분산 학습 환경의 성능 측정
4. **DCGM 메트릭 검증**: GPU 모니터링 시스템의 메트릭 정확성 검증
5. **시스템 벤치마킹**: 다양한 하드웨어 구성에서의 성능 비교

---

**문제 발생 시 `python check_env.py`를 먼저 실행하여 환경을 진단하세요.**