# MIG 환경에서 GPU 사용률 메트릭 계산 가이드

## 📋 목차
1. [MIG(Multi-Instance GPU) 개요](#mig-multi-instance-gpu-개요)
2. [DCGM(Data Center GPU Manager) 이해](#dcgm-data-center-gpu-manager-이해)
3. [GPU 사용률 계산 방법의 차이점](#gpu-사용률-계산-방법의-차이점)
4. [프로메테우스 메트릭 쿼리](#프로메테우스-메트릭-쿼리)
5. [실제 사용 예시](#실제-사용-예시)
6. [Grafana 대시보드 구성](#grafana-대시보드-구성)
7. [문제 해결](#문제-해결)

---

## 🎯 MIG(Multi-Instance GPU) 개요

### MIG란?
**Multi-Instance GPU**는 단일 물리적 GPU를 여러 개의 독립적인 GPU 인스턴스로 분할하는 NVIDIA의 기술입니다.

### 표준 MIG 프로파일 (A100 기준)
```
GPU Instance Profile    | 슬라이스 | 가중치 | 메모리   | SM 개수
1g.10gb                | 1/7      | 0.143  | 10GB     | 14
2g.20gb                | 2/7      | 0.286  | 20GB     | 28  
3g.40gb                | 3/7      | 0.429  | 40GB     | 42
4g.40gb                | 4/7      | 0.571  | 40GB     | 56
7g.80gb                | 7/7      | 1.000  | 80GB     | 108
```

---

## 🔧 DCGM(Data Center GPU Manager) 이해

### DCGM이란?
**Data Center GPU Manager(DCGM)**는 NVIDIA에서 제공하는 데이터센터 환경에서 GPU 모니터링, 관리, 진단을 위한 도구입니다. 특히 MIG 환경에서 GPU 리소스를 정확히 모니터링하기 위해 필수적입니다.

### 🎯 DCGM의 주요 기능

#### 1. GPU 메트릭 수집
- **실시간 모니터링**: GPU 사용률, 메모리 사용량, 온도 등
- **MIG 인스턴스 추적**: 각 MIG 인스턴스별 개별 메트릭
- **성능 카운터**: 상세한 GPU 성능 지표

#### 2. 시스템 상태 관리
- **헬스 체크**: GPU 하드웨어 상태 모니터링
- **오류 감지**: GPU 오류 및 경고 알림
- **정책 기반 관리**: 임계값 설정 및 자동 대응

#### 3. 프로메테우스 연동
- **메트릭 익스포트**: 프로메테우스 형식으로 메트릭 제공
- **라벨링**: MIG 인스턴스별 구분 가능한 라벨 제공
- **스케일링**: 대규모 GPU 클러스터 지원

### 🏗️ MIG 환경에서의 DCGM 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    물리적 GPU (A100)                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   MIG Instance  │   MIG Instance  │      MIG Instance       │
│    1g.10gb      │    2g.20gb      │       3g.40gb          │
│                 │                 │                         │
│  ┌─────────────┐│  ┌─────────────┐│   ┌─────────────────┐   │
│  │ DCGM Agent  ││  │ DCGM Agent  ││   │   DCGM Agent    │   │
│  │  Metrics    ││  │  Metrics    ││   │    Metrics      │   │
│  └─────────────┘│  └─────────────┘│   └─────────────────┘   │
└─────────────────┴─────────────────┴─────────────────────────┘
           │                 │                       │
           ▼                 ▼                       ▼
    ┌─────────────────────────────────────────────────────────┐
    │                DCGM Exporter                            │
    │  - GPU_I_PROFILE="1g.10gb"                              │
    │  - GPU_I_PROFILE="2g.20gb"                              │
    │  - GPU_I_PROFILE="3g.40gb"                              │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   Prometheus                            │
    │  - DCGM_FI_PROF_GR_ENGINE_ACTIVE                        │
    │  - DCGM_FI_PROF_PCIE_TX_BYTES                           │
    │  - DCGM_FI_PROF_PCIE_RX_BYTES                           │
    │  - DCGM_FI_DEV_MEM_COPY_UTIL                            │
    └─────────────────────────────────────────────────────────┘
```

### 📊 주요 DCGM 메트릭 (MIG 환경)

#### 1. GPU 사용률 메트릭
```promql
# GPU 엔진 활성도 (가장 중요한 메트릭)
DCGM_FI_PROF_GR_ENGINE_ACTIVE{GPU_I_PROFILE="1g.10gb"}

# SM(Streaming Multiprocessor) 활성도
DCGM_FI_PROF_SM_ACTIVE{GPU_I_PROFILE="2g.20gb"}

# 텐서 코어 활성도
DCGM_FI_PROF_TENSOR_ACTIVE{GPU_I_PROFILE="3g.40gb"}
```

#### 2. 메모리 사용률 메트릭
```promql
# 메모리 복사 활용률
DCGM_FI_DEV_MEM_COPY_UTIL{GPU_I_PROFILE="1g.10gb"}

# 메모리 사용량 (바이트)
DCGM_FI_DEV_FB_USED{GPU_I_PROFILE="2g.20gb"}

# 메모리 여유 공간 (바이트)
DCGM_FI_DEV_FB_FREE{GPU_I_PROFILE="3g.40gb"}
```

#### 3. 네트워크 및 I/O 메트릭
```promql
# PCIe 송신 바이트
DCGM_FI_PROF_PCIE_TX_BYTES{GPU_I_PROFILE="1g.10gb"}

# PCIe 수신 바이트
DCGM_FI_PROF_PCIE_RX_BYTES{GPU_I_PROFILE="2g.20gb"}

# NVLink 대역폭 사용률
DCGM_FI_PROF_NVLINK_TX_BYTES{GPU_I_PROFILE="3g.40gb"}
```

#### 4. 온도 및 전력 메트릭
```promql
# GPU 온도 (°C)
DCGM_FI_DEV_GPU_TEMP{GPU_I_PROFILE="1g.10gb"}

# 전력 소비 (W)
DCGM_FI_DEV_POWER_USAGE{GPU_I_PROFILE="2g.20gb"}

# 팬 속도 (%)
DCGM_FI_DEV_FAN_SPEED{GPU_I_PROFILE="3g.40gb"}
```

### 🚀 DCGM 설정 및 배포

#### 1. Kubernetes 환경에서 DCGM 배포
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: gpu-operator-resources
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
        ports:
        - containerPort: 9400
          name: http-metrics
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        - name: DCGM_EXPORTER_KUBERNETES
          value: "true"
        - name: DCGM_EXPORTER_COLLECTORS
          value: "/etc/dcgm-exporter/dcp-metrics-included.csv"
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        securityContext:
          privileged: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      hostNetwork: true
      hostPID: true
```

#### 2. DCGM 설정 커스터마이징
```csv
# dcp-metrics-included.csv 예시
# 필요한 메트릭만 선택하여 성능 최적화
DCGM_FI_PROF_GR_ENGINE_ACTIVE, gauge, GPU 엔진 활성도
DCGM_FI_PROF_SM_ACTIVE, gauge, SM 활성도
DCGM_FI_PROF_TENSOR_ACTIVE, gauge, 텐서 코어 활성도
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, 메모리 복사 활용률
DCGM_FI_DEV_FB_USED, gauge, 메모리 사용량
DCGM_FI_DEV_FB_FREE, gauge, 메모리 여유 공간
DCGM_FI_PROF_PCIE_TX_BYTES, counter, PCIe 송신 바이트
DCGM_FI_PROF_PCIE_RX_BYTES, counter, PCIe 수신 바이트
DCGM_FI_DEV_GPU_TEMP, gauge, GPU 온도
DCGM_FI_DEV_POWER_USAGE, gauge, 전력 소비
```

#### 3. 프로메테우스 설정
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'dcgm-exporter'
    static_configs:
      - targets: ['dcgm-exporter:9400']
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: /metrics
```

### 📋 DCGM 메트릭 해석 가이드

#### 1. 메트릭 값 범위 이해
| 메트릭 | 단위 | 범위 | 의미 |
|--------|------|------|------|
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | % | 0-100 | GPU 엔진 활성도 |
| `DCGM_FI_PROF_SM_ACTIVE` | % | 0-100 | SM 활성도 |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | % | 0-100 | 메모리 복사 활용률 |
| `DCGM_FI_DEV_FB_USED` | MB | 0-MAX | 메모리 사용량 |
| `DCGM_FI_DEV_GPU_TEMP` | °C | 0-100+ | GPU 온도 |
| `DCGM_FI_DEV_POWER_USAGE` | W | 0-MAX | 전력 소비 |

#### 2. MIG 인스턴스 식별
```promql
# 라벨을 통한 MIG 인스턴스 구분
DCGM_FI_PROF_GR_ENGINE_ACTIVE{
  GPU_I_PROFILE="1g.10gb",
  GPU_I_ID="0",
  DCGM_FI_DRIVER_VERSION="530.30.02"
}
```

#### 3. 정상적인 메트릭 값 기준
- **GPU 사용률**: 80% 이상 시 고부하
- **메모리 사용률**: 90% 이상 시 주의 필요
- **온도**: 80°C 이상 시 경고
- **전력 소비**: 카드별 TDP 대비 90% 이상 시 주의

### 🔍 DCGM 문제 해결

#### 1. 메트릭이 수집되지 않는 경우
```bash
# DCGM 서비스 상태 확인
systemctl status dcgm

# DCGM 프로세스 확인
ps aux | grep dcgm

# DCGM 로그 확인
journalctl -u dcgm -f
```

#### 2. MIG 인스턴스가 인식되지 않는 경우
```bash
# MIG 모드 확인
nvidia-smi -q | grep -i mig

# MIG 인스턴스 목록 확인
nvidia-smi mig -lgip

# DCGM 필드 확인
dcgmi discovery -l
```

#### 3. 메트릭 값이 부정확한 경우
```bash
# DCGM 재시작
systemctl restart dcgm

# 메트릭 캐시 클리어
dcgmi stats -j 0 --reset

# 강제 메트릭 업데이트
dcgmi stats -j 0 --update
```

### 💡 DCGM 성능 최적화 팁

#### 1. 메트릭 수집 주기 조정
```yaml
# 성능 최적화를 위한 수집 주기 설정
DCGM_EXPORTER_INTERVAL: "30s"  # 기본값보다 길게
DCGM_EXPORTER_KUBERNETES_GPU_ID_TYPE: "device-name"
```

#### 2. 필요한 메트릭만 수집
```bash
# 커스텀 메트릭 설정 파일 생성
cat > custom-metrics.csv << EOF
DCGM_FI_PROF_GR_ENGINE_ACTIVE, gauge, GPU 엔진 활성도
DCGM_FI_DEV_FB_USED, gauge, 메모리 사용량
DCGM_FI_DEV_GPU_TEMP, gauge, GPU 온도
EOF
```

#### 3. 메트릭 라벨 최적화
```promql
# 불필요한 라벨 제거로 성능 향상
DCGM_FI_PROF_GR_ENGINE_ACTIVE{GPU_I_PROFILE="1g.10gb"}
```

---

## 📊 GPU 사용률 계산 방법의 차이점

### 1. 슬라이스 크기 차이

**MIG 환경에서는 각 인스턴스가 차지하는 물리적 GPU 리소스가 다릅니다:**

- **1g 인스턴스**: 전체 GPU의 1/7 (14.3%)
- **2g 인스턴스**: 전체 GPU의 2/7 (28.6%)
- **3g 인스턴스**: 전체 GPU의 3/7 (42.9%)

### 2. 영향력 차이

**예시 시나리오:**
```
GPU Instance 0 (1g.10gb): 90% 사용률
GPU Instance 1 (2g.20gb): 60% 사용률
GPU Instance 2 (1g.10gb): 80% 사용률
```

**단순 평균**:
```
(90 + 60 + 80) ÷ 3 = 76.7%
```

**가중 평균**:
```
(90×0.143 + 60×0.286 + 80×0.143) ÷ (0.143 + 0.286 + 0.143) = 74.8%
```

### 3. 실제 GPU 리소스 관점

| 계산 방법 | 고려사항 | 정확성 |
|-----------|----------|--------|
| 단순 평균 | 인스턴스 개수만 고려 | 부정확 |
| 가중 평균 | 실제 GPU 리소스 사용량 고려 | ✅ 정확 |

---

## 🔧 프로메테우스 메트릭 쿼리

### 기본 DCGM 메트릭
```promql
# 모든 MIG 인스턴스 사용률 조회
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"}
```

### 1. 단순 평균 계산
```promql
# 방법 1: 모든 인스턴스의 단순 평균
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"})

# 방법 2: 인스턴스별 개별 값 확인
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"}
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="3g.40gb"}
```

### 2. 가중 평균 계산 (권장)
```promql
# 슬라이스 크기를 고려한 가중 평균
(
  sum(
    DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143
  ) +
  sum(
    DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"} * 0.286
  ) +
  sum(
    DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="3g.40gb"} * 0.429
  ) +
  sum(
    DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="4g.40gb"} * 0.571
  ) +
  sum(
    DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="7g.80gb"} * 1.000
  )
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="3g.40gb"}) * 0.429 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="4g.40gb"}) * 0.571 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="7g.80gb"}) * 1.000
)
```

### 3. 동적 환경을 위한 간소화된 쿼리
```promql
# 현재 활성 인스턴스 개수 확인
count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"})

# 프로파일별 평균 사용률
avg by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"})
```

---

## 🎯 실제 사용 예시

### 예시 1: 혼합 MIG 구성
```
현재 구성:
- 1g.10gb × 2개 인스턴스 (90%, 80% 사용률)
- 2g.20gb × 1개 인스턴스 (60% 사용률)
- 1g.10gb × 1개 인스턴스 (70% 사용률)
```

**프로메테우스 쿼리 결과:**
```promql
# 단순 평균
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="example-pod"}) = 75%

# 가중 평균
(90×0.143 + 80×0.143 + 60×0.286 + 70×0.143) ÷ (0.143×3 + 0.286×1) = 74.8%
```

### 예시 2: 대용량 인스턴스 구성
```
현재 구성:
- 3g.40gb × 1개 인스턴스 (95% 사용률)
- 2g.20gb × 2개 인스턴스 (70%, 85% 사용률)
```

**프로메테우스 쿼리 결과:**
```promql
# 단순 평균
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="example-pod"}) = 83.3%

# 가중 평균
(95×0.429 + 70×0.286 + 85×0.286) ÷ (0.429 + 0.286×2) = 85.7%
```

---

## 📈 Grafana 대시보드 구성

### 패널 1: 현재 MIG 구성 정보
```promql
# 현재 활성 인스턴스 개수
count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"})
```

### 패널 2: 인스턴스별 개별 사용률
```promql
# 시계열 그래프
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"}
```

### 패널 3: 가중 평균 사용률 (단일 지표)
```promql
# 게이지 차트
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="3g.40gb"} * 0.429) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="4g.40gb"} * 0.571) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="7g.80gb"} * 1.000)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="3g.40gb"}) * 0.429 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="4g.40gb"}) * 0.571 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE="7g.80gb"}) * 1.000
)
```

### 패널 4: 단순 평균 vs 가중 평균 비교
```promql
# 단순 평균
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"})

# 가중 평균 (위 쿼리 동일)
```

---

## 🔍 문제 해결

### 1. 메트릭이 보이지 않는 경우
```bash
# DCGM-Exporter 상태 확인
kubectl get pods -n gpu-operator-resources | grep dcgm

# MIG 모드 확인
nvidia-smi -L
nvidia-smi mig -lgip
```

### 2. 잘못된 가중치 적용
```promql
# 현재 활성 프로파일 확인
label_values(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"}, GPU_I_PROFILE)
```

### 3. 동적 구성 변경 대응
```promql
# 시간대별 인스턴스 구성 변화 추적
changes(count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"})[1h:5m])
```

---

## 📋 핵심 포인트

### ✅ 올바른 접근법
1. **가중 평균 사용**: 실제 GPU 리소스 사용량 반영
2. **동적 구성 감지**: 런타임 MIG 변경 대응
3. **프로파일별 모니터링**: 각 인스턴스 개별 추적
4. **시간대별 분석**: 구성 변화 추적

### ❌ 피해야 할 실수
1. **단순 평균만 사용**: 실제 리소스 사용량 무시
2. **고정 가중치 하드코딩**: 동적 환경 미대응
3. **물리적 GPU 기준만 고려**: MIG 특성 무시
4. **인스턴스 변경 시 모니터링 누락**

---

## 📈 모니터링 베스트 프랙티스

### 1. 알림 설정
```yaml
# Prometheus Alert Rules
- alert: MIGHighUtilization
  expr: |
    (
      sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{job="dcgm-exporter"} * on(GPU_I_PROFILE) group_left() 
        (label_replace(label_replace(label_replace(label_replace(label_replace(
          vector(0), "GPU_I_PROFILE", "1g.10gb", "", "") + 0.143,
          "GPU_I_PROFILE", "2g.20gb", "", "") + 0.286,
          "GPU_I_PROFILE", "3g.40gb", "", "") + 0.429,
          "GPU_I_PROFILE", "4g.40gb", "", "") + 0.571,
          "GPU_I_PROFILE", "7g.80gb", "", "") + 1.000))
    ) > 90
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "MIG GPU 가중 사용률 90% 초과"
```

### 2. 리소스 계획
```promql
# 시간대별 최대 사용률 예측
max_over_time(
  # 가중 평균 쿼리
  [1h]
) by (pod)
```

### 3. 효율성 분석
```promql
# 인스턴스 효율성 점수
DCGM_FI_PROF_GR_ENGINE_ACTIVE / 
on(GPU_I_PROFILE) group_left() 
(label_replace(
  # 가중치 매핑
))
```

---

## 🎯 결론

**MIG 환경에서는 가중 평균이 더 정확한 전체 GPU 사용률을 나타냅니다.**

- **기본 쿼리**: 각 인스턴스별 개별 값들
- **가중 평균**: 슬라이스 크기를 고려한 실제 GPU 활용률
- **실제 리소스 관점**: 물리적 GPU 사용량의 정확한 반영

이 가이드를 통해 MIG 환경에서 정확한 GPU 사용률 모니터링이 가능합니다! 🚀 