# MIG 환경에서 GPU 사용률 메트릭 계산 가이드

## 📋 목차
1. [MIG(Multi-Instance GPU) 개요](#mig-multi-instance-gpu-개요)
2. [GPU 사용률 계산 방법의 차이점](#gpu-사용률-계산-방법의-차이점)
3. [프로메테우스 메트릭 쿼리](#프로메테우스-메트릭-쿼리)
4. [실제 사용 예시](#실제-사용-예시)
5. [Grafana 대시보드 구성](#grafana-대시보드-구성)
6. [문제 해결](#문제-해결)

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