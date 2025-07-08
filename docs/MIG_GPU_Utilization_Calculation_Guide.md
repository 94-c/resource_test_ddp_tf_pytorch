# MIG GPU 사용률 계산 가이드

## 📋 핵심 요약

### 🎯 **두 가지 계산 방법**
1. **단순 평균**: 인스턴스 개수 기준
2. **가중 평균**: 실제 하드웨어 자원 비율 기준 (권장)

### 🧮 **A100 80GB MIG 가중치 (NVIDIA 공식)**
| MIG 프로필 | 가중치 | 설명 |
|------------|--------|------|
| 1g.10gb    | 0.143  | 14.3% (1/7 컴퓨팅 슬라이스) |
| 2g.20gb    | 0.286  | 28.6% (2/7 컴퓨팅 슬라이스) |
| 3g.20gb    | 0.429  | 42.9% (3/7 컴퓨팅 슬라이스) |
| 4g.20gb    | 0.571  | 57.1% (4/7 컴퓨팅 슬라이스) |
| 7g.40gb    | 1.000  | 100% (7/7 컴퓨팅 슬라이스) |

## 🚨 **MIG vs 일반 GPU 구분 방법**

### **⚡ 빠른 구분법**
```bash
# DCGM 메트릭에서 확인
GPU_I_PROFILE 라벨 존재 여부 확인:
- 있음 → MIG 환경 (가중 평균 사용)
- 없음 → 일반 GPU 환경 (단순 쿼리 사용)
```

### **📊 환경별 특징 비교**

| 구분 | MIG 환경 | 일반 GPU 환경 |
|------|----------|---------------|
| **GPU_I_PROFILE** | ✅ 있음 (`1g.10gb`, `2g.20gb` 등) | ❌ 없음 |
| **GPU_I_ID** | ✅ 있음 (MIG 인스턴스 ID) | ❌ 없음 |
| **device** | `nvidia0-mig-xxx` 형태 | `nvidia0` 형태 |
| **지원 GPU** | A100, A30, H100 | V100, P100, T4, RTX |
| **사용률 계산** | 평균 또는 가중 평균 | 단순 메트릭 × 100 |

### **🔍 실제 메트릭 예시**

#### **MIG 환경 (A100)**
```bash
{
  DCGM_FI_DEV_NAME="NVIDIA A100-SXM4-80GB",
  GPU_I_PROFILE="1g.10gb",           # ← 이게 있으면 MIG
  GPU_I_ID="8",                      # ← MIG 인스턴스 ID
  device="nvidia0-mig-1g.10gb",      # ← MIG 디바이스
  gpu="0",
  pod="mig-pod-example"
}
```

#### **일반 GPU 환경 (V100)**
```bash
{
  DCGM_FI_DEV_NAME="Tesla V100-PCIE-32GB",
  device="nvidia0",                   # ← 일반 GPU 디바이스
  gpu="0",                           # ← 물리적 GPU 0번
  pod="regular-gpu-pod"
  # GPU_I_PROFILE 없음               # ← 이게 없으면 일반 GPU
}
```

### **🎯 GPU 아키텍처별 MIG 지원**

#### **✅ MIG 지원 (Ampere/Hopper)**
```bash
- A100 80GB
- A100 40GB  
- A30
- H100
- H200
```

#### **❌ MIG 미지원 (이전 아키텍처)**
```bash
- V100 (Volta)
- P100 (Pascal)
- T4 (Turing)
- RTX 시리즈
- GTX 시리즈
```

## 🔧 **MIG 슬라이싱 개념과 avg 사용 이유**

### 🎯 **핵심 질문: 왜 avg를 사용하는가?**

#### **MIG 슬라이싱 기본 개념**
```
A100 80GB GPU = 7개의 컴퓨팅 슬라이스로 분할 가능
┌─────────────────────────────────────────────────────┐
│ 전체 A100 GPU (108개 SM, 80GB 메모리)                │
├─────┬─────┬─────┬─────┬─────┬─────┬─────────────────┤
│ 1/7 │ 1/7 │ 1/7 │ 1/7 │ 1/7 │ 1/7 │    1/7      │
│슬라이스│슬라이스│슬라이스│슬라이스│슬라이스│슬라이스│  슬라이스    │
└─────┴─────┴─────┴─────┴─────┴─────┴─────────────────┘
```

#### **Pod가 여러 MIG 인스턴스를 할당받는 경우**
```
Pod 요청: nvidia.com/gpu: 2
실제 할당: 1g.10gb × 2개 MIG 인스턴스

┌─────────────────────────────────────────────────────┐
│ 전체 A100 GPU                                        │
├─────┬─────┬─────────────────────────────────────────┤
│ 1/7 │ 1/7 │         나머지 5/7 슬라이스              │
│GPU_I_ID│GPU_I_ID│     (다른 워크로드 또는 미사용)        │
│  ="8" │ ="12" │                                    │
│  0%   │98.39% │                                    │
└─────┴─────┴─────────────────────────────────────────┘
```

#### **❌ 왜 사용률을 단순히 더할 수 없는가?**
```bash
# 잘못된 계산
총 사용률 = 0% + 98.39% = 98.39%  # ❌ 틀림!

# 문제점: 각 인스턴스 사용률은 해당 슬라이스 내에서의 사용률
# - GPU_I_ID="8": 1/7 슬라이스 내에서 0% 사용
# - GPU_I_ID="12": 1/7 슬라이스 내에서 98.39% 사용
# 전체 GPU 관점에서는 단순히 더할 수 없음
```

#### **✅ 왜 평균을 사용해야 하는가?**
```bash
# 올바른 계산 (Pod 전체 사용률 관점)
Pod 사용률 = (0% + 98.39%) / 2 = 49.2%

# 의미: Pod가 할당받은 자원(2개 슬라이스) 중 평균적으로 49.2% 사용
```

#### **🔍 실제 하드웨어 자원 관점**
```bash
# 전체 GPU 사용률 = 각 슬라이스 사용률 × 슬라이스 비율
전체 GPU 사용률 = (0% × 1/7) + (98.39% × 1/7) + (0% × 5/7)
                = 0% + 14.06% + 0%
                = 14.06%

# Pod 할당 자원 대비 사용률 = Pod 사용률
Pod 사용률 = (0% + 98.39%) / 2 = 49.2%
```

### 🎯 **단순 평균 vs 가중 평균 이유**

#### **단순 평균 (동일 프로필 시)**
```prometheus
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="pod-name"}) * 100

# 이유: 동일한 크기의 슬라이스이므로 가중치가 같음
# 1g.10gb × 2개 = 0.143 × 2 = 0.286 (총 가중치)
# 각 인스턴스 가중치: 0.143 / 0.286 = 0.5 (50% 씩)
```

#### **가중 평균 (혼합 프로필 시)**
```prometheus
# 1g.10gb (50%) + 2g.20gb (80%) 혼합 사용 예시
(50% × 0.143 + 80% × 0.286) / (0.143 + 0.286) = 69.7%

# 이유: 다른 크기의 슬라이스이므로 가중치가 다름
# 1g.10gb 가중치: 0.143 / 0.429 = 33.3%
# 2g.20gb 가중치: 0.286 / 0.429 = 66.7%
```

### 📊 **시각적 이해**

#### **사례 1: 동일 프로필 (1g.10gb × 2개)**
```
┌─────────────────────────────────────────────────────┐
│ 전체 A100 GPU (7개 슬라이스)                          │
├─────┬─────┬─────┬─────┬─────┬─────┬─────────────────┤
│ 1/7 │ 1/7 │ 1/7 │ 1/7 │ 1/7 │ 1/7 │    1/7      │
│  0% │98.39%│  -  │  -  │  -  │  -  │     -       │
│Pod A│Pod A │     │     │     │     │             │
└─────┴─────┴─────┴─────┴─────┴─────┴─────────────────┘

Pod A 사용률 = (0% + 98.39%) / 2 = 49.2%
전체 GPU 사용률 = (0% × 1/7) + (98.39% × 1/7) = 14.06%
```

#### **사례 2: 혼합 프로필 (1g.10gb + 2g.20gb)**
```
┌─────────────────────────────────────────────────────┐
│ 전체 A100 GPU (7개 슬라이스)                          │
├─────┬─────┬─────┬─────┬─────┬─────┬─────────────────┤
│ 1/7 │ 2/7 (합쳐진 슬라이스) │ 1/7 │ 1/7 │    1/7      │
│ 50% │      80%           │  -  │  -  │     -       │
│Pod B│      Pod B         │     │     │             │
└─────┴─────┴─────┴─────┴─────┴─────┴─────────────────┘

Pod B 사용률 = (50% × 0.143 + 80% × 0.286) / (0.143 + 0.286) = 69.7%
전체 GPU 사용률 = (50% × 1/7) + (80% × 2/7) = 30.0%
```

### 💡 **핵심 이해**

#### **각 인스턴스 사용률의 의미**
- `DCGM_FI_PROF_GR_ENGINE_ACTIVE = 0.5` = 해당 슬라이스 내에서 50% 사용
- 전체 GPU 관점에서는 `0.5 × 슬라이스_가중치` 만큼 사용

#### **평균 사용하는 이유**
1. **Pod 할당 자원 관점**: Pod가 받은 자원 중 얼마나 사용하는가?
2. **슬라이스 정규화**: 각 슬라이스 크기가 다르므로 정규화 필요
3. **직관적 이해**: 100%를 넘지 않는 합리적인 수치

## 🔧 실무 적용

### **🎯 환경별 쿼리 선택 가이드**

#### **1단계: 환경 구분**
```bash
# DCGM 메트릭 확인
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"}

# GPU_I_PROFILE 라벨 존재 여부 확인
- 있음 → MIG 환경 (2-4번 쿼리 사용)
- 없음 → 일반 GPU 환경 (1번 쿼리 사용)
```

### **Prometheus 쿼리 예시**

#### **1. 일반 GPU 환경 (V100, P100, T4 등)**
```prometheus
# Tesla V100, P100, T4 등 MIG 미지원 GPU용
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"} * 100

# 실제 사용 예시
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-f0ed4e9e-1989-4ff0-8277-1c74cd5e55eb-548f8b5749-tmcs2"} * 100
# 결과: 98.29 (Tesla V100 사용률 98.29%)
```

#### **2. MIG 환경 - 단순 평균 (동일 프로필 시)**
```prometheus
# 1g.10gb × 2개 등 동일 프로필 사용 시
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"}) * 100

# 실제 사용 예시
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-dda582f7-4d74-4d67-9b52-a91aeeac7ac0-858c47bbd5-nlmfg"}) * 100
```

#### **3. MIG 환경 - 가중 평균 (단일 프로필)**
```prometheus
# 1g.10gb만 사용하는 경우
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143
) * 100

# 실제 사용 예시
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-dda582f7-4d74-4d67-9b52-a91aeeac7ac0-858c47bbd5-nlmfg", GPU_I_PROFILE="1g.10gb"} * 0.143)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-dda582f7-4d74-4d67-9b52-a91aeeac7ac0-858c47bbd5-nlmfg", GPU_I_PROFILE="1g.10gb"}) * 0.143
) * 100
```

#### **4. MIG 환경 - 가중 평균 (혼합 프로필)**
```prometheus
# 1g.10gb + 2g.20gb 혼합 사용 시
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"} * 0.286)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"}) * 0.286
) * 100

# 범용 쿼리 (모든 MIG 프로필 자동 처리)
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="3g.20gb"} * 0.429) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="4g.20gb"} * 0.571) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="7g.40gb"} * 1.000)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="3g.20gb"}) * 0.429 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="4g.20gb"}) * 0.571 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="7g.40gb"}) * 1.000
) * 100
```

### **계산 예시**

#### 사례 1: 동일 프로필 (1g.10gb × 2개)
```
메트릭 데이터:
- GPU_I_ID="8": 0% 사용률
- GPU_I_ID="12": 98.39% 사용률

단순 평균: (0% + 98.39%) / 2 = 49.2%
가중 평균: (0% × 0.143 + 98.39% × 0.143) / (0.143 + 0.143) = 49.2%

결과: 동일 프로필 시 두 방법 결과 같음
```

#### 사례 2: 혼합 프로필 (1g.10gb + 2g.20gb)
```
메트릭 데이터:
- 1g.10gb: 50% 사용률
- 2g.20gb: 80% 사용률

단순 평균: (50% + 80%) / 2 = 65%
가중 평균: (50% × 0.143 + 80% × 0.286) / (0.143 + 0.286) = 69.7%

결과: 가중 평균이 더 정확 (큰 인스턴스 사용률 높음)
```

## ⚠️ 중요 제약사항

### **검증 불가능한 부분**
- ❌ **실시간 사용률**: 하드웨어 레벨 정보 접근 불가
- ❌ **물리적 할당**: 실제 SM/메모리 매핑 확인 불가
- ❌ **하드웨어 스케줄러**: 블랙박스 내부 동작

### **DCGM vs nvidia-smi 차이**
| 구분 | DCGM | nvidia-smi |
|------|------|------------|
| **데이터 유형** | 시계열 (과거 데이터 보존) | 현재 스냅샷 |
| **MIG 사용률** | 인스턴스별 상세 제공 | 제한적 정보 |
| **워크로드 종료 시** | 과거 데이터 유지 | 프로세스 없음 표시 |

## 🎯 실무 권장사항

### **🔍 환경별 쿼리 선택 가이드**

#### **1️⃣ 일반 GPU 환경** (V100, P100, T4 등)
```bash
# 조건: GPU_I_PROFILE 라벨 없음
# 사용 쿼리: DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="xxx"} * 100
# 장점: 단순명료, 즉시 사용 가능
```

#### **2️⃣ MIG 환경 - 단순 평균** (동일 프로필)
```bash
# 조건: 동일한 MIG 프로필만 사용
# 예시: 1g.10gb × 2개, 2g.20gb × 3개
# 사용 쿼리: avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="xxx"}) * 100
# 장점: 계산 간단, 결과 동일
```

#### **3️⃣ MIG 환경 - 가중 평균** (혼합 프로필)
```bash
# 조건: 서로 다른 MIG 프로필 혼합 사용
# 예시: 1g.10gb × 1개 + 2g.20gb × 1개
# 사용 쿼리: 가중 평균 공식
# 장점: 실제 하드웨어 자원 반영
```

### **🔧 구현 체크리스트**
1. **GPU 타입 확인**: `kubectl describe node <node-name>` → V100/A100 구분
2. **Pod 메트릭 확인**: `DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="xxx"}` → 라벨 존재 여부
3. **환경 판단**: 
   - `GPU_I_PROFILE` 없음 → 일반 GPU (1번 쿼리)
   - `GPU_I_PROFILE` 있음 → MIG (2-3번 쿼리)
4. **MIG 프로필 동일성 확인**: 같으면 단순 평균, 다르면 가중 평균
5. **Prometheus 쿼리 구현**: 위 예시 참조

### **⚠️ 주의사항**
- **잘못된 쿼리 사용**: V100에서 MIG 쿼리 사용 시 값 없음
- **Pod 이름 오류**: 정확한 Pod 이름 사용 필요
- **쿼리 구조 오류**: `count(metric * weight)` 대신 `count(metric) * weight` 사용

## 💡 FAQ

### Q: 내 Pod가 MIG 환경인지 일반 GPU 환경인지 어떻게 확인하나?
**A**: DCGM 메트릭에서 `GPU_I_PROFILE` 라벨 확인. 있으면 MIG, 없으면 일반 GPU

### Q: V100 GPU에서 MIG 쿼리를 사용하면 안 되나?
**A**: V100은 MIG 미지원. 단순 쿼리(`DCGM_FI_PROF_GR_ENGINE_ACTIVE * 100`) 사용 필요

### Q: MIG 환경에서 가중 평균 쿼리가 작동하지 않는 이유는?
**A**: 
- Pod 이름 오류: 다른 Pod 이름 사용 시 메트릭 없음
- 쿼리 구조 오류: `count(metric * weight)` 대신 `count(metric) * weight` 사용
- 프로필 불일치: 실제 사용 중인 프로필과 쿼리 프로필 다름

### Q: 가중치는 어디서 나온 값인가?
**A**: NVIDIA 공식 하드웨어 스펙 (A100 80GB 컴퓨팅 슬라이스 비율)

### Q: 실시간 사용률 확인이 안 되는 이유는?
**A**: MIG 환경에서 하드웨어 레벨 정보 접근 제한, 블랙박스 스케줄러

### Q: 두 방법 중 어느 것이 더 정확한가?
**A**: 가중 평균이 더 정확하지만, 동일 프로필 시 결과 동일

### Q: DCGM 메트릭이 0%인데 실제로는 사용 중일 수 있나?
**A**: 과거 시점 데이터임. 현재 nvidia-smi로 실제 프로세스 확인 필요

---

## 🔗 참고 자료
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [DCGM Metrics Reference](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/group__dcgmFieldIdentifiers.html)

**이 문서는 MIG GPU 환경에서의 사용률 계산 방법론을 실무 중심으로 정리한 것입니다.** 