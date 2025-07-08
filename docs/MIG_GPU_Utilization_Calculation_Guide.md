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

## 🔧 실무 적용

### **Prometheus 쿼리 예시**

#### 1. 단순 평균 (동일 프로필 시 권장)
```prometheus
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name"}) * 100
```

#### 2. 가중 평균 (혼합 프로필 시 권장)
```prometheus
# 1g.10gb 프로필만 사용하는 경우
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143
) * 100

# 혼합 프로필 사용하는 경우
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"} * 0.286)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="your-pod-name", GPU_I_PROFILE="2g.20gb"}) * 0.286
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

### **언제 어떤 방법을 사용할까?**

#### ✅ **단순 평균 사용** (더 간단)
```bash
# 조건: 동일한 MIG 프로필만 사용
# 예시: 1g.10gb × 2개, 2g.20gb × 3개
# 장점: 계산 간단, 결과 동일
```

#### ✅ **가중 평균 사용** (더 정확)
```bash
# 조건: 서로 다른 MIG 프로필 혼합 사용
# 예시: 1g.10gb × 1개 + 2g.20gb × 1개
# 장점: 실제 하드웨어 자원 반영
```

### **구현 체크리스트**
1. **Pod MIG 프로필 확인**: `kubectl describe pod <pod-name>`
2. **DCGM 메트릭 존재 확인**: `DCGM_FI_PROF_GR_ENGINE_ACTIVE`
3. **프로필 동일성 판단**: 같으면 단순 평균, 다르면 가중 평균
4. **Prometheus 쿼리 구현**: 위 예시 참조

## 💡 FAQ

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