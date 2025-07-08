# MIG 환경 GPU 모니터링 프로메테우스 쿼리 모음

## 📋 쿼리 카테고리
1. [기본 DCGM 메트릭](#기본-dcgm-메트릭)
2. [단순 평균 계산](#단순-평균-계산)
3. [가중 평균 계산](#가중-평균-계산)
4. [동적 환경 대응](#동적-환경-대응)
5. [알림 및 모니터링](#알림-및-모니터링)

---

## 🔧 기본 DCGM 메트릭

### 모든 MIG 인스턴스 사용률 조회
```promql
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}
```

### 특정 프로파일 사용률 조회
```promql
# 1g.10gb 인스턴스만
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"}

# 2g.20gb 인스턴스만
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"}

# 3g.40gb 인스턴스만
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"}
```

### 현재 활성 인스턴스 개수 확인
```promql
count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
```

---

## 📊 단순 평균 계산

### 전체 인스턴스 단순 평균
```promql
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
```

### 프로파일별 평균 사용률
```promql
avg by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
```

### 시간대별 평균 (5분, 1시간)
```promql
# 5분 평균
avg_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}[5m])

# 1시간 평균
avg_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}[1h])
```

---

## 🎯 가중 평균 계산 (권장)

### 완전한 가중 평균 계산
```promql
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"} * 0.429) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="4g.40gb"} * 0.571) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="7g.80gb"} * 1.000)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"}) * 0.429 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="4g.40gb"}) * 0.571 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="7g.80gb"}) * 1.000
)
```

### 간소화된 가중 평균 (주요 프로파일만)
```promql
# 1g, 2g, 3g 프로파일만 있는 경우
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"} * 0.429)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"}) * 0.429
)
```

---

## 🔄 동적 환경 대응

### 현재 활성 프로파일 라벨 추출
```promql
label_values(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}, GPU_I_PROFILE)
```

### 인스턴스 구성 변화 추적
```promql
# 지난 1시간 동안 구성 변화
changes(count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})[1h:5m])

# 현재 vs 1시간 전 구성 비교
(
  count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}) 
  != 
  count by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"} offset 1h)
)
```

### 최대/최소 사용률 추적
```promql
# 최대 사용률
max(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})

# 최소 사용률
min(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})

# 표준 편차
stddev(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
```

---

## 🚨 알림 및 모니터링

### 고사용률 알림 (90% 이상)
```promql
# 단순 평균 기준
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}) > 90

# 가중 평균 기준
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"} * 0.429)
) > 90
```

### 개별 인스턴스 알림
```promql
# 특정 인스턴스 고사용률
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"} > 95

# 모든 인스턴스 고사용률
min(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}) > 90
```

### 사용률 불균형 감지
```promql
# 인스턴스 간 사용률 차이가 큰 경우
(
  max(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}) - 
  min(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
) > 30
```

---

## 📈 성능 분석

### 시간대별 최대 사용률
```promql
# 지난 1시간 최대 사용률
max_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}[1h])

# 지난 24시간 최대 사용률
max_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}[24h])
```

### 증가율 계산
```promql
# 지난 5분 대비 증가율
rate(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"}[5m])

# 지난 1시간 대비 변화율
(
  DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"} - 
  DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"} offset 1h
) / DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"} offset 1h * 100
```

---

## 🎯 비교 분석

### 단순 평균 vs 가중 평균 비교
```promql
# 단순 평균
avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})

# 가중 평균
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"} * 0.143) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"} * 0.286) +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"} * 0.429)
) / (
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  count(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"}) * 0.429
)

# 차이값 계산
(
  # 가중 평균 쿼리
) - avg(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
```

---

## 🔧 유틸리티 쿼리

### 인스턴스 효율성 점수
```promql
# 각 인스턴스의 상대적 효율성
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"} / 
on(GPU_I_PROFILE) group_left() 
(
  avg by(GPU_I_PROFILE) (DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn"})
)
```

### 리소스 활용도 점수
```promql
# 전체 GPU 대비 활용도
(
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="1g.10gb"}) * 0.143 +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="2g.20gb"}) * 0.286 +
  sum(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="wl-49b21fe9-1ed8-4866-8b5c-d54395ea7fb9-f694b44b9-5w2vn", GPU_I_PROFILE="3g.40gb"}) * 0.429
) / 100 * 100
```

---

## 🎯 사용 팁

### 1. Pod 이름 변수 사용
```promql
# Grafana에서 변수 $pod 사용
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"}
```

### 2. 정규표현식 매칭
```promql
# 특정 패턴의 pod들
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod=~"wl-.*"}

# 1g 또는 2g 프로파일만
DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod", GPU_I_PROFILE=~"[12]g\\..*"}
```

### 3. 시간 범위 유연하게 설정
```promql
# 변수 시간 범위
avg_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod="$pod"}[$__range])
```

---

## 📋 권장 사항

### ✅ 추천 쿼리
1. **일반적인 모니터링**: 가중 평균 사용
2. **개별 인스턴스 추적**: 프로파일별 개별 쿼리
3. **알림 설정**: 가중 평균 기준 90% 이상
4. **성능 분석**: 시간대별 최대/평균 사용률

### ❌ 피해야 할 쿼리
1. **물리적 GPU 기준만**: MIG 특성 무시
2. **고정 프로파일 가정**: 동적 변경 미고려
3. **단순 평균만 사용**: 리소스 크기 무시

이 쿼리들을 사용하여 MIG 환경에서 정확한 GPU 사용률 모니터링을 할 수 있습니다! 🚀 