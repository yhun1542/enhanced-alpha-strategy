# Enhanced Alpha Strategy v5.0 - 완전 재현 패키지

**버전**: v5.0 (Turnover Penalty Optimized)
**성과**: Sharpe 2.27, Max DD -10.92%, Annual Return 34.43%
**작성일**: 2024-12-04

---

## 📋 패키지 내용

```
enhanced_alpha_backup/
├── README.md                          # 본 파일 (재현 가이드)
├── enhanced_alpha_strategy_v5.py      # 최적 전략 소스코드
├── config.json                        # 전략 설정 파일
├── requirements.txt                   # Python 패키지 의존성
├── run.sh                             # 실행 스크립트
└── test_results_summary.md            # 8가지 테스트 결과 요약
```

---

## 🚀 빠른 시작

### 1. 환경 요구사항

- **Python**: 3.11+ (권장) 또는 3.8+
- **OS**: Linux, macOS, Windows (WSL)
- **메모리**: 최소 2GB RAM
- **디스크**: 최소 500MB

### 2. API 키 준비

Polygon.io API 키 필요:
- 무료 계정: https://polygon.io/
- API 키 발급 후 환경변수 설정

### 3. 설치 및 실행

```bash
# 1. 패키지 설치
pip3 install -r requirements.txt

# 2. API 키 설정
export POLYGON_API_KEY='your_polygon_api_key_here'

# 3. 실행
./run.sh
```

또는 직접 실행:

```bash
python3.11 enhanced_alpha_strategy_v5.py
```

### 4. 결과 확인

실행 완료 후 생성되는 파일:
- `enhanced_alpha_results.json` - 백테스트 결과 (JSON)
- `data_cache/*.pkl` - 캐시된 시장 데이터

---

## 📊 예상 결과

### 성과 지표
```
Sharpe Ratio:      2.27
Sortino Ratio:     2.83
Calmar Ratio:      3.15
Total Return:      1815.82%
Annual Return:     34.43%
Annual Volatility: 14.30%
Max Drawdown:      -10.92%
CVaR (95%):        1.15%
Win Rate:          59.8%
Txn Costs (Ann.):  9.07%
Taxes (Ann.):      12.58%
Total Costs:       21.65%
```

### 레짐 분포
```
BULL: 352 days (75.1%)
NEUTRAL: 44 days (9.4%)
BEAR: 73 days (15.6%)
DD Protection: 67 events
```

### 최종 포트폴리오
```
GOOGL: 17.4%
AAPL: 15.4%
NVDA: 12.2%
AMZN: 5.3%
TSLA: 4.6%
META: 1.4%
MSFT: 0.7%
```

---

## ⚙️ 설정 커스터마이징

### config.json 주요 파라미터

#### 유니버스
```json
"universe": {
  "tech_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
  "market_etf": "SPY"
}
```

#### 백테스트 기간
```json
"backtest": {
  "start_date": "2015-01-01",
  "end_date": "2024-12-31",
  "rebalance_freq": "W",  // W=주간, M=월간
  "transaction_cost_bps": 23,
  "risk_free_rate": 0.02
}
```

#### 전략 파라미터
```json
"strategy": {
  "target_vol": 0.14,
  "turnover_penalty": 0.05,  // v5.0 핵심 개선
  "enable_turnover_control": true
}
```

#### 알파 가중치
```json
"alpha_weights": {
  "drawdown_recovery": 0.50,  // IC = 0.32 (최고)
  "price_acceleration": 0.30, // IC = 0.18
  "vol_adj_momentum": 0.20    // IC = 0.08
}
```

#### 레짐별 Exposure
```json
"regime_exposure": {
  "BULL": 1.0,      // 100% 투자
  "NEUTRAL": 0.65,  // 65% 투자
  "BEAR": 0.25      // 25% 투자 (방어적)
}
```

---

## 🔬 핵심 개선 사항 (v5.0)

### Turnover Penalty
```python
# Config
turnover_penalty: float = 0.05
enable_turnover_control: bool = True

# 로직
lambda_turn = turnover_penalty / (1 + turnover_penalty)
final_exposure = (1 - lambda_turn) * final_exposure + lambda_turn * prev_exposure
```

**효과**:
- Sharpe: 2.28 → 2.46 (+7.9%, 세금 제외)
- Sharpe: 2.15 → 2.27 (+5.6%, 세금 포함)
- Max DD: -10.89% → -10.92% (거의 동일)

---

## 📈 백테스트 상세

### 데이터 소스
- **가격 데이터**: Polygon.io API
- **기간**: 2015-01-01 ~ 2024-12-31 (10년)
- **빈도**: 일간 (Daily)
- **종목**: 7개 Tech 주식

### 리밸런싱
- **빈도**: 주간 (매주 금요일)
- **횟수**: 469회 (10년간)
- **회전율**: ~43x annually

### 비용 모델
- **거래 비용**: 23 bps (0.23%)
- **단기 양도세**: 35% (보유 < 1년)
- **장기 양도세**: 20% (보유 ≥ 1년)
- **슬리피지**: 포함됨

### 리스크 관리
1. **GMM 레짐 감지**: Bull/Neutral/Bear
2. **Drawdown Protection**: 3단계 (-5%, -10%, -15%)
3. **변동성 타겟팅**: 14% annually
4. **Turnover Control**: v5.0 신규

---

## 🧪 검증 방법

### 1. 결과 재현성 확인
```bash
# 두 번 실행하여 결과 동일한지 확인
./run.sh > result1.txt
./run.sh > result2.txt
diff result1.txt result2.txt
```

### 2. 성과 지표 검증
```python
import json

with open('enhanced_alpha_results.json') as f:
    results = json.load(f)

assert 2.25 < results['sharpe'] < 2.30, "Sharpe 범위 확인"
assert -0.11 < results['max_dd'] < -0.10, "Max DD 범위 확인"
assert 0.34 < results['annualized_return'] < 0.35, "Return 범위 확인"
```

### 3. 데이터 캐시 확인
```bash
# 첫 실행: 데이터 다운로드 (느림)
time ./run.sh

# 두 번째 실행: 캐시 사용 (빠름)
time ./run.sh
```

---

## 🐛 트러블슈팅

### API 키 오류
```
Error: Missing Polygon API key
```
**해결**: `export POLYGON_API_KEY='your_key'` 실행

### 패키지 설치 오류
```
ModuleNotFoundError: No module named 'pandas'
```
**해결**: `pip3 install -r requirements.txt` 실행

### 데이터 다운로드 실패
```
Failed to fetch any data
```
**해결**:
1. 인터넷 연결 확인
2. API 키 유효성 확인
3. Polygon.io 서비스 상태 확인

### 메모리 부족
```
MemoryError
```
**해결**:
1. 캐시 삭제: `rm -rf data_cache/`
2. 기간 단축: `config.json`에서 `start_date` 조정

---

## 📚 추가 자료

### 전략 문서
- `test_results_summary.md` - 8가지 개선 테스트 결과

### 코드 구조
```python
# 주요 클래스
DataFetcher         # 데이터 다운로드 및 캐싱
HighICAlphaEngine   # 알파 시그널 생성
MacroTiming         # 레짐 감지
StockTiming         # 리스크 관리
EnhancedAlphaStrategy  # 전략 통합
Backtester          # 백테스트 엔진
```

### 알파 시그널
1. **Drawdown Recovery** (IC = 0.32)
   - 60일 drawdown에서 회복 속도
   - 가중치: 50%

2. **Price Acceleration** (IC = 0.18)
   - 10일 vs 60일 모멘텀 차이
   - 가중치: 30%

3. **Vol-Adj Momentum** (IC = 0.08)
   - 변동성 조정 20일 수익률
   - 가중치: 20%

---

## 🔐 보안 주의사항

1. **API 키 보호**
   - `config.json`에 API 키 직접 입력 금지
   - 환경변수 사용 권장
   - Git에 커밋 금지

2. **결과 파일**
   - `enhanced_alpha_results.json`은 민감정보 없음
   - 공유 가능

3. **캐시 파일**
   - `data_cache/*.pkl`은 공개 시장 데이터
   - 공유 가능

---

## 📞 지원

### 문의
- GitHub Issues: (리포지토리 URL)
- Email: (담당자 이메일)

### 라이선스
- 본 코드는 연구 및 교육 목적으로 제공됨
- 실제 투자 시 자기 책임 원칙

---

## 🎯 다음 단계

### 추가 개선 방향
1. **Long-Short 전략** (130/30)
   - 예상: Sharpe 2.27 → 2.80+
   - 소요: 1-2주

2. **옵션 헤징** (VIX calls, Put spreads)
   - 예상: Max DD -10.92% → -6%
   - 소요: 1-2주

3. **머신러닝 알파**
   - 예상: Sharpe 3.0+
   - 소요: 2-3주

---

**버전**: v5.0 (Turnover Penalty Optimized)
**최종 업데이트**: 2024-12-04
**작성자**: Manus AI Agent
