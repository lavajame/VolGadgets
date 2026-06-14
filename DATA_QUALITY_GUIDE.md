# Data Validation & Cleaning Strategy

## Summary

Your market data has **serious quality issues** that corrupt backtest results:

### Issues Found in 130103_SPX

| Issue | Count | Example | Impact |
|-------|-------|---------|--------|
| **Intrinsic violations** | 57 (14.8%) | mid < intrinsic for deep-ITM calls | ❌ Data corrupted |
| **Bid = 0 anomalies** | 55 | Options with bid=0 but ask > 0 | ⚠️ Illogical |
| **Extreme spreads** | 41 (12.5%) | bid/ask spread > 50% of mid | ⚠️ Illiquid |
| **Trading gaps** | 66 contracts | Missing for 3-4 days then reappear | ⚠️ Suspicious |
| **Sudden zero-mid** | 5 | Deep-ITM calls: 166 → 0 in 1 day | ❌ **Your Jan 18 issue** |

### Root Cause: Jan 18 Anomaly

```
Jan 17: Spot ≈1486, near_mid = -29.20
Jan 18: Spot ≈1490, near_mid = 1.79    (+30.99 PnL jump on short)
```

**What happened:**
- 5 deep-ITM calls (K=1315-1430) had normal mid on Jan 17 (52-166)
- Jan 18: Same calls show mid=0, bid=0, ask=0
- These had 0.298 and 0.153 weights → massive portfolio impact
- **Contributing -25.64 to PnL** (61% + 49% from just 2 contracts)

**Why:** Bid/ask data error (possibly data provider issue, not trader error)

## Cleaning Strategies

### Option 1: LENIENT (Remove only critical corruption)
```bash
python scripts/clean_mkt_data.py mkt_data.csv mkt_data_lenient.csv lenient
```
- Removes: intrinsic violations only
- Keeps: 327/400 records (81.8%)
- ✓ Minimal data loss
- ✗ Still has bid=0, extreme spreads, zero-mid events

### Option 2: MODERATE (Recommended - Balance quality & quantity)
```bash
python scripts/clean_mkt_data.py mkt_data.csv mkt_data_moderate.csv moderate
```
- Removes: intrinsic violations, bid=0, extreme spreads
- Forward-fills: zero-mid events
- Keeps: 236/400 records (59.0%)
- ✓ Removes questionable data
- ✓ Forward-fill preserves continuity
- ⚡ Recommended for production use

### Option 3: CONSERVATIVE (Remove all suspect records)
```bash
python scripts/clean_mkt_data.py mkt_data.csv mkt_data_conservative.csv conservative
```
- Removes: intrinsic violations, bid=0, extreme spreads, zero-mid
- Keeps: 236/400 records (59.0%)
- ✓ Most trustworthy data
- ✗ Aggressive filtering may miss good data

## Usage with Solver

### Run backtest with specific cleaned data:
```bash
python scripts/fit_and_evolve.py 130103_SPX \
  --constraints 'delta;gamma;theta' \
  --no-delta-hedge \
  --market-csv backtests/130103_SPX/mkt_data_moderate.csv
```

### Validate any market data:
```bash
python scripts/validate_mkt_data.py backtests/130103_SPX/mkt_data.csv
```

## Recommendations

1. **Always validate before backtesting** - Run validation script first
2. **Use MODERATE cleaning for production** - Best balance of quality/quantity
3. **Compare raw vs cleaned results** - Understand impact on your strategy
4. **Check data source** - If bid/ask errors are frequent, escalate to data provider
5. **Document your choice** - Record which cleaning strategy was used

## Key Insight

The Jan 18 anomaly is **not a solver bug** - it's a **data quality issue**. Your static weights are doing exactly what they should. The problem is that market data fed into the solver contained corrupted prices (mid=0 when it should be 50+).

With cleaned data, the solver produces more stable results. However, some portfolio drift is expected as options move deep-ITM/OTM and liquidity disappears.
