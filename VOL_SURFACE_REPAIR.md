# Market Data Preprocessing: Vol Surface Repair

## Problem Identified

Your market data contained **5 deep-ITM calls that collapsed to zero bid/ask overnight:**

```
2013-01-17 → 2013-01-18:
  K=1315: mid 166.35 → 0.00
  K=1345: mid 136.45 → 0.00
  K=1370: mid 111.50 → 0.00
  K=1400: mid  81.70 → 0.00
  K=1430: mid  52.15 → 0.00
```

This caused a **+30.99 PnL anomaly** on the short leg that looked like a solver bug.

## Solution: Vol Surface Repair

Rather than filtering out bad data, we **reconstruct missing prices using the vol surface**:

1. Identify when ITM side has zero bid/ask but OTM side has valid quotes
2. Extract the implied volatility from the OTM side
3. Use Black-Scholes to regenerate the ITM prices with that IV
4. Replace the zero prices with reconstructed values

**Result:** Anomaly reduced from **+30.99 to -1.88** (reasonable daily change)

## Automatic Preprocessing

`fit_and_evolve.py` now **automatically preprocesses data** before solving:

```bash
# Automatic preprocessing (default):
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta'

# Skip preprocessing:
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-preprocess

# Use specific market data:
python scripts/fit_and_evolve.py 130103_SPX --market-csv path/to/mkt_data.csv
```

**Workflow:**
```
Step -1: Preprocess (vol surface repair) → mkt_data_preprocessed.csv
Step 0:  Compute 2nd-order Greeks
Step 1:  Solve for weights
Step 2:  Plot evolution & attribution
```

## Manual Preprocessing

To preprocess a single dataset:

```bash
python scripts/preprocess_vol_repair.py backtests/130103_SPX/mkt_data.csv \
                                         backtests/130103_SPX/clean_mkt_data.csv
```

To preprocess all datasets:

```bash
python scripts/batch_preprocess_all.py
```

This creates `clean_mkt_data.csv` in each backtest folder.

## How It Works

For each quote_date and strike:

1. **Find call/put pair** at same date and strike
2. **Identify ITM vs OTM**: 
   - If S > K → call is ITM, put is OTM
   - If S < K → put is ITM, call is OTM
3. **Repair ITM side**:
   - If ITM has zero bid/ask but OTM doesn't
   - Use OTM's implied vol
   - Regenerate ITM prices: 
     ```
     call_price = BS(S, K, T, sigma=OTM_IV)
     call_mid = call_price
     call_bid = call_mid * 0.99
     call_ask = call_mid * 1.01
     ```
4. **Preserve vol surface consistency**: Both sides now use same IV

## Quality Checks

The preprocessing script reports all repairs:

```
[Repaired] CALL K=1315.0 on 2013-01-18: IV=0.1767 -> mid=170.98
[Repaired] CALL K=1345.0 on 2013-01-18: IV=0.1767 -> mid=140.98
[Repaired] CALL K=1370.0 on 2013-01-18: IV=0.1767 -> mid=115.98
[Repaired] CALL K=1400.0 on 2013-01-18: IV=0.1767 -> mid=86.07
[Repaired] CALL K=1430.0 on 2013-01-18: IV=0.1608 -> mid=56.54

5 bad quotes repaired using vol surface
```

## Results: Before vs After

### Raw Data (Zero ITM Prices)
```
Jan 17: near_mid = -29.20
Jan 18: near_mid =   1.79
Jump:  +30.99  ❌ ANOMALY
```

### Repaired Data (Vol Surface Consistent)
```
Jan 17: near_mid = -28.45
Jan 18: near_mid = -30.33
Jump:   -1.88  ✓ REASONABLE
```

## Script Reference

| Script | Purpose |
|--------|---------|
| `preprocess_vol_repair.py` | Single dataset preprocessing |
| `batch_preprocess_all.py` | Batch preprocessing of all backtests |
| `fit_and_evolve.py` | Main orchestration (now with automatic preprocessing) |

## Recommendations

1. **Always preprocess before backtesting** - Automatic in `fit_and_evolve.py`
2. **Understand your data** - Review reported repairs to spot issues
3. **Validate data quality** - Use `validate_mkt_data.py` to check raw data
4. **Preserve vol surface** - This approach maintains consistent IV across strikes
