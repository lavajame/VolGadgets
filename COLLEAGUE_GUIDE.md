# Quick Start Guide for Colleague Review

## Setup

### 1. Clone and Install
```bash
git clone <repo-url> VolatilityGadgets
cd VolatilityGadgets
python -m venv venv
.\venv\Scripts\activate
pip install pandas numpy scipy matplotlib
```

### 2. Run the Example
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-delta-hedge
```

This will:
- Load 16 days of SPX weekly option data (Jan 2013)
- Repair corrupted ITM prices using OTM vol surface
- Solve for portfolio weights that isolate vega
- Generate plots showing portfolio evolution
- Output daily PnL breakdown (8 factors)

Outputs: `backtests/130103_SPX/plots/`

## Understanding the Vega Isolation Problem

### The Goal
You have a target option (far OTM) with some vega exposure. You want to:
1. **Replicate its vega exactly** using liquid options
2. **Hedge away delta and gamma** to near-zero
3. **Measure how pure the vega exposure is**

### How It Works
1. Build a **constraint matrix** with Greeks you want to match/hedge
2. Use **scale-invariant ridge regression** to solve for weights
3. Track **portfolio evolution** over 16 days
4. Measure **vega isolation quality** via Greek noise metric

## Key Scripts

### 1. Main Entry Point: `fit_and_evolve.py`
```bash
# Basic usage
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta'

# With custom parameters
python scripts/fit_and_evolve.py 130103_SPX \
  --constraints 'delta;gamma;vanna;volga' \
  --alpha 1e-3 \
  --no-delta-hedge
```

**What it does:**
- Step -1: Preprocess (repair ITM prices)
- Step 0: Compute 2nd-order Greeks
- Step 1: Solve for weights
- Step 2: Plot evolution + PnL attribution

### 2. Measure Vega Isolation: `vega_isolation_diagnostic.py`
```bash
# Measure vega isolation quality
python scripts/vega_isolation_diagnostic.py 130103_SPX

# Compare multiple constraint combinations
python scripts/compare_constraint_configs.py 130103_SPX
```

**Output metrics:**
- `Greek noise = |delta| + |gamma| + |theta|` (want < 0.01)
- `Vega exposure` (in $/1% vol)
- `Quality score`: ✓✓✓ if <0.01, ✓✓ if <0.1, ✓ if <1.0, ✗ if >1.0

### 3. Validate Data: `validate_mkt_data.py`
```bash
python scripts/validate_mkt_data.py backtests/130103_SPX/mkt_data.csv
```

**Checks for:**
- Missing data (NaN values)
- Quote missing flags
- Zero/negative prices
- Extreme bid-ask spreads
- Intrinsic value violations (should never happen!)
- Trading gaps > 1 day
- Sudden zero-mid collapses

## Understanding the Example Results

### What You'll See

#### 1. **Portfolio Weights** (near_weights_by_strike.png)
Shows how portfolio is distributed across 10 strikes:
- Some positive weights (long calls)
- Some negative weights (short calls)
- Typically smooth distribution = good portfolio

#### 2. **Greeks Evolution** (greeks_evolution.png)
Shows how Greeks change over 16 days:
- Delta should stay near 0 (hedged)
- Gamma should stay near 0 (hedged)
- Vega = main exposure (what we want!)
- Theta = secondary effect

#### 3. **PnL Attribution** (pnl_attribution_cumulative_near_*.png)
8-factor breakdown of cumulative PnL:
1. **Delta PnL**: Δ₀ × dS
2. **Gamma PnL**: 0.5 × Γ₀ × dS²
3. **Vega PnL**: ν₀ × dIV (main driver!)
4. **Vanna PnL**: Ψ₀ × dS × dIV
5. **Volga PnL**: Λ₀ × (dIV)²/2
6. **dTheta/dS**: ∂θ/∂S × dS
7. **Theta PnL**: θ₀ × dt
8. **Residual**: Unexplained (want this small)

**For vega isolation, you want:**
- Vega PnL to dominate
- Delta/Gamma/Theta PnL to be minimal
- Residual to be <5% of total

### Key Metrics

```
Constraint: 'delta;gamma;theta'

At day 0 (initial):
  Delta = 0.00 (hedged!)
  Gamma = 0.00 (hedged!)
  Theta = -1.23 (small, acceptable)
  Vega  = 88.82 (main exposure ✓)

Quality score: ✓✓ (greek_noise=0.01, excellent)
```

## Constraint Combinations Explained

| Constraints | When to Use | Result |
|-----------|-----------|--------|
| `delta` | Hedge directional risk | Basic protection |
| `delta;gamma` | Hedge curve risk + gamma convexity | Better |
| `delta;gamma;theta` | Also neutralize decay | Best for vega isolation |
| `delta;gamma;vanna` | Add cross-gamma sensitivity | For advanced models |
| `delta;gamma;vanna;volga` | Use all 2nd-order Greeks | Maximum complexity |

**For vega isolation: `delta;gamma;theta` is optimal** (provides ✓✓ quality)

## Troubleshooting

### Problem: Small weights (<0.01) or zero weights
- **Cause**: Too many strikes vs constraints (ill-posed)
- **Fix**: Reduce number of strikes in near_grid, or add more constraints

### Problem: Negative gamma in portfolio
- **Cause**: Solver overfitting to match constraints
- **Fix**: Increase `--alpha` parameter (e.g., --alpha 1e-2)

### Problem: Large PnL jumps on single days
- **Cause**: Likely data corruption (zero bids, intrinsic violations)
- **Fix**: Run `validate_mkt_data.py`, then preprocess

### Problem: Residual PnL too large (>10% of total)
- **Cause**: Greeks not explained by linear model (convexity, etc.)
- **Fix**: Try adding vanna/volga constraints, or accept residual

## Data Format: clean_mkt_data.csv

Pre-cleaned example dataset with columns:
```
quote_date,selection_group,underlying_close,contract,expiration,
option_type,strike,style,bid,ask,mid,volume,open_interest,
delta,gamma,theta,vega,implied_volatility
```

- **selection_group**: "underlying", "near_grid", or "far_target"
- **style**: European ("E") or American ("A")
- **mid**: Pre-processed using vol surface repair
- **Greeks**: Pre-computed from bid/ask midpoint

## Questions?

1. **"How do I add a new constraint?"**
   - Edit the constraint string in `fit_and_evolve.py` call
   - Or modify `scale_invariant_ridge_solver.py` to add new Greeks

2. **"Why is my portfolio concentrated at one strike?"**
   - Check if you have enough options in near_grid
   - Try reducing alpha (more regularization)

3. **"Can I use this with different options (stocks, bonds)?"**
   - Yes! Preprocess your data to clean_mkt_data.csv format
   - Run batch_preprocess_all.py first
   - Then fit_and_evolve.py

## Next Steps

1. **Read README.md** for comprehensive documentation
2. **Try different constraints**: delta;gamma vs delta;gamma;theta vs delta;gamma;vanna
3. **Compare quality scores** using vega_isolation_diagnostic.py
4. **Review PnL attribution plots** to understand what drives returns
5. **Run on your own data** after preprocessing

---

**Ready?**
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-delta-hedge
```
