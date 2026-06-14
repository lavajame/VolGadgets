# Volatility Gadgets: Vega Isolation Solver

A portfolio construction framework for creating **pure vega exposure** by hedging delta and gamma using a scale-invariant ridge regression solver.

## Quick Start

### Run the complete workflow:
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-delta-hedge
```

This will:
1. **Preprocess** market data (repair ITM prices using OTM vol surface)
2. **Compute** second-order Greeks (vanna, volga, dθ/dS)
3. **Solve** for near-grid weights using scale-invariant ridge regression
4. **Generate** plots showing evolution and PnL attribution

Outputs go to `backtests/130103_SPX/plots/`

## What It Does

### Core Problem
You have a target option (far OTM call/put). You want to:
- **Replicate its vega exposure** as purely as possible
- **Hedge delta and gamma** dynamically or statically
- **Isolate vega** from all other Greeks

### Solution
Build a long-short portfolio of near-grid options (liquid strikes) that:
1. **Matches the target's vega**
2. **Hedges delta and gamma** to near-zero (constrained optimization)
3. **Maintains consistent implied vol** across the portfolio

## Key Files

| File | Purpose |
|------|---------|
| `fit_and_evolve.py` | Main orchestration script (entry point) |
| `preprocess_vol_repair.py` | Repair missing ITM prices using OTM vol |
| `compute_second_order_greeks.py` | Pre-compute vanna, volga, dθ/dS |
| `scale_invariant_ridge_solver.py` | Core solver (ridge regression with vol consistency) |
| `plot_evolution.py` | Generate evolution plots + PnL attribution |
| `vega_isolation_diagnostic.py` | Measure vega isolation quality |
| `bs_pricer.py` | Black-Scholes pricing engine |

## Usage Examples

### Basic: Delta & Gamma hedging
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma'
```

### With theta constraint
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta'
```

### With second-order Greeks
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;vanna;volga'
```

### Custom regularization strength
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --alpha 1e-2
```

### No delta hedge (static weights)
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-delta-hedge
```

### Skip preprocessing
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-preprocess
```

## Vega Isolation Measurement

Measure how well different constraint configurations isolate vega:

```bash
python scripts/vega_isolation_diagnostic.py 130103_SPX
```

This outputs:
- **Greek noise**: Sum of |delta| + |gamma| + |theta| (should be <0.01)
- **Vega exposure**: Dollar vega at day 0 (want to maximize)
- **Quality score**: ✓✓✓ (<0.01), ✓✓ (<0.1), ✓ (<1.0), ✗ (>1.0)

### Compare multiple configurations:
```bash
python scripts/compare_constraint_configs.py 130103_SPX
```

Tests 8 different constraint combinations and ranks by vega isolation quality.

## Scale-Invariant Ridge Solver Explained

### The Constraint Matrix
```
A = [ delta_1   delta_2   ...   delta_n  ]
    [ gamma_1   gamma_2   ...   gamma_n  ]
    [ vega_1    vega_2    ...   vega_n   ]
    [ ...                              ]
    [ intrinsic intrinsic ... intrinsic at S_min ]
    [ intrinsic intrinsic ... intrinsic at S_max ]
```

### The Ridge Regularization
```
w = A.T @ inv(A @ A.T + V_scaled) @ b

where V_scaled = alpha * diag(diag(A @ A.T))
```

This maintains **scale invariance**: Greeks are scaled to same order of magnitude before solving.

### Key Parameters
- **`alpha`**: Regularization strength (default: 1e-4)
  - Higher → more spread weights, smoother across strikes
  - Lower → more concentrated weights, better hedging
- **`--S-min`, `--S-max`**: Spot boundaries for intrinsic value anchoring
  - Default: S_min=0, S_max=10×S₀
  - Forces portfolio to have same payoff at boundaries

## Market Data Preprocessing

The pipeline automatically repairs corrupted market data:

### Problem
- Deep-ITM options sometimes show zero bid/ask while OTM counterparts trade
- Creates phantom "zero prices" that break the solver

### Solution
- Extract IV from liquid (OTM) side
- Use Black-Scholes to regenerate prices for illiquid (ITM) side
- Preserve vol surface consistency

See `VOL_SURFACE_REPAIR.md` for details.

## Outputs

### Daily Files
- `evolution.csv`: Spot, portfolio values, Greeks over time
- `near_weights.csv`: Solved portfolio weights
- `diagnostics.csv`: Constraint satisfaction report

### Plots
- `evolution_*.png`: Greeks and portfolio value evolution
- `pnl_attribution_*.png`: Daily PnL decomposition (8 factors)

### Data Quality
- `second_order_greeks_day0.csv`: Pre-computed vanna, volga, dθ/dS

## PnL Attribution (8-Factor Decomposition)

Each day's PnL is decomposed into:

1. **Delta**: Δ₀ × dS
2. **Gamma**: 0.5 × Γ₀ × dS²
3. **Vega**: ν₀ × dIV
4. **Vanna**: Ψ₀ × dS × dIV (cross-Greek)
5. **Volga**: Λ₀ × (dIV)²/2 (convexity in vol)
6. **dTheta/dS**: ∂θ/∂S × dS (theta drift)
7. **Theta**: θ₀ × dt (time decay)
8. **Residual**: Unexplained PnL

All Greeks scaled to daily values (per-year Greeks ÷ 365).

## Example Dataset: 130103_SPX

16 trading days (Jan 3-25, 2013) with:
- **Target**: Feb 1 ATM call (SPXW130201C01460000)
- **Near-grid**: 10 strike call/put pairs (SPX weeklies Jan 25)
- **Data quality**: Pre-cleaned with vol surface repair

Run the example:
```bash
python scripts/fit_and_evolve.py 130103_SPX --constraints 'delta;gamma;theta' --no-delta-hedge
```

## Interpretation Tips

### Good Portfolio Isolation
- Greek noise < 0.01 (✓✓✓ quality)
- Smooth weight distribution across strikes
- Vega exposure > 80 $/1% vol
- Minimal PnL from delta/gamma/theta in attribution plots

### Signs of Trouble
- Weights concentrated at 1-2 strikes (ill-posed problem)
- Negative gamma despite hedging (solver overfitting)
- Large residuals in PnL attribution (data/pricing issues)
- Sudden portfolio jumps (data corruption)

## References

### Key Papers
- Hull & White: "The Vol of Vol" - Vanna/Volga pricing model
- Fengler et al.: "Scale-Invariant Pricing and Hedging"

### Files to Read First
1. `VOL_SURFACE_REPAIR.md` - Data preprocessing
2. `DATA_QUALITY_GUIDE.md` - Market data validation
3. Run `vega_isolation_diagnostic.py` to understand quality metrics

## Troubleshooting

### Error: "No near_grid data"
- Check your backtest folder has `mkt_data.csv` with near_grid rows

### Large weights (>10)
- Likely ill-posed problem (not enough constraints vs. options)
- Try reducing strikes or adding constraints

### Negative gamma in portfolio
- Overfitting; try increasing alpha (less regularization)

### Sudden PnL jumps
- Check `validate_mkt_data.py` for data corruption
- Rerun with `--no-preprocess` to see raw data issues

## Contact & Questions

For questions on:
- **Solver theory**: See scale_invariant_ridge_solver.py docstring
- **Data preprocessing**: See VOL_SURFACE_REPAIR.md
- **Greeks computation**: See bs_pricer.py and compute_second_order_greeks.py
