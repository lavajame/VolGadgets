import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple


def solve_scale_invariant_ridge(A: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    # A is m x n (rows: target features, e.g. premium, delta, gamma, [theta])
    V_scaled = alpha * np.diag(np.diag(A.dot(A.T)))
    w = A.T.dot(np.linalg.inv(A.dot(A.T) + V_scaled)).dot(b)
    return w


def find_start_date(df: pd.DataFrame, target_type: str) -> pd.Timestamp:
    dates = sorted(df['quote_date'].dropna().unique())
    for d in dates:
        has_near = ((df['quote_date'] == d) & (df['selection_group'] == 'near_grid')).any()
        has_far = ((df['quote_date'] == d) & (df['selection_group'] == 'far_target') & (df['option_type'] == target_type)).any()
        if has_near and has_far:
            return pd.to_datetime(d)
    raise ValueError('No date found with both near_grid and far_target rows for target_type')


def build_portfolios(df: pd.DataFrame, start_date: pd.Timestamp, target_type: str, alpha: float, constraints_list: list = None, subfolder: str = None, S_min: float = None, S_max: float = None):
    """
    Build near and far portfolios with flexible constraint selection.
    
    constraints_list: list of constraint names, e.g. ['delta', 'gamma', 'vanna', 'theta']
                     Always includes mid/premium. If None, defaults to ['delta', 'gamma']
    S_min, S_max: optional asymptotic boundaries for intrinsic value matching
                 If provided, adds constraints that portfolio values match at these spots
    """
    if constraints_list is None:
        constraints_list = ['delta', 'gamma']
    
    # Normalize constraint names to lowercase
    constraints_list = [c.lower() for c in constraints_list]
    
    # mid/premium is always included
    if 'mid' not in constraints_list:
        constraints_list = ['mid'] + constraints_list
    
    df['quote_date'] = pd.to_datetime(df['quote_date'])
    near0 = df[(df['quote_date'] == start_date) & (df['selection_group'] == 'near_grid')].copy()
    far0 = df[(df['quote_date'] == start_date) & (df['selection_group'] == 'far_target') & (df['option_type'] == target_type)].copy()
    if far0.empty:
        raise ValueError('No far_target found at start date for requested option_type')
    # if multiple far targets pick the first
    far0 = far0.iloc[0]

    # Ensure numeric columns for all standard greeks
    standard_cols = ['mid', 'delta', 'gamma', 'theta', 'vega']
    for c in standard_cols:
        if c in near0.columns:
            near0[c] = pd.to_numeric(near0[c], errors='coerce')

    # Load second-order greeks if needed
    so = None
    if any(c in constraints_list for c in ['vanna', 'volga', 'dtheta_dS', 'dtheta_dS_year']):
        so_path = os.path.join('backtests', subfolder, 'second_order_greeks_day0.csv') if subfolder else None
        if so_path and os.path.exists(so_path):
            so = pd.read_csv(so_path, index_col='contract')
    
    # Build A matrix and b vector dynamically
    rows = []
    b_vals = []
    constraint_names_used = []
    
    for constraint in constraints_list:
        constraint_lower = constraint.lower()
        
        if constraint_lower == 'mid':
            rows.append(near0['mid'].values)
            b_vals.append(float(far0['mid']))
            constraint_names_used.append('mid')
        elif constraint_lower == 'delta':
            rows.append(near0['delta'].values)
            b_vals.append(float(far0['delta']))
            constraint_names_used.append('delta')
        elif constraint_lower == 'gamma':
            rows.append(near0['gamma'].values)
            b_vals.append(float(far0['gamma']))
            constraint_names_used.append('gamma')
        elif constraint_lower == 'vega':
            rows.append(near0['vega'].values)
            b_vals.append(float(far0['vega']))
            constraint_names_used.append('vega')
        elif constraint_lower == 'theta':
            rows.append(near0['theta'].values)
            b_vals.append(float(far0['theta']))
            constraint_names_used.append('theta')
        elif constraint_lower in ['vanna', 'volga', 'dtheta_dS', 'dtheta_dS_year']:
            if so is None:
                print(f"WARNING: {constraint_lower} requested but second_order_greeks_day0.csv not found")
                continue
            
            # Map constraint name to CSV column
            so_col_name = constraint_lower
            if so_col_name not in so.columns:
                print(f"WARNING: {so_col_name} not found in second_order_greeks_day0.csv")
                continue
            
            # Build vector for near_grid contracts
            vec = []
            for c in near0['contract'].values:
                if c in so.index:
                    val = float(so.loc[c, so_col_name]) if pd.notna(so.loc[c, so_col_name]) else 0.0
                    vec.append(val)
                else:
                    vec.append(0.0)
            
            # Get target value
            target_contract = str(far0['contract'])
            target_val = float(so.loc[target_contract, so_col_name]) if target_contract in so.index and pd.notna(so.loc[target_contract, so_col_name]) else 0.0
            
            rows.append(np.array(vec))
            b_vals.append(target_val)
            constraint_names_used.append(constraint_lower)

    # Optional: Add asymptotic intrinsic value constraints at S_min and S_max
    if S_min is not None and S_max is not None:
        # Compute intrinsic values at boundaries
        # Intrinsic value = max(S - K, 0) for calls, max(K - S, 0) for puts
        def intrinsic(S, K, opt_type):
            if opt_type.lower() == 'call':
                return max(S - K, 0)
            else:  # put
                return max(K - S, 0)
        
        # S_min constraint: near portfolio intrinsic at S_min should equal target intrinsic at S_min
        near_strike = pd.to_numeric(near0['strike'], errors='coerce')
        near_type = near0['option_type']
        target_strike = float(far0['strike'])
        target_type_val = far0['option_type']
        
        intrinsic_min_near = np.array([intrinsic(S_min, float(k), t) for k, t in zip(near_strike, near_type)])
        intrinsic_min_target = intrinsic(S_min, target_strike, target_type_val)
        
        intrinsic_max_near = np.array([intrinsic(S_max, float(k), t) for k, t in zip(near_strike, near_type)])
        intrinsic_max_target = intrinsic(S_max, target_strike, target_type_val)
        
        rows.append(intrinsic_min_near)
        b_vals.append(intrinsic_min_target)
        constraint_names_used.append('intrinsic_at_Smin')
        
        rows.append(intrinsic_max_near)
        b_vals.append(intrinsic_max_target)
        constraint_names_used.append('intrinsic_at_Smax')

    A = np.vstack(rows)
    b = np.array(b_vals)

    w = solve_scale_invariant_ridge(A, b, alpha)
    near0 = near0.reset_index(drop=True)
    near0['weight'] = w

    # diagnostics: store all constraints used
    diag = {
        'constraints_used': constraint_names_used,
        'target_mid': b[0] if 'mid' in constraint_names_used else float(far0['mid']),
        'near_mid_replication': float(w.dot(near0['mid'].values)),
        'weights_l2': float(np.sum(w**2)),
        'weights_max_abs': float(np.max(np.abs(w)))
    }
    
    # Add constraint-specific diagnostics
    for i, cname in enumerate(constraint_names_used):
        if cname != 'mid':  # mid already handled
            diag[f'target_{cname}'] = b[i]
            diag[f'near_{cname}_replication'] = float(w.dot(rows[i]))

    # effective implied vol metric
    near0['implied_volatility'] = pd.to_numeric(near0['implied_volatility'], errors='coerce')
    diag['eff_iv_start'] = float(np.sqrt(np.nansum((w * near0['implied_volatility'].fillna(0).values) ** 2)))

    return near0, far0, diag


def evolution_over_time(df: pd.DataFrame, near0: pd.DataFrame, far_target_type: str, delta_hedge: bool = False, rebalance_daily: bool = False, start_date: pd.Timestamp = None) -> pd.DataFrame:
    # static weights from near0
    weights = near0['weight'].values
    contracts = near0['contract'].values

    df['quote_date'] = pd.to_datetime(df['quote_date'])
    dates = sorted(df['quote_date'].unique())
    rows = []
    # prepare hedge bookkeeping
    hedge_size_target_prev = 0.0
    hedge_size_near_prev = 0.0
    prev_date = None

    for d in dates:
        # target row for date
        far = df[(df['quote_date'] == d) & (df['selection_group'] == 'far_target') & (df['option_type'] == far_target_type)]
        if far.empty:
            target_mid = np.nan
            target_delta = np.nan
            target_gamma = np.nan
            target_theta = np.nan
            target_vega = np.nan
        else:
            far = far.iloc[0]
            target_mid = float(far['mid']) if pd.notna(far['mid']) else np.nan
            target_delta = float(far['delta']) if pd.notna(far['delta']) else np.nan
            target_gamma = float(far['gamma']) if pd.notna(far['gamma']) else np.nan
            target_theta = float(far['theta']) if pd.notna(far['theta']) else np.nan
            target_vega = float(far['vega']) if pd.notna(far['vega']) else np.nan

        # near grid rows matching initial contracts
        near_rows = df[(df['quote_date'] == d) & (df['selection_group'] == 'near_grid') & (df['contract'].isin(contracts))]
        # align order to contracts
        near_rows = near_rows.set_index('contract').reindex(contracts).reset_index()

        # fetch numeric columns
        def v(col):
            return pd.to_numeric(near_rows[col], errors='coerce').fillna(0).values

        near_mid = float(np.nansum(weights * v('mid')))
        near_delta = float(-np.nansum(weights * v('delta')))
        near_gamma = float(-np.nansum(weights * v('gamma')))
        near_theta = float(-np.nansum(weights * v('theta')))
        near_vega = float(-np.nansum(weights * v('vega')))

        # implied vol effect
        ivs = pd.to_numeric(near_rows['implied_volatility'], errors='coerce').fillna(0).values
        eff_iv = float(np.sqrt(np.nansum((weights * ivs) ** 2)))

        # values: long target, short near (so near_side is negative of notional of options)
        target_value = target_mid
        near_value_raw = float(-np.nansum(weights * v('mid')))
        near_value = near_value_raw

        # dynamic daily rebalanced hedge handling
        if delta_hedge and rebalance_daily and start_date is not None:
            # on first date we set hedge sizes at close (no pnl yet)
            if prev_date is None and d == start_date:
                # set hedge sizes to neutralize delta at start (placed at close)
                hedge_size_target_prev = 0.0
                hedge_size_near_prev = 0.0
                # compute raw deltas at start
                target_delta0 = float(target_delta) if pd.notna(target_delta) else 0.0
                near_delta0 = float(near_delta) if pd.notna(near_delta) else 0.0
                # hedge size to neutralize (underlying units)
                hedge_size_target_prev = -target_delta0
                hedge_size_near_prev = -near_delta0
                # apply no pnl on start date; but show hedged deltas (after rebalance)
                if not np.isnan(target_delta):
                    target_delta = target_delta + hedge_size_target_prev
                near_delta = near_delta + hedge_size_near_prev
            else:
                # for subsequent dates, apply previous hedge pnl for move from prev_date -> d
                try:
                    S_prev = float(df[df['quote_date'] == prev_date]['underlying_close'].dropna().unique()[0])
                    S_cur = float(df[df['quote_date'] == d]['underlying_close'].dropna().unique()[0])
                    dS_from_prev = S_cur - S_prev
                except Exception:
                    dS_from_prev = 0.0

                hedge_pnl_target = hedge_size_target_prev * dS_from_prev
                hedge_pnl_near = hedge_size_near_prev * dS_from_prev

                if not np.isnan(target_mid):
                    target_mid = float(target_mid) + hedge_pnl_target
                near_value = float(near_value_raw) + hedge_pnl_near

                # after applying prev hedge pnl, compute new hedge sizes at close of current date
                target_delta_curr = float(target_delta) if pd.notna(target_delta) else 0.0
                near_delta_curr = float(near_delta) if pd.notna(near_delta) else 0.0
                hedge_size_target_prev = -target_delta_curr
                hedge_size_near_prev = -near_delta_curr

                # for reporting, show deltas after rebalancing (should be ~0)
                target_delta = target_delta + hedge_size_target_prev
                near_delta = near_delta + hedge_size_near_prev
        net_value = float((target_value if not np.isnan(target_value) else 0.0) + near_value)

        rows.append({
            'quote_date': d,
            'underlying_close': float(df[df['quote_date'] == d]['underlying_close'].dropna().unique()[0]) if not df[df['quote_date'] == d]['underlying_close'].dropna().empty else np.nan,
            'target_mid': target_mid,
            'near_portfolio_mid_short': near_value,
            'net_mid': net_value,
            'target_delta': target_delta,
            'near_delta_short': near_delta,
            'net_delta': (target_delta + near_delta) if not np.isnan(target_delta) else near_delta,
            'target_gamma': target_gamma,
            'near_gamma_short': near_gamma,
            'net_gamma': (target_gamma + near_gamma) if not np.isnan(target_gamma) else near_gamma,
            'target_theta': target_theta,
            'near_theta_short': near_theta,
            'net_theta': (target_theta + near_theta) if not np.isnan(target_theta) else near_theta,
            'target_vega': target_vega,
            'near_vega_short': near_vega,
            'net_vega': (target_vega + near_vega) if not np.isnan(target_vega) else near_vega,
            'near_eff_iv': eff_iv
        })

        # advance prev_date for next iteration
        prev_date = d

    evo = pd.DataFrame(rows)
    return evo


def main():
    p = argparse.ArgumentParser(description='Scale-invariant ridge solver for near_grid -> far_target replication')
    p.add_argument('subfolder', help="Backtests subfolder name under 'backtests' (e.g., 130206_SPX)")
    p.add_argument('--market-csv', help='Optional explicit market CSV path (overrides subfolder)')
    p.add_argument('--target-type', choices=['call', 'put'], default='call', help='Far target option type to replicate')
    p.add_argument('--alpha', type=float, default=1e-4, help='Ridge regularization intensity')
    p.add_argument('--constraints', type=str, default='delta;gamma', help='Semicolon-delimited list of constraints to use (e.g., "delta;gamma;vanna;theta"). Always includes mid.')
    p.add_argument('--S-min', type=float, help='Lower spot boundary for asymptotic intrinsic value constraint (e.g., 1200). If provided with --S-max, adds boundary constraints.')
    p.add_argument('--S-max', type=float, help='Upper spot boundary for asymptotic intrinsic value constraint (e.g., 1800). If provided with --S-min, adds boundary constraints.')
    p.add_argument('--delta-hedge', action='store_true', help='Apply a daily-rebalanced delta hedge to target and near portfolios in evolution.csv (default when set)')
    p.add_argument('--rebalance-daily', action='store_true', help='(Deprecated) kept for compatibility')
    
    # Keep old flags for backward compatibility but don't document them
    p.add_argument('--include-theta', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--include-vanna-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--include-second-order', action='store_true', help=argparse.SUPPRESS)
    
    # outputs are written to backtests/<subfolder> by default
    args = p.parse_args()

    # Handle backward compatibility: convert old flags to constraints list
    if args.include_theta or args.include_vanna_only or args.include_second_order:
        constraints_list = ['delta', 'gamma']
        if args.include_theta:
            constraints_list.append('theta')
        if args.include_vanna_only:
            constraints_list.append('vanna')
        if args.include_second_order:
            constraints_list.extend(['vanna', 'volga', 'dtheta_dS_year'])
    else:
        # Use --constraints flag
        constraints_list = [c.strip() for c in args.constraints.split(';')]

    if args.market_csv:
        market_csv = args.market_csv
    else:
        market_csv = os.path.join('backtests', args.subfolder, 'mkt_data.csv')

    out_dir = os.path.join('backtests', args.subfolder)

    df = pd.read_csv(market_csv, dtype={'contract': str})
    start_date = find_start_date(df, args.target_type)

    # Extract initial spot price for auto-bounds
    S0 = float(df[df['quote_date'] == start_date.strftime('%Y-%m-%d')]['underlying_close'].iloc[0])

    # Check if both S_min and S_max are provided
    S_min = args.S_min if hasattr(args, 'S_min') else None
    S_max = args.S_max if hasattr(args, 'S_max') else None
    
    # Auto-compute bounds if not provided: S_min=0, S_max=10*S0
    if S_min is None and S_max is None:
        S_min = 0.0
        S_max = 10.0 * S0
    elif (S_min is None) != (S_max is None):
        print("WARNING: Both --S-min and --S-max must be provided together. Using auto-bounds.")
        S_min = 0.0
        S_max = 10.0 * S0
    
    near0, far0, diag = build_portfolios(df, start_date, args.target_type, args.alpha, constraints_list=constraints_list, subfolder=args.subfolder, S_min=S_min, S_max=S_max)

    print('\n=== SCALE-INVARIANT RIDGE SOLVER REPORT ===')
    print(f"Start date: {start_date.date()}")
    print(f"Target contract: {far0['contract']}  strike={far0['strike']} type={far0['option_type']}")
    print(f"Initial spot: {S0:.2f}")
    print(f"Alpha: {args.alpha}")
    print(f"Constraints: {', '.join(constraints_list)}")
    print(f"Asymptotic intrinsic value bounds: S_min={S_min:.2f}, S_max={S_max:.2f}")
    print()
    print('--- Initial replication diagnostics ---')
    
    # Print target and replication for each constraint
    constraint_names = diag.get('constraints_used', constraints_list)
    for cname in constraint_names:
        target_key = f'target_{cname}'
        repl_key = f'near_{cname}_replication'
        if target_key in diag and repl_key in diag:
            target_val = diag[target_key]
            repl_val = diag[repl_key]
            print(f"{cname:15s}: target={target_val:12.6f}, replicating={repl_val:12.6f}, error={abs(target_val - repl_val):12.6f}")
    
    print(f"Weights L2: {diag['weights_l2']:.6f}  max abs: {diag['weights_max_abs']:.6f}")
    print(f"Effective IV (start): {diag['eff_iv_start']:.6f}\n")

    os.makedirs(out_dir, exist_ok=True)
    weights_out = os.path.join(out_dir, 'near_weights.csv')
    near0.to_csv(weights_out, index=False)
    print(f'Wrote static weights to {weights_out}')

    # make --delta-hedge imply daily rebalancing by default
    rebalance_flag = bool(args.delta_hedge) or bool(args.rebalance_daily)
    evo = evolution_over_time(df, near0, args.target_type, delta_hedge=args.delta_hedge, rebalance_daily=rebalance_flag, start_date=start_date)
    evo_out = os.path.join(out_dir, 'evolution.csv')
    evo.to_csv(evo_out, index=False)
    print(f'Wrote evolution and greeks to {evo_out}')


if __name__ == '__main__':
    main()
