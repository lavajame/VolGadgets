import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bs_pricer import black_scholes_greeks, black_scholes_price


def compute_second_order_greeks(S, K, T, r, q, sigma, option_type):
    """Compute second-order greeks (vanna, volga, dtheta/dS) via finite differences."""
    eps_S_frac = 1e-3
    eps_sigma = 1e-4
    
    eps_S = eps_S_frac * S
    
    # Base greeks
    d0 = black_scholes_greeks(S, K, T, r, q, sigma, option_type)
    delta0, gamma0, vega0, theta0 = d0
    
    # For vanna = d(vega)/dS
    # Vega at S + eps_S
    d_up = black_scholes_greeks(S + eps_S, K, T, r, q, sigma, option_type)
    vega_up = d_up[2]
    # Vega at S - eps_S
    d_dn = black_scholes_greeks(S - eps_S, K, T, r, q, sigma, option_type)
    vega_dn = d_dn[2]
    vanna = (vega_up - vega_dn) / (2 * eps_S)
    
    # For volga = d(vega)/d(sigma)
    vega_high = black_scholes_greeks(S, K, T, r, q, sigma + eps_sigma, option_type)[2]
    vega_low = black_scholes_greeks(S, K, T, r, q, sigma - eps_sigma, option_type)[2]
    volga = (vega_high - vega_low) / (2 * eps_sigma)
    
    # For dtheta/dS = d(theta)/dS
    theta_up = black_scholes_greeks(S + eps_S, K, T, r, q, sigma, option_type)[3]
    theta_dn = black_scholes_greeks(S - eps_S, K, T, r, q, sigma, option_type)[3]
    dtheta_dS = (theta_up - theta_dn) / (2 * eps_S)
    
    return vanna, volga, dtheta_dS


def plot_mid(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['target_mid'], label='Target mid', marker='o')
    ax.plot(df.index, df['near_portfolio_mid_short'], label='Near portfolio (short)', marker='o')
    ax.plot(df.index, df['net_mid'], label='Net mid', marker='o')
    ax.set_title('MTM Evolution (mid prices)')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    out = os.path.join(out_dir, 'mid_evolution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_greeks(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    axes[0].plot(df.index, df['target_delta'], label='Target', marker='o')
    axes[0].plot(df.index, df['near_delta_short'], label='Near short', marker='o')
    axes[0].plot(df.index, df['net_delta'], label='Net', marker='o')
    axes[0].set_title('Delta')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(df.index, df['target_gamma'], label='Target', marker='o')
    axes[1].plot(df.index, df['near_gamma_short'], label='Near short', marker='o')
    axes[1].plot(df.index, df['net_gamma'], label='Net', marker='o')
    axes[1].set_title('Gamma')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(df.index, df['target_theta'], label='Target', marker='o')
    axes[2].plot(df.index, df['near_theta_short'], label='Near short', marker='o')
    axes[2].plot(df.index, df['net_theta'], label='Net', marker='o')
    axes[2].set_title('Theta')
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(df.index, df['target_vega'], label='Target', marker='o')
    axes[3].plot(df.index, df['near_vega_short'], label='Near short', marker='o')
    axes[3].plot(df.index, df['net_vega'], label='Net', marker='o')
    axes[3].set_title('Vega')
    axes[3].legend(); axes[3].grid(True)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()
    out = os.path.join(out_dir, 'greeks_evolution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_near_weights(subfolder: str, out_dir: str):
    """Plot near grid call and put weights by strike on same chart."""
    weights_path = os.path.join('backtests', subfolder, 'near_weights.csv')
    if not os.path.exists(weights_path):
        return ''
    
    nw = pd.read_csv(weights_path, dtype={'contract': str})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # calls
    calls = nw[nw['option_type'] == 'call'].sort_values('strike')
    if not calls.empty:
        ax.plot(calls['strike'], calls['weight'], marker='o', linestyle='-', linewidth=2.5, markersize=7, color='steelblue', label='Calls')
    
    # puts
    puts = nw[nw['option_type'] == 'put'].sort_values('strike')
    if not puts.empty:
        ax.plot(puts['strike'], puts['weight'], marker='s', linestyle='-', linewidth=2.5, markersize=7, color='coral', label='Puts')
    
    ax.set_title('Near Portfolio Weights by Strike and Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strike', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    
    fig.tight_layout()
    out = os.path.join(out_dir, 'near_weights_by_strike.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out



def plot_iv(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['near_eff_iv'], label='Near effective IV', marker='o')
    ax.set_title('Effective IV of Near Portfolio')
    ax.set_ylabel('IV')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    out = os.path.join(out_dir, 'near_eff_iv.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_pnl_attribution(df: pd.DataFrame, out_dir: str):
    # compute pnl relative to first available date (mark-to-market)
    baseline = df.iloc[0]
    pnl_target = df['target_mid'] - baseline['target_mid']
    pnl_near = df['near_portfolio_mid_short'] - baseline['near_portfolio_mid_short']
    pnl_net = df['net_mid'] - baseline['net_mid']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, pnl_target, label='Target PnL (long)', marker='o')
    ax.plot(df.index, pnl_near, label='Near Portfolio PnL (short)', marker='o')
    ax.plot(df.index, pnl_net, label='Net PnL', marker='o')
    ax.set_title('PnL Attribution (MTM from start)')
    ax.set_ylabel('PnL')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    out = os.path.join(out_dir, 'pnl_attribution.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def print_premium_greeks_table(df: pd.DataFrame, subfolder: str = None, target_type: str = 'call'):
    if df.shape[0] == 0:
        print('No data to summarize')
        return
    # use initial date (first row)
    first_date = df.index.min()
    first = df.loc[first_date]
    def col(name):
        return first[name] if name in first.index else 0.0

    # initialize greeks (will be read from market data if available, not from evolution.csv which may have hedge logic applied)
    target_mid = target_delta = target_gamma = target_vega = target_theta = 0.0
    near_mid = near_delta = near_gamma = near_vega = near_theta = 0.0
    net_mid = col('net_mid')

    # prepare optional second-order greeks (vanna, volga, dtheta/dS)
    target_vanna = target_volga = target_dtheta_dS_year = 0.0
    near_vanna = near_volga = near_dtheta_dS_year = 0.0

    if subfolder is not None:
        so_path = os.path.join('backtests', subfolder, 'second_order_greeks_day0.csv')
        weights_path = os.path.join('backtests', subfolder, 'near_weights.csv')
        mkt_path = os.path.join('backtests', subfolder, 'mkt_data.csv')
        
        # read unhedged greeks from market data at day 0
        if os.path.exists(mkt_path):
            mkt = pd.read_csv(mkt_path, parse_dates=['quote_date'], dtype={'contract': str})
            
            # target greeks (far_target at first date)
            far_rows = mkt[(mkt['quote_date'] == first_date) & (mkt['selection_group'] == 'far_target') & (mkt['option_type'] == target_type)]
            if not far_rows.empty:
                far_row = far_rows.iloc[0]
                target_mid = float(far_row['mid']) if pd.notna(far_row.get('mid', np.nan)) else 0.0
                target_delta = float(far_row['delta']) if pd.notna(far_row.get('delta', np.nan)) else 0.0
                target_gamma = float(far_row['gamma']) if pd.notna(far_row.get('gamma', np.nan)) else 0.0
                target_vega = float(far_row['vega']) if pd.notna(far_row.get('vega', np.nan)) else 0.0
                target_theta = float(far_row['theta']) if pd.notna(far_row.get('theta', np.nan)) else 0.0
                target_contract = str(far_row['contract'])
            
            # near portfolio greeks: sum weighted contributions from near_grid at first date
            if os.path.exists(weights_path):
                nw = pd.read_csv(weights_path, dtype={'contract': str})
                contracts = list(nw['contract'].values)
                wmap = dict(zip(nw['contract'].values, nw['weight'].values))
                
                near_rows = mkt[(mkt['quote_date'] == first_date) & (mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))]
                if not near_rows.empty:
                    for _, row in near_rows.iterrows():
                        c = str(row['contract'])
                        w = wmap.get(c, 0.0)
                        mid_val = float(row['mid']) if pd.notna(row.get('mid', np.nan)) else 0.0
                        delta_val = float(row['delta']) if pd.notna(row.get('delta', np.nan)) else 0.0
                        gamma_val = float(row['gamma']) if pd.notna(row.get('gamma', np.nan)) else 0.0
                        vega_val = float(row['vega']) if pd.notna(row.get('vega', np.nan)) else 0.0
                        theta_val = float(row['theta']) if pd.notna(row.get('theta', np.nan)) else 0.0
                        # near portfolio is short, so negate weights
                        near_mid += -w * mid_val
                        near_delta += -w * delta_val
                        near_gamma += -w * gamma_val
                        near_vega += -w * vega_val
                        near_theta += -w * theta_val
        
        # now load second-order greeks if available
        if os.path.exists(so_path):
            so = pd.read_csv(so_path, index_col='contract')
            # target second-order greeks
            if 'target_contract' in locals() and target_contract in so.index:
                target_vanna = float(so.loc[target_contract].get('vanna', 0.0))
                target_volga = float(so.loc[target_contract].get('volga', 0.0))
                target_dtheta_dS_year = float(so.loc[target_contract].get('dtheta_dS_year', 0.0))
            
            # near aggregated second-order greeks
            if os.path.exists(weights_path):
                nw = pd.read_csv(weights_path, dtype={'contract': str})
                # join with so but only pull necessary second-order columns to avoid name collisions
                so_sel = so[['vanna', 'volga', 'dtheta_dS_year']].copy() if set(['vanna', 'volga', 'dtheta_dS_year']).issubset(so.columns) else so.copy()
                merged = nw.set_index('contract').join(so_sel, how='left')
                # weighted sum (near is short, so use negative sign)
                w = merged['weight'].fillna(0.0).values
                vanna_vals = merged['vanna'].fillna(0.0).values
                volga_vals = merged['volga'].fillna(0.0).values
                dtheta_vals = merged['dtheta_dS_year'].fillna(0.0).values
                near_vanna = -float((w * vanna_vals).sum())
                near_volga = -float((w * volga_vals).sum())
                near_dtheta_dS_year = -float((w * dtheta_vals).sum())

    # compute net greeks
    net_delta = target_delta + near_delta
    net_gamma = target_gamma + near_gamma
    net_vega = target_vega + near_vega
    net_theta = target_theta + near_theta

    rows = [
        {'Portfolio': 'Target', 'Premium': target_mid, 'Delta': target_delta, 'Gamma': target_gamma, 'Vega': target_vega, 'Theta': target_theta, 'Vanna': target_vanna, 'Volga': target_volga, 'dTheta_dS_year': target_dtheta_dS_year},
        {'Portfolio': 'Replicating (near)', 'Premium': near_mid, 'Delta': near_delta, 'Gamma': near_gamma, 'Vega': near_vega, 'Theta': near_theta, 'Vanna': near_vanna, 'Volga': near_volga, 'dTheta_dS_year': near_dtheta_dS_year},
        {'Portfolio': 'Net', 'Premium': net_mid, 'Delta': net_delta, 'Gamma': net_gamma, 'Vega': net_vega, 'Theta': net_theta, 'Vanna': (target_vanna + near_vanna), 'Volga': (target_volga + near_volga), 'dTheta_dS_year': (target_dtheta_dS_year + near_dtheta_dS_year)},
    ]

    tbl = pd.DataFrame(rows).set_index('Portfolio')
    # format numbers
    def fmt(x):
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    print(f'\nPremium and Greeks (initial date: {first_date.date()})')
    print(tbl.applymap(fmt).to_string())


def main():
    import warnings
    warnings.filterwarnings('ignore')

    p = argparse.ArgumentParser(description='Plot evolution CSV produced by solver')
    p.add_argument('subfolder', help="Backtests subfolder name under 'backtests' (e.g., 130206_SPX)")
    p.add_argument('--evolution-csv', help='Optional explicit evolution CSV path (overrides subfolder)')
    p.add_argument('--target-type', choices=['call', 'put'], default='call', help='Far target option type to use when sourcing implied vols')
    p.add_argument('--near-eff-iv-day0', type=float, default=0.097210558, help='Desired near effective IV at day0 to scale to')
    p.add_argument('--rebalance-daily', action='store_true', help='(Deprecated) kept for compatibility')
    p.add_argument('--delta-hedge', action='store_true', help='Overlay a daily-rebalanced delta hedge on net portfolio plots (default)')
    # outputs written to backtests/<subfolder> by default
    args = p.parse_args()

    if args.evolution_csv:
        evo_path = args.evolution_csv
    else:
        evo_path = os.path.join('backtests', args.subfolder, 'evolution.csv')

    out_dir = os.path.join('backtests', args.subfolder)

    df = pd.read_csv(evo_path, parse_dates=['quote_date'])
    df = df.set_index('quote_date').sort_index()

    # print summary table of premium and greeks for latest date
    print_premium_greeks_table(df, args.subfolder, args.target_type)

    os.makedirs(out_dir, exist_ok=True)

    mid_png = plot_mid(df, out_dir)
    greeks_png = plot_greeks(df, out_dir)
    pnl_png = plot_pnl_attribution(df, out_dir)
    weights_png = plot_near_weights(args.subfolder, out_dir)
    iv_comp_png = plot_iv_comparison(df, out_dir, args.subfolder, args.target_type, args.near_eff_iv_day0)
    # compute BS-based PnL attribution (uses BS repricing + greeks)
    # make --delta-hedge imply daily rebalancing by default
    rebalance_flag = bool(args.delta_hedge) or bool(args.rebalance_daily)
    pf_near, pf_near_cum, pf_target, pf_target_cum = compute_pnl_factors_bs(args.subfolder, target_type=args.target_type, delta_hedge=args.delta_hedge, rebalance_daily=rebalance_flag) or (None, None, None, None)
    pnl_cum_png = ''
    target_cum_png = ''
    net_cum_png = ''
    if pf_near_cum is not None:
        pnl_cum_png = plot_pnl_factors_cum(pf_near_cum, out_dir, 'Near (short)')
    if pf_target_cum is not None:
        target_cum_png = plot_pnl_factors_cum(pf_target_cum, out_dir, 'Target (long)')

    # Net = Target (long) + Near (short)
    if pf_near_cum is not None and pf_target_cum is not None:
        pf_net_cum = pf_target_cum.add(pf_near_cum, fill_value=0.0)
        # If requested, overlay a static day0 delta hedge: hedge size = -net_delta0
        if args.delta_hedge:
            if 'net_delta' in df.columns:
                net_delta0 = float(df['net_delta'].iloc[0])
                hedge_size = -net_delta0
                # cumulative hedge pnl = hedge_size * cumulative underlying changes from start
                underlying = df['underlying_close'] if 'underlying_close' in df.columns else None
                if underlying is not None:
                    hedge_cum = (underlying.diff().fillna(0).cumsum() * hedge_size).reindex(pf_net_cum.index).fillna(method='ffill').fillna(0.0)
                    # add hedge cumulative pnl into delta column
                    if 'delta' in pf_net_cum.columns:
                        pf_net_cum['delta'] = pf_net_cum['delta'] + hedge_cum
                    else:
                        pf_net_cum['delta'] = hedge_cum
        net_cum_png = plot_pnl_factors_cum(pf_net_cum, out_dir, 'Net')

    print('Saved plots:')
    print(mid_png)
    print(greeks_png)
    print(pnl_png)
    print(weights_png)
    print(iv_comp_png)
    if pnl_cum_png:
        print(pnl_cum_png)
    if target_cum_png:
        print(target_cum_png)
    if net_cum_png:
        print(net_cum_png)


def plot_iv_comparison(df: pd.DataFrame, out_dir: str, subfolder: str, target_type: str, desired_day0: float):
    # read market data to get target implied vol per date
    mkt_path = os.path.join('backtests', subfolder, 'mkt_data.csv')
    if not os.path.exists(mkt_path):
        return ''
    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date'])
    mkt_target = mkt[(mkt['selection_group'] == 'far_target') & (mkt['option_type'] == target_type)][['quote_date', 'implied_volatility']]
    mkt_target = mkt_target.dropna(subset=['implied_volatility']).drop_duplicates(subset=['quote_date']).set_index('quote_date').sort_index()

    target_iv = mkt_target['implied_volatility']

    # compute raw near effective IV from near_weights and market data using formula sqrt(sum(w * IV^2)) per your request
    weights_path = os.path.join('backtests', subfolder, 'near_weights.csv')
    if not os.path.exists(weights_path):
        return ''
    near0 = pd.read_csv(weights_path, dtype={'contract': str})
    # select near_grid rows for contracts in near_weights
    contracts = list(near0['contract'].values)
    wmap = dict(zip(near0['contract'].values, near0['weight'].values))

    m_near = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))][['quote_date', 'contract', 'implied_volatility']].copy()
    m_near['implied_volatility'] = pd.to_numeric(m_near['implied_volatility'], errors='coerce')

    # group by date and compute sqrt(sum(w * iv^2))
    def compute_iv_for_date(g):
        s = 0.0
        for _, r in g.iterrows():
            w = wmap.get(r['contract'], 0.0)
            iv = r['implied_volatility'] if pd.notna(r['implied_volatility']) else 0.0
            s += w * (iv ** 2)
        return np.sqrt(max(s, 0.0))

    near_iv_series = m_near.groupby('quote_date').apply(compute_iv_for_date)
    combined = df.join(target_iv.rename('target_iv'), how='left')
    combined = combined.join(near_iv_series.rename('near_iv_from_weights'), how='left')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(combined.index, combined['near_iv_from_weights'], label='Near effective IV (sqrt(sum(w * IV^2)))', marker='o')
    ax.plot(combined.index, combined['target_iv'], label='Target implied IV', marker='o')
    ax.set_title('Near Effective IV vs Target Implied IV')
    ax.set_ylabel('IV')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    out = os.path.join(out_dir, 'iv_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def compute_theta_scale(subfolder: str, target_type: str) -> float:
    # Compute median scale factor to convert market theta to BS theta-per-day
    base = os.path.join('backtests', subfolder)
    mkt_path = os.path.join(base, 'mkt_data.csv')
    weights_path = os.path.join(base, 'near_weights.csv')
    if not os.path.exists(mkt_path) or not os.path.exists(weights_path):
        return 1.0

    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date', 'expiration'], dtype={'contract': str})
    near0 = pd.read_csv(weights_path, dtype={'contract': str})
    contracts = list(near0['contract'].values)

    m = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))].copy()
    if m.empty:
        return 1.0

    ratios = []
    for _, row in m.iterrows():
        try:
            S = float(row['underlying_close'])
            K = float(row['strike'])
            exp = pd.to_datetime(row['expiration'])
            q = 0.0
            r = 0.0
            T = max((exp - row['quote_date']).days / 365.0, 1/365.0)
            iv = float(row['implied_volatility']) if pd.notna(row['implied_volatility']) else None
            mkt_theta = float(row['theta']) if pd.notna(row['theta']) else None
            if iv is None or mkt_theta is None or mkt_theta == 0.0:
                continue
            _, _, _, bs_theta_year = black_scholes_greeks(S, K, T, r, q, iv, row['option_type'])
            bs_theta_day = bs_theta_year / 365.0
            # avoid division by zero
            if mkt_theta == 0:
                continue
            ratios.append({'day_ratio': bs_theta_day / mkt_theta, 'year_ratio': bs_theta_year / mkt_theta})
        except Exception:
            continue

    if len(ratios) == 0:
        return 1.0

    df_r = pd.DataFrame(ratios)
    med_day = float(df_r['day_ratio'].median())
    med_year = float(df_r['year_ratio'].median())

    # If market theta matches BS yearly theta (median ~1), then market theta is per-year.
    # In that case, use scale = 1/365 to convert market theta (per-year) to per-day.
    if abs(med_year - 1.0) < 0.02:
        return 1.0 / 365.0

    # fallback: use median day-ratio
    return med_day


def compute_pnl_factors(subfolder: str, theta_scale: float = 1.0):
    # Read near_weights and market data to compute daily factor PnL for the near portfolio
    base = os.path.join('backtests', subfolder)
    weights_path = os.path.join(base, 'near_weights.csv')
    mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(weights_path) or not os.path.exists(mkt_path):
        return None

    near0 = pd.read_csv(weights_path, dtype={'contract': str})
    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date'], dtype={'contract': str})

    contracts = list(near0['contract'].values)
    wmap = dict(zip(near0['contract'].values, near0['weight'].values))

    # Filter near_grid rows for these contracts
    m = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))].copy()
    if m.empty:
        return None

    # Ensure numeric columns
    for c in ['delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'mid']:
        m[c] = pd.to_numeric(m[c], errors='coerce')

    # underlying close per date
    underlying = mkt[['quote_date', 'underlying_close']].drop_duplicates(subset=['quote_date']).set_index('quote_date').sort_index()

    dates = sorted(underlying.index.unique())

    rows = []
    prev_underlying = None
    prev_iv = {}
    prev_mid = {}
    prev_delta = {}
    prev_gamma = {}
    prev_theta = {}
    prev_vega = {}
    prev_date = None

    # create dict of market rows by (date, contract)
    m_idx = m.set_index(['quote_date', 'contract'])

    for d in dates:
        if prev_date is None:
            # initialize
            rows.append({'quote_date': d, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'residual': 0.0, 'total': 0.0})
            prev_underlying = underlying.loc[d, 'underlying_close'] if d in underlying.index else np.nan
            prev_date = d
            # record prev IVs and mids
            for c in contracts:
                key = (d, c)
                if key in m_idx.index:
                    row0 = m_idx.loc[key]
                    prev_iv[c] = row0['implied_volatility'] if 'implied_volatility' in m_idx.columns else np.nan
                    prev_mid[c] = row0['mid'] if 'mid' in m_idx.columns else np.nan
                    prev_delta[c] = row0['delta'] if 'delta' in m_idx.columns else 0.0
                    prev_gamma[c] = row0['gamma'] if 'gamma' in m_idx.columns else 0.0
                    prev_theta[c] = row0['theta'] if 'theta' in m_idx.columns else 0.0
                    prev_vega[c] = row0['vega'] if 'vega' in m_idx.columns else 0.0
                else:
                    prev_iv[c] = np.nan
                    prev_mid[c] = np.nan
                    prev_delta[c] = 0.0
                    prev_gamma[c] = 0.0
                    prev_theta[c] = 0.0
                    prev_vega[c] = 0.0
            continue

        cur_underlying = underlying.loc[d, 'underlying_close'] if d in underlying.index else np.nan
        dS = float(cur_underlying - prev_underlying) if (pd.notna(cur_underlying) and pd.notna(prev_underlying)) else 0.0
        dt = (d - prev_date).days

        delta_sum = 0.0
        gamma_sum = 0.0
        vega_sum = 0.0
        theta_sum = 0.0
        residual_sum = 0.0

        for c in contracts:
            key = (d, c)
            if key in m_idx.index:
                row = m_idx.loc[key]
                cur_iv = row['implied_volatility']
                cur_mid = row['mid']
                cur_delta = row['delta']
                cur_gamma = row['gamma']
                cur_theta = row['theta']
                cur_vega = row['vega']
            else:
                cur_iv = np.nan
                cur_mid = np.nan
                cur_delta = 0.0
                cur_gamma = 0.0
                cur_theta = 0.0
                cur_vega = 0.0

            dIV = float(cur_iv - prev_iv.get(c, np.nan)) if pd.notna(cur_iv) and pd.notna(prev_iv.get(c, np.nan)) else 0.0
            dmid = float(cur_mid - prev_mid.get(c, np.nan)) if pd.notna(cur_mid) and pd.notna(prev_mid.get(c, np.nan)) else 0.0

            # use previous greeks for PnL approximation (more stable attribution)
            pdlt = prev_delta.get(c, 0.0)
            pgam = prev_gamma.get(c, 0.0)
            pveg = prev_vega.get(c, 0.0)
            ptht = prev_theta.get(c, 0.0)
            # scale market-reported theta to match BS analytic theta-per-day
            ptht_scaled = ptht * theta_scale

            delta_pc = pdlt * dS
            gamma_pc = 0.5 * pgam * (dS ** 2)
            vega_pc = pveg * dIV
            # dt is in days; ptht_scaled is interpreted as per-day theta
            theta_pc = ptht_scaled * dt

            w = wmap.get(c, 0.0)
            delta_sum += w * delta_pc
            gamma_sum += w * gamma_pc
            vega_sum += w * vega_pc
            theta_sum += w * theta_pc

            # update prev for next day
            prev_iv[c] = cur_iv
            prev_mid[c] = cur_mid
            prev_delta[c] = cur_delta
            prev_gamma[c] = cur_gamma
            prev_theta[c] = cur_theta
            prev_vega[c] = cur_vega

        # compute actual weighted portfolio mid change (short position)
        # dP = - sum_w dmid
        dP = 0.0
        for c in contracts:
            key = (d, c)
            if key in m_idx.index:
                cur_mid = m_idx.loc[key]['mid']
                prev_m = prev_mid.get(c, np.nan)
                if pd.notna(cur_mid) and pd.notna(prev_m):
                    dP += - wmap.get(c, 0.0) * (cur_mid - prev_m)

        # contributions for short are negative of weighted sums computed above
        delta_sum_short = -delta_sum
        gamma_sum_short = -gamma_sum
        vega_sum_short = -vega_sum
        theta_sum_short = -theta_sum

        # residual is whatever is left from actual dP
        residual_sum_short = dP - (delta_sum_short + gamma_sum_short + vega_sum_short + theta_sum_short)

        total = delta_sum_short + gamma_sum_short + vega_sum_short + theta_sum_short + residual_sum_short

        rows.append({'quote_date': d, 'delta': delta_sum_short, 'gamma': gamma_sum_short, 'vega': vega_sum_short, 'theta': theta_sum_short, 'residual': residual_sum_short, 'total': total})

        prev_underlying = cur_underlying
        prev_date = d

    pf = pd.DataFrame(rows).set_index('quote_date')
    pf = pf.fillna(0.0)
    pf_cum = pf.cumsum()
    return pf, pf_cum


def compute_pnl_factors_target(subfolder: str, theta_scale: float = 1.0, target_type: str = 'call'):
    # Compute PnL factor contributions for the far target (long)
    base = os.path.join('backtests', subfolder)
    mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(mkt_path):
        return None

    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date', 'expiration'], dtype={'contract': str})
    # select far target rows of the requested option type
    m = mkt[(mkt['selection_group'] == 'far_target') & (mkt['option_type'] == target_type)].copy()
    if m.empty:
        return None

    # Ensure numeric columns
    for c in ['delta', 'gamma', 'theta', 'vega', 'implied_volatility', 'mid']:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors='coerce')

    # underlying close per date
    underlying = mkt[['quote_date', 'underlying_close']].drop_duplicates(subset=['quote_date']).set_index('quote_date').sort_index()
    dates = sorted(underlying.index.unique())

    rows = []
    prev_underlying = None
    prev_iv = {}
    prev_mid = {}
    prev_delta = {}
    prev_gamma = {}
    prev_theta = {}
    prev_vega = {}
    prev_date = None

    # create dict of market rows by (date, contract)
    m_idx = m.set_index(['quote_date', 'contract'])

    # build set of contracts seen
    contracts = sorted(m['contract'].dropna().unique())

    for d in dates:
        if prev_date is None:
            # initialize
            rows.append({'quote_date': d, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'residual': 0.0, 'total': 0.0})
            prev_underlying = underlying.loc[d, 'underlying_close'] if d in underlying.index else np.nan
            prev_date = d
            # record prev IVs and mids
            for c in contracts:
                key = (d, c)
                if key in m_idx.index:
                    row0 = m_idx.loc[key]
                    prev_iv[c] = row0['implied_volatility'] if 'implied_volatility' in m_idx.columns else np.nan
                    prev_mid[c] = row0['mid'] if 'mid' in m_idx.columns else np.nan
                    prev_delta[c] = row0['delta'] if 'delta' in m_idx.columns else 0.0
                    prev_gamma[c] = row0['gamma'] if 'gamma' in m_idx.columns else 0.0
                    prev_theta[c] = row0['theta'] if 'theta' in m_idx.columns else 0.0
                    prev_vega[c] = row0['vega'] if 'vega' in m_idx.columns else 0.0
                else:
                    prev_iv[c] = np.nan
                    prev_mid[c] = np.nan
                    prev_delta[c] = 0.0
                    prev_gamma[c] = 0.0
                    prev_theta[c] = 0.0
                    prev_vega[c] = 0.0
            continue

        cur_underlying = underlying.loc[d, 'underlying_close'] if d in underlying.index else np.nan
        dS = float(cur_underlying - prev_underlying) if (pd.notna(cur_underlying) and pd.notna(prev_underlying)) else 0.0
        dt = (d - prev_date).days

        delta_sum = 0.0
        gamma_sum = 0.0
        vega_sum = 0.0
        theta_sum = 0.0
        residual_sum = 0.0

        for c in contracts:
            key = (d, c)
            if key in m_idx.index:
                row = m_idx.loc[key]
                cur_iv = row['implied_volatility']
                cur_mid = row['mid']
                cur_delta = row['delta']
                cur_gamma = row['gamma']
                cur_theta = row['theta']
                cur_vega = row['vega']
            else:
                cur_iv = np.nan
                cur_mid = np.nan
                cur_delta = 0.0
                cur_gamma = 0.0
                cur_theta = 0.0
                cur_vega = 0.0

            dIV = float(cur_iv - prev_iv.get(c, np.nan)) if pd.notna(cur_iv) and pd.notna(prev_iv.get(c, np.nan)) else 0.0
            dmid = float(cur_mid - prev_mid.get(c, np.nan)) if pd.notna(cur_mid) and pd.notna(prev_mid.get(c, np.nan)) else 0.0

            # use previous greeks for PnL approximation (more stable attribution)
            pdlt = prev_delta.get(c, 0.0)
            pgam = prev_gamma.get(c, 0.0)
            pveg = prev_vega.get(c, 0.0)
            ptht = prev_theta.get(c, 0.0)
            ptht_scaled = ptht * theta_scale

            delta_pc = pdlt * dS
            gamma_pc = 0.5 * pgam * (dS ** 2)
            vega_pc = pveg * dIV
            theta_pc = ptht_scaled * dt

            w = 1.0
            delta_sum += w * delta_pc
            gamma_sum += w * gamma_pc
            vega_sum += w * vega_pc
            theta_sum += w * theta_pc

            # update prev for next day
            prev_iv[c] = cur_iv
            prev_mid[c] = cur_mid
            prev_delta[c] = cur_delta
            prev_gamma[c] = cur_gamma
            prev_theta[c] = cur_theta
            prev_vega[c] = cur_vega

        # compute actual weighted portfolio mid change (long position for target)
        dP = 0.0
        for c in contracts:
            key = (d, c)
            if key in m_idx.index:
                cur_mid = m_idx.loc[key]['mid']
                prev_m = prev_mid.get(c, np.nan)
                if pd.notna(cur_mid) and pd.notna(prev_m):
                    dP += 1.0 * (cur_mid - prev_m)

        delta_sum_long = delta_sum
        gamma_sum_long = gamma_sum
        vega_sum_long = vega_sum
        theta_sum_long = theta_sum

        residual_sum_long = dP - (delta_sum_long + gamma_sum_long + vega_sum_long + theta_sum_long)

        total = delta_sum_long + gamma_sum_long + vega_sum_long + theta_sum_long + residual_sum_long

        rows.append({'quote_date': d, 'delta': delta_sum_long, 'gamma': gamma_sum_long, 'vega': vega_sum_long, 'theta': theta_sum_long, 'residual': residual_sum_long, 'total': total})

        prev_underlying = cur_underlying
        prev_date = d

    pf = pd.DataFrame(rows).set_index('quote_date')
    pf = pf.fillna(0.0)
    pf_cum = pf.cumsum()
    return pf, pf_cum


def compute_pnl_factors_bs(subfolder: str, target_type: str = 'call', delta_hedge: bool = False, rebalance_daily: bool = False):
    """Compute PnL attribution using Black-Scholes repricing and BS greeks.
    Returns (pf_near, pf_near_cum, pf_target, pf_target_cum)
    """
    base = os.path.join('backtests', subfolder)
    weights_path = os.path.join(base, 'near_weights.csv')
    mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(weights_path) or not os.path.exists(mkt_path):
        return None

    near0 = pd.read_csv(weights_path, dtype={'contract': str})
    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date', 'expiration'], dtype={'contract': str})

    contracts = list(near0['contract'].values)
    wmap = dict(zip(near0['contract'].values, near0['weight'].values))

    # Near: use contracts from near_weights
    m_near = mkt[(mkt['selection_group'] == 'near_grid') & (mkt['contract'].isin(contracts))].copy()
    m_near.set_index(['quote_date', 'contract'], inplace=True)

    # Target rows per date (pick first far_target of requested type per date)
    m_far = mkt[(mkt['selection_group'] == 'far_target') & (mkt['option_type'] == target_type)].copy()
    m_far.set_index(['quote_date'], inplace=True)

    # underlying close per date
    underlying = mkt[['quote_date', 'underlying_close']].drop_duplicates(subset=['quote_date']).set_index('quote_date').sort_index()
    dates = sorted(underlying.index.unique())

    # precompute per-date close deltas for target and near (used for hedging)
    # target_delta_close: dict date->delta (long 1 unit)
    target_delta_close = {}
    if not m_far.empty:
        # m_far may have multiple rows per date; take first
        m_far_group = m_far.reset_index().groupby('quote_date')
        for d, g in m_far_group:
            row = g.iloc[0]
            target_delta_close[pd.to_datetime(d)] = float(row['delta']) if pd.notna(row.get('delta', np.nan)) else 0.0

    # near_delta_close: dict date-> portfolio delta (short value as used elsewhere)
    near_delta_close = {}
    m_near_reset = m_near.reset_index()
    if not m_near_reset.empty:
        for d in sorted(m_near_reset['quote_date'].unique()):
            grp = m_near_reset[m_near_reset['quote_date'] == d]
            s = 0.0
            for _, r in grp.iterrows():
                s += wmap.get(r['contract'], 0.0) * (float(r['delta']) if pd.notna(r.get('delta', np.nan)) else 0.0)
            # portfolio short delta = -sum(w * delta)
            near_delta_close[pd.to_datetime(d)] = -s

    # underlying close per date mapping
    underlying_close = {pd.to_datetime(idx): float(val) for idx, val in underlying['underlying_close'].items()}

    rows_near = []
    rows_target = []

    prev_date = None
    for d in dates:
        if prev_date is None:
            rows_near.append({'quote_date': d, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'vanna': 0.0, 'volga': 0.0, 'dtheta_dS': 0.0, 'theta': 0.0, 'residual': 0.0, 'total': 0.0})
            rows_target.append({'quote_date': d, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'vanna': 0.0, 'volga': 0.0, 'dtheta_dS': 0.0, 'theta': 0.0, 'residual': 0.0, 'total': 0.0})
            prev_date = d
            continue

        dS = float(underlying.loc[d, 'underlying_close'] - underlying.loc[prev_date, 'underlying_close']) if (pd.notna(underlying.loc[d, 'underlying_close']) and pd.notna(underlying.loc[prev_date, 'underlying_close'])) else 0.0
        dt = (d - prev_date).days

        # Near portfolio contributions
        delta_sum = 0.0
        gamma_sum = 0.0
        vega_sum = 0.0
        vanna_sum = 0.0
        volga_sum = 0.0
        dtheta_dS_sum = 0.0
        theta_sum = 0.0
        residual_sum = 0.0

        for c in contracts:
            key_prev = (prev_date, c)
            key_cur = (d, c)
            if key_prev not in m_near.index or key_cur not in m_near.index:
                continue
            rprev = m_near.loc[key_prev]
            rcur = m_near.loc[key_cur]
            try:
                S0 = float(rprev['underlying_close'])
            except Exception:
                S0 = float(underlying.loc[prev_date, 'underlying_close'])
            try:
                S1 = float(rcur['underlying_close'])
            except Exception:
                S1 = float(underlying.loc[d, 'underlying_close'])
            iv0 = float(rprev['implied_volatility']) if pd.notna(rprev.get('implied_volatility', np.nan)) else 0.0
            iv1 = float(rcur['implied_volatility']) if pd.notna(rcur.get('implied_volatility', np.nan)) else iv0
            K = float(rprev['strike'])
            exp = pd.to_datetime(rprev['expiration'])
            T0 = max((exp - prev_date).days / 365.0, 1/365.0)
            T1 = max((exp - d).days / 365.0, 0.0)

            # BS repricing and greeks at prev
            p0 = black_scholes_price(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            delta_prev, gamma_prev, vega_prev, theta_prev_year = black_scholes_greeks(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            theta_prev = theta_prev_year / 365.0
            vanna_prev, volga_prev, dtheta_dS_prev_year = compute_second_order_greeks(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            dtheta_dS_prev = dtheta_dS_prev_year / 365.0  # Scale to daily

            dIV = iv1 - iv0
            dS = S1 - S0

            delta_pc = delta_prev * dS
            gamma_pc = 0.5 * gamma_prev * dS ** 2
            vega_pc = vega_prev * dIV
            vanna_pc = vanna_prev * dS * dIV
            volga_pc = 0.5 * volga_prev * dIV ** 2
            dtheta_dS_pc = dtheta_dS_prev * dS  # Now in daily units
            theta_pc = theta_prev * dt

            # exact end price by BS
            p_end = black_scholes_price(S1, K, T1, 0.0, 0.0, iv1, rcur['option_type'])
            residual = p_end - (p0 + delta_pc + gamma_pc + vega_pc + vanna_pc + volga_pc + dtheta_dS_pc + theta_pc)

            w = wmap.get(c, 0.0)
            # near portfolio is short these weights
            sign = -1.0
            delta_sum += sign * w * delta_pc
            gamma_sum += sign * w * gamma_pc
            vega_sum += sign * w * vega_pc
            vanna_sum += sign * w * vanna_pc
            volga_sum += sign * w * volga_pc
            dtheta_dS_sum += sign * w * dtheta_dS_pc
            theta_sum += sign * w * theta_pc
            residual_sum += sign * w * residual

        # Hedge pnl handling for near portfolio
        if delta_hedge:
            if rebalance_daily:
                # on first date there's no pnl; otherwise use previous hedge size
                # compute prev and cur dates
                # find index of current date in dates
                idx = dates.index(d)
                if idx == 0:
                    hedge_pnl = 0.0
                    # initialize hedge size for next period
                    hedge_size = -near_delta_close.get(d, 0.0)
                    # store for next iteration by attaching to local var
                    prev_hedge_size_near = hedge_size
                else:
                    prev_d = dates[idx-1]
                    prev_hedge_size_near = -near_delta_close.get(prev_d, 0.0)
                    S_prev = underlying_close.get(prev_d, np.nan)
                    S_cur = underlying_close.get(d, np.nan)
                    dS = S_cur - S_prev if (not np.isnan(S_prev) and not np.isnan(S_cur)) else 0.0
                    hedge_pnl = prev_hedge_size_near * dS
                delta_sum += hedge_pnl
            else:
                # static hedge from start: hedge size = -near_delta_close[start_date]
                start = dates[0]
                hedge_size_start = -near_delta_close.get(start, 0.0)
                S_start = underlying_close.get(start, np.nan)
                S_cur = underlying_close.get(d, np.nan)
                dS = S_cur - S_start if (not np.isnan(S_start) and not np.isnan(S_cur)) else 0.0
                hedge_pnl = hedge_size_start * dS
                delta_sum += hedge_pnl

        total_near = delta_sum + gamma_sum + vega_sum + vanna_sum + volga_sum + dtheta_dS_sum + theta_sum + residual_sum
        rows_near.append({'quote_date': d, 'delta': delta_sum, 'gamma': gamma_sum, 'vega': vega_sum, 'vanna': vanna_sum, 'volga': volga_sum, 'dtheta_dS': dtheta_dS_sum, 'theta': theta_sum, 'residual': residual_sum, 'total': total_near})

        # Target contributions (pick far target row for date prev and cur)
        delta_sum = gamma_sum = vega_sum = vanna_sum = volga_sum = dtheta_dS_sum = theta_sum = residual_sum = 0.0
        if prev_date in m_far.index and d in m_far.index:
            rprev = m_far.loc[prev_date]
            rcur = m_far.loc[d]
            # if multiple entries (DataFrame), take first
            if isinstance(rprev, pd.DataFrame):
                rprev = rprev.iloc[0]
            if isinstance(rcur, pd.DataFrame):
                rcur = rcur.iloc[0]

            S0 = float(rprev['underlying_close']) if pd.notna(rprev.get('underlying_close', np.nan)) else float(underlying.loc[prev_date, 'underlying_close'])
            S1 = float(rcur['underlying_close']) if pd.notna(rcur.get('underlying_close', np.nan)) else float(underlying.loc[d, 'underlying_close'])
            iv0 = float(rprev['implied_volatility']) if pd.notna(rprev.get('implied_volatility', np.nan)) else 0.0
            iv1 = float(rcur['implied_volatility']) if pd.notna(rcur.get('implied_volatility', np.nan)) else iv0
            K = float(rprev['strike'])
            exp = pd.to_datetime(rprev['expiration'])
            T0 = max((exp - prev_date).days / 365.0, 1/365.0)
            T1 = max((exp - d).days / 365.0, 0.0)

            p0 = black_scholes_price(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            delta_prev, gamma_prev, vega_prev, theta_prev_year = black_scholes_greeks(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            theta_prev = theta_prev_year / 365.0
            vanna_prev, volga_prev, dtheta_dS_prev_year = compute_second_order_greeks(S0, K, T0, 0.0, 0.0, iv0, rprev['option_type'])
            dtheta_dS_prev = dtheta_dS_prev_year / 365.0  # Scale to daily

            dIV = iv1 - iv0
            dS = S1 - S0
            
            delta_pc = delta_prev * dS
            gamma_pc = 0.5 * gamma_prev * dS ** 2
            vega_pc = vega_prev * dIV
            vanna_pc = vanna_prev * dS * dIV
            volga_pc = 0.5 * volga_prev * dIV ** 2
            dtheta_dS_pc = dtheta_dS_prev * dS  # Now in daily units
            theta_pc = theta_prev * dt

            p_end = black_scholes_price(S1, K, T1, 0.0, 0.0, iv1, rcur['option_type'])
            residual = p_end - (p0 + delta_pc + gamma_pc + vega_pc + vanna_pc + volga_pc + dtheta_dS_pc + theta_pc)

            # target is long 1 unit
            delta_sum += 1.0 * delta_pc
            gamma_sum += 1.0 * gamma_pc
            vega_sum += 1.0 * vega_pc
            vanna_sum += 1.0 * vanna_pc
            volga_sum += 1.0 * volga_pc
            dtheta_dS_sum += 1.0 * dtheta_dS_pc
            theta_sum += 1.0 * theta_pc
            residual_sum += 1.0 * residual

        total_target = delta_sum + gamma_sum + vega_sum + vanna_sum + volga_sum + dtheta_dS_sum + theta_sum + residual_sum
        # add hedge pnl for target if requested
        if delta_hedge:
            if rebalance_daily:
                idx = dates.index(d)
                if idx == 0:
                    hedge_pnl_t = 0.0
                    prev_hedge_size_target = -target_delta_close.get(d, 0.0)
                else:
                    prev_d = dates[idx-1]
                    prev_hedge_size_target = -target_delta_close.get(prev_d, 0.0)
                    S_prev = underlying_close.get(prev_d, np.nan)
                    S_cur = underlying_close.get(d, np.nan)
                    dS = S_cur - S_prev if (not np.isnan(S_prev) and not np.isnan(S_cur)) else 0.0
                    hedge_pnl_t = prev_hedge_size_target * dS
                delta_sum += hedge_pnl_t
            else:
                start = dates[0]
                hedge_size_start_t = -target_delta_close.get(start, 0.0)
                S_start = underlying_close.get(start, np.nan)
                S_cur = underlying_close.get(d, np.nan)
                dS = S_cur - S_start if (not np.isnan(S_start) and not np.isnan(S_cur)) else 0.0
                hedge_pnl_t = hedge_size_start_t * dS
                delta_sum += hedge_pnl_t

        rows_target.append({'quote_date': d, 'delta': delta_sum, 'gamma': gamma_sum, 'vega': vega_sum, 'vanna': vanna_sum, 'volga': volga_sum, 'dtheta_dS': dtheta_dS_sum, 'theta': theta_sum, 'residual': residual_sum, 'total': total_target})

        prev_date = d

    pf_near = pd.DataFrame(rows_near).set_index('quote_date').fillna(0.0)
    pf_target = pd.DataFrame(rows_target).set_index('quote_date').fillna(0.0)
    return pf_near, pf_near.cumsum(), pf_target, pf_target.cumsum()


def plot_pnl_factors_cum(pf_cum: pd.DataFrame, out_dir: str, name: str = 'Near'):
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in ['delta', 'gamma', 'vega', 'vanna', 'volga', 'dtheta_dS', 'theta', 'residual']:
        if col in pf_cum.columns:
            ax.plot(pf_cum.index, pf_cum[col], label=col.capitalize(), marker='o')
    ax.set_title(f'Cumulative PnL Attribution by Factor ({name})')
    ax.set_ylabel('Cumulative PnL')
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    safe = name.lower().replace(' ', '_')
    out = os.path.join(out_dir, f'pnl_attribution_cumulative_{safe}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_pnl_factors_daily(pf: pd.DataFrame, out_dir: str, name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in ['delta', 'gamma', 'vega', 'vanna', 'volga', 'dtheta_dS', 'theta', 'residual']:
        if col in pf.columns:
            ax.plot(pf.index, pf[col], label=col.capitalize(), marker='o')
    ax.plot(pf.index, pf['total'], label='Net', linestyle='--', color='k', marker='o')
    ax.set_title(f'Daily PnL Attribution by Factor ({name})')
    ax.set_ylabel('Daily PnL')
    ax.legend(fontsize=10)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    out = os.path.join(out_dir, f'pnl_attribution_{name.lower().replace(" ", "_")}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


if __name__ == '__main__':
    main()
