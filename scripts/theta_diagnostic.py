import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs_pricer import black_scholes_greeks


def safe_log(x):
    return math.log(x) if x > 0 else float('nan')


def run_diagnostic(subfolder: str):
    base = os.path.join('backtests', subfolder)
    mkt_path = os.path.join(base, 'mkt_data.csv')
    if not os.path.exists(mkt_path):
        print('Market CSV not found:', mkt_path)
        return

    mkt = pd.read_csv(mkt_path, parse_dates=['quote_date', 'expiration'], dtype={'contract': str})

    rows = []
    for _, row in mkt.iterrows():
        try:
            if pd.isna(row.get('implied_volatility')) or pd.isna(row.get('theta')):
                continue
            S = float(row['underlying_close'])
            K = float(row['strike'])
            iv = float(row['implied_volatility'])
            mkt_theta = float(row['theta'])
            q = 0.0
            r = 0.0
            T_days = max((pd.to_datetime(row['expiration']) - pd.to_datetime(row['quote_date'])).days, 1)
            T = max(T_days / 365.0, 1/365.0)

            # BS theta per year
            _, _, _, bs_theta_year = black_scholes_greeks(S, K, T, r, q, iv, row['option_type'])
            bs_theta_day = bs_theta_year / 365.0

            # require same sign and non-zero for multiplicative comparison
            if bs_theta_day == 0 or mkt_theta == 0 or (bs_theta_day * mkt_theta) <= 0:
                continue

            ratio = bs_theta_day / mkt_theta
            log_ratio = math.log(abs(bs_theta_day)) - math.log(abs(mkt_theta))

            rows.append({
                'quote_date': row['quote_date'],
                'contract': row['contract'],
                'S': S,
                'K': K,
                'moneyness': S / K if K != 0 else np.nan,
                'T': T,
                'iv': iv,
                'mkt_theta': mkt_theta,
                'bs_theta_day': bs_theta_day,
                'ratio': ratio,
                'log_ratio': log_ratio,
                'option_type': row['option_type']
            })
        except Exception:
            continue

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        print('No usable rows for diagnostic')
        return

    out_csv = os.path.join(base, 'theta_diagnostic.csv')
    df.to_csv(out_csv, index=False)
    print('Wrote diagnostic data to', out_csv)

    # Simple correlations
    print('\nCorrelations with ratio:')
    print(df[['ratio', 'S', 'moneyness', 'T', 'iv']].corr()['ratio'].sort_values(ascending=False))

    # Fit log-linear model: log_ratio ~ log(S) + log(T) + log(iv) + log(moneyness)
    X_cols = []
    df['logS'] = np.log(df['S'].replace(0, np.nan))
    df['logT'] = np.log(df['T'].replace(0, np.nan))
    df['logiv'] = np.log(df['iv'].replace(0, np.nan))
    df['logm'] = np.log(df['moneyness'].replace(0, np.nan).abs())

    for c in ['logS', 'logT', 'logiv', 'logm']:
        if df[c].notna().any():
            X_cols.append(c)

    X = df[X_cols].values
    X = np.column_stack([np.ones(len(X)), X])
    y = df['log_ratio'].values

    # remove rows with nan
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    Xm = X[mask]
    ym = y[mask]

    if len(ym) < 5:
        print('Not enough rows for regression')
        return

    coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    ypred = Xm.dot(coef)
    ssr = np.sum((ypred - ym) ** 2)
    sst = np.sum((ym - ym.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else 0.0

    print('\nLog-linear regression results (dependent = log(bs_theta_day) - log(mkt_theta))')
    names = ['const'] + X_cols
    for n, c in zip(names, coef):
        print(f'{n:>8}: {c: .4f}')
    print(f'R^2: {r2:.4f}   n={len(ym)}')

    # scatter plots
    figs = []
    for fld in ['S', 'moneyness', 'T', 'iv']:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[fld], df['ratio'], alpha=0.6)
        ax.set_xscale('log') if (df[fld] > 0).all() else None
        ax.set_yscale('log')
        ax.set_xlabel(fld)
        ax.set_ylabel('bs_theta_day / market_theta')
        ax.grid(True)
        outp = os.path.join(base, f'theta_ratio_vs_{fld}.png')
        fig.savefig(outp, dpi=120, bbox_inches='tight')
        plt.close(fig)
        figs.append(outp)

    print('\nSaved scatter plots:')
    for p in figs:
        print(p)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Theta diagnostics vs Black-Scholes')
    p.add_argument('subfolder')
    args = p.parse_args()
    run_diagnostic(args.subfolder)
