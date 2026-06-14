#!/usr/bin/env python
"""
Pre-processing script: Repair missing ITM prices using vol surface consistency.

For each strike where one side (ITM) has zero/missing bid but the other side (OTM) 
has valid quotes, extract the IV and regenerate prices for the missing side.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
from datetime import datetime

def black_scholes_call(S, K, T, r, sigma):
    """BS call price"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """BS put price"""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def is_itm(S, K, opt_type):
    """Check if option is ITM"""
    if opt_type.lower() == 'call':
        return S > K
    else:
        return S < K

def has_bad_quote(bid, ask, mid):
    """Check if quote is problematic"""
    if bid is None or ask is None or mid is None:
        return True
    if pd.isna(bid) or pd.isna(ask) or pd.isna(mid):
        return True
    if bid == 0 or ask == 0 or mid == 0:
        return True
    return False

def repair_mkt_data(input_csv, output_csv):
    """
    Repair missing ITM prices using OTM implied volatility.
    """
    
    df = pd.read_csv(input_csv, dtype={'contract': str})
    df['quote_date'] = pd.to_datetime(df['quote_date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    
    print("="*100)
    print(f"MARKET DATA PRE-PROCESSING: Vol Surface Repair")
    print("="*100)
    print(f"Input: {input_csv}")
    print(f"Records: {len(df)}")
    
    original_df = df.copy()
    repaired_count = 0
    
    # Group by date and strike to identify call/put pairs
    for quote_date in df['quote_date'].unique():
        date_df = df[df['quote_date'] == quote_date]
        
        # Skip underlying
        date_df = date_df[date_df['selection_group'].isin(['near_grid', 'far_target'])]
        
        # Group by strike
        for strike in date_df['strike'].dropna().unique():
            strike_data = date_df[date_df['strike'] == strike].copy()
            
            if len(strike_data) < 2:
                continue  # Need both call and put
            
            call_rows = strike_data[strike_data['option_type'] == 'call']
            put_rows = strike_data[strike_data['option_type'] == 'put']
            
            if len(call_rows) == 0 or len(put_rows) == 0:
                continue
            
            S = strike_data.iloc[0]['underlying_close']
            T = (strike_data.iloc[0]['expiration'] - quote_date).days / 365.0
            
            if T <= 0:
                continue
            
            call_row = call_rows.iloc[0]
            put_row = put_rows.iloc[0]
            
            # Determine which is ITM
            call_is_itm = is_itm(S, strike, 'call')
            put_is_itm = is_itm(S, strike, 'put')
            
            # Get IV from OTM side to repair ITM side
            call_bad = has_bad_quote(call_row['bid'], call_row['ask'], call_row['mid'])
            put_bad = has_bad_quote(put_row['bid'], put_row['ask'], put_row['mid'])
            
            # Case 1: Call is ITM & bad, Put is OTM & good → use put IV to fix call
            if call_is_itm and call_bad and not put_is_itm and not put_bad:
                put_iv = put_row['implied_volatility']
                
                if pd.notna(put_iv) and put_iv > 0:
                    # Regenerate call prices
                    call_price = black_scholes_call(S, strike, T, 0.0, put_iv)
                    call_mid = call_price
                    call_bid = call_mid * 0.99  # Assume 1% bid-ask
                    call_ask = call_mid * 1.01
                    
                    # Update the dataframe
                    idx = call_row.name
                    df.loc[idx, 'bid'] = call_bid
                    df.loc[idx, 'ask'] = call_ask
                    df.loc[idx, 'mid'] = call_mid
                    df.loc[idx, 'implied_volatility'] = put_iv
                    
                    repaired_count += 1
                    print(f"[Repaired] CALL K={strike} on {quote_date.date()}: IV={put_iv:.4f} -> mid={call_mid:.2f}")
            
            # Case 2: Put is ITM & bad, Call is OTM & good → use call IV to fix put
            elif put_is_itm and put_bad and not call_is_itm and not call_bad:
                call_iv = call_row['implied_volatility']
                
                if pd.notna(call_iv) and call_iv > 0:
                    # Regenerate put prices
                    put_price = black_scholes_put(S, strike, T, 0.0, call_iv)
                    put_mid = put_price
                    put_bid = put_mid * 0.99
                    put_ask = put_mid * 1.01
                    
                    # Update the dataframe
                    idx = put_row.name
                    df.loc[idx, 'bid'] = put_bid
                    df.loc[idx, 'ask'] = put_ask
                    df.loc[idx, 'mid'] = put_mid
                    df.loc[idx, 'implied_volatility'] = call_iv
                    
                    repaired_count += 1
                    print(f"[Repaired] PUT K={strike} on {quote_date.date()}: IV={call_iv:.4f} -> mid={put_mid:.2f}")
    
    print(f"\n{repaired_count} bad quotes repaired using vol surface")
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"\nOutput: {output_csv}")
    print(f"Records: {len(df)}")
    
    return df

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python preprocess_vol_repair.py <input.csv> [output.csv]")
        print()
        print("Example:")
        print("  python preprocess_vol_repair.py backtests/130103_SPX/mkt_data.csv backtests/130103_SPX/clean_mkt_data.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else input_csv.replace('.csv', '_clean.csv')
    
    repair_mkt_data(input_csv, output_csv)
