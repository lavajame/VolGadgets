import math


def std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    # S: spot, K: strike, T: time to expiry in years, r: rate, q: dividend yield, sigma: vol
    if T <= 0 or sigma <= 0:
        # payoff at expiry or degenerate
        if option_type.lower().startswith('c'):
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = std_norm_cdf(d1)
    Nd2 = std_norm_cdf(d2)
    Nmd1 = std_norm_cdf(-d1)
    Nmd2 = std_norm_cdf(-d2)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type.lower().startswith('c'):
        price = S * disc_q * Nd1 - K * disc_r * Nd2
    else:
        price = K * disc_r * Nmd2 - S * disc_q * Nmd1

    return price


def black_scholes_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str):
    # returns (delta, gamma, vega, theta)
    if T <= 0 or sigma <= 0:
        # At expiry, greeks are degenerate; approximate as 0
        return 0.0, 0.0, 0.0, 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type.lower().startswith('c'):
        delta = disc_q * std_norm_cdf(d1)
    else:
        delta = -disc_q * std_norm_cdf(-d1)

    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf_d1 * math.sqrt(T)
    # Theta per year; convert to per day when needed by dividing by 365
    theta = - (S * disc_q * pdf_d1 * sigma) / (2 * math.sqrt(T)) - (r * K * disc_r * std_norm_cdf(d2) if option_type.lower().startswith('c') else -r * K * disc_r * std_norm_cdf(-d2)) + q * S * disc_q * (std_norm_cdf(d1) if option_type.lower().startswith('c') else -std_norm_cdf(-d1))

    return float(delta), float(gamma), float(vega), float(theta)
