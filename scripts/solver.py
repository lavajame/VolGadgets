import numpy as np
import pandas as pd
from io import StringIO

# 1. Setup near-grid market matrix data
market_data_csv = """strike,option_type,mid,delta,gamma
1440,call,72.000,0.9766,0.0015
1440,put,5.000,-0.1382,0.0032
1450,call,63.050,0.9563,0.0025
1450,put,6.100,-0.1666,0.0037
1470,call,46.050,0.8431,0.0055
1470,put,9.050,-0.2398,0.0050
1480,call,38.100,0.7783,0.0068
1480,put,11.100,-0.2871,0.0056
1500,call,23.900,0.6203,0.0089
1500,put,16.950,-0.4057,0.0068
1520,call,12.750,0.4322,0.0095
1520,put,25.750,-0.5477,0.0073
1530,call,8.550,0.3337,0.0092
1530,put,31.650,-0.6191,0.0070
1550,call,3.250,0.1666,0.0066
1550,put,46.200,-0.7396,0.0057
1570,call,1.075,0.0666,0.0035
1570,put,64.050,-0.8089,0.0043
1580,call,0.600,0.0400,0.0023
1580,put,73.600,-0.8293,0.0037
1600,call,0.200,0.0122,0.0008
1600,put,93.150,-0.8565,0.0028"""

df = pd.read_csv(StringIO(market_data_csv))

# Build target linear constraint arrays
A = np.vstack([df['mid'].values, df['delta'].values, df['gamma'].values])
b = np.array([11.25, 0.3192, 0.0065])

# Hyperparameter for regularization intensity
alpha = 1e-3

# -------------------------------------------------------------
# APPROACH A: Standard Uniform Ridge Regression (Dual Form)
# -------------------------------------------------------------
# w = A^T * (A * A^T + alpha * I)^-1 * b
identity_3x3 = np.eye(3)
w_uniform = np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + alpha * identity_3x3)) @ b

# -------------------------------------------------------------
# APPROACH B: Scale-Invariant Weighted Ridge Regression
# -------------------------------------------------------------
# Scale the penalty matrix V relative to the intrinsic magnitudes of A*A^T
diag_AA_T = np.diag(np.dot(A, A.T))
V_scaled = alpha * np.diag(diag_AA_T)
w_scaled = np.dot(A.T, np.linalg.inv(np.dot(A, A.T) + V_scaled)) @ b

# -------------------------------------------------------------
# Verification Printout
# -------------------------------------------------------------
def print_metrics(label, weights):
    print(f"--- {label} ---")
    print(f"Premium: {np.dot(weights, df['mid'].values):.4f} (Target: {b[0]})")
    print(f"Delta:   {np.dot(weights, df['delta'].values):.4f} (Target: {b[1]})")
    print(f"Gamma:   {np.dot(weights, df['gamma'].values):.4f} (Target: {b[2]})")
    print(f"L2 Norm of weights: {np.sum(weights**2):.4f}")
    print(f"Max allocation size: {np.max(np.abs(weights)):.4f}\n")

print("==================================================")
print(f"         RIDGE REGRESSION SOLVER REPORT           ")
print(f"                 (Alpha = {alpha})                ")
print("==================================================\n")

print_metrics("Uniform Ridge (Distorts Gamma Match)", w_uniform)
print_metrics("Scale-Invariant Weighted Ridge", w_scaled)

# Add the robust weights to the frame to inspect allocation distribution
df['naive_ridge_wgts'] = w_uniform
df['smart_ridge_wgts'] = w_scaled
print("==================================================")
print("       SCALE-INVARIANT RIDGE WEIGHT VECTOR        ")
print("==================================================")
print(df[['strike', 'option_type', 'smart_ridge_wgts', 'naive_ridge_wgts', 'mid']].round(4).to_string(index=False))