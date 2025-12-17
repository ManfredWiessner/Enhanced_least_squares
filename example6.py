import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numdifftools as nd
import sys

# ==============================================================================
# AUTHORSHIP AND ENVIRONMENT INFORMATION
# ==============================================================================
# Author: Manfred Wiessner: https://scholar.google.at/citations?user=-SxDth0AAAAJ&hl=en
# Used versions:
# Python: 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]
# NumPy: 1.26.4
# Pandas: 2.2.2
# SciPy: 1.14.0
# numdifftools: 0.9.42

# ==============================================================================
# 1. RAW DATA & INITIAL PARAMETERS (P0) - SCALED Y-DATA
# ==============================================================================
# !!! IMPORTANT SCALING NOTE !!!
# The y_data has been scaled down by dividing by 100.
# Initial parameters P0 (A1, A2, s0, s1) and bounds have been adjusted accordingly
# to ensure the optimization starts within the correct range for the new scale.
# The Dislocation Density scaling factor for x-axis related parameters (x0, dx) remains.

# Column A: x-data (53 data points)
x_data = np.array(
    [22, 22, 27.5, 42.5, 50, 50, 62.5, 87.5, 100, 100, 112.5, 137.5, 150, 150, 162.5, 187.5, 200, 200, 212.5, 237.5,
     250, 250, 262.5, 287.5, 300, 300, 312.5, 337.5, 350, 350, 362.5, 387.5, 400, 400, 412.5, 437.5, 450, 450, 462.5,
     487.5, 500, 500, 512.5, 537.5, 550, 550, 562.5, 587.5, 600, 600, 612.5, 637.5, 650])

# Column B: y-data (measured values) - DIVIDED BY scaling_factor
scaling_factor = 0.01  # Scaling factor for y_data (division by 100)
y_data_original = np.array(
    [780.736, 694.997, 634.522, 710.078, 652.855, 732.355, 732.608, 660.579, 640.615, 706.121, 721.219, 729.574,
     701.986, 721.649, 629.576, 641.398, 660.333, 660.576, 647.118, 672.643, 716.937, 646.079, 635.233, 618.923,
     620.085, 621.435, 609.958, 630.906, 634.038, 621.224, 647.268, 621.815, 644.31, 699.29, 628.957, 642.193,
     624.068, 610.783, 607.593, 605.209, 578.407, 517.969, 570.978, 513.906, 291.448, 216.813, 175.107, 113.919,
     73.087, 72.024, 47.622, 45.960, 43.852])
y_data = y_data_original * scaling_factor  # Scale y_data by dividing by 100

# Initial parameters P0 = [A1, A2, x0, dx, s0, s1] - SCALED P0
P0_original = np.array([659.1169792, 40.53185252, 543.1219922, 19.19853954, 4.120469757, 0.088031614])

# Scale P0 according to the scaling_factor:
# A1, A2, s0: Scaled by 100 (approx. 6.59, 0.40, 0.0412)
# x0, dx, s1: Unchanged (x0, dx are x-axis related; s1 is a relative parameter)
P0 = np.array([
    P0_original[0] * scaling_factor,  # A1
    P0_original[1] * scaling_factor,  # A2
    P0_original[2],                   # x0
    P0_original[3],                   # dx
    P0_original[4] * scaling_factor,  # s0
    P0_original[5]                    # s1
])

N_data = len(x_data)  # Number of data points
labels = ['A1', 'A2', 'x0', 'dx', 's0', 's1']  # Parameter labels for output

# Bounds for the L-BFGS-B optimizer - SCALED BOUNDS
# Each parameter's bounds are adjusted according to the scaling_factor
bounds = [
    (100 * scaling_factor, 1000 * scaling_factor),  # A1: (1, 10)
    (1 * scaling_factor, 200 * scaling_factor),    # A2: (0.01, 2)
    (100, 600),                                    # x0: (Unchanged)
    (1, 100),                                      # dx: (Unchanged)
    (0.1 * scaling_factor, 10 * scaling_factor),   # s0: (0.001, 0.1)
    (0.001, 10)                                    # s1: (Unchanged)
]

# ==============================================================================
# 2. OBJECTIVE FUNCTION S DEFINITION (Negative Log-Likelihood)
# ==============================================================================

def objective_function_s_wrapper(P):
    """
    Core function for numerical derivatives (Numdifftools).
    Computes the negative log-likelihood (S) for the sigmoid model.

    Parameters:
    -----------
    P : array-like
        Array of parameters [A1, A2, x0, dx, s0, s1].

    Returns:
    --------
    S_value : float
        The value of the objective function S.
    """
    A1, A2, x0, dx, s0, s1 = P

    # Sigmoid Model (y_model)
    y_model = A2 + (A1 - A2) / (1 + np.exp((x_data - x0) / dx))

    # Sigma Model (Variance Function)
    sigma = s0 + s1 * y_model

    # Avoid division by zero or log of zero
    if np.any(sigma <= 1e-9) or dx <= 1e-9:
        return 1e15  # Return a large value to avoid invalid parameter sets

    # Objective Function S (Negative Log-Likelihood)
    ln_sigma = np.log(sigma)
    weighted_sq_error_half = ((y_data - y_model) ** 2) / (2 * sigma ** 2)
    S_value = 2 * np.sum(ln_sigma) + 2 * np.sum(weighted_sq_error_half)

    return S_value

def objective_function_s(P, x_data, y_data):
    """
    Wrapper for the objective function, compatible with scipy.optimize.minimize.

    Parameters:
    -----------
    P : array-like
        Array of parameters [A1, A2, x0, dx, s0, s1].
    x_data : array-like
        Array of x-values.
    y_data : array-like
        Array of y-values.

    Returns:
    --------
    S_value : float
        The value of the objective function S.
    """
    return objective_function_s_wrapper(P)

# ==============================================================================
# 3. MINIMIZATION OPTIMIZATION (L-BFGS-B)
# ==============================================================================

print("--- STARTING MINIMIZATION OF S with SCALED Y-DATA ---")
print(f"Initial Parameters P0 (Scaled): {P0}")

# Minimize the objective function using L-BFGS-B method
result = minimize(
    objective_function_s,
    P0,
    args=(x_data, y_data),
    method='L-BFGS-B',
    bounds=bounds,
    options={'ftol': 1e-9, 'disp': False}
)

P_fit = result.x  # Optimized parameters
S_min = result.fun  # Minimum value of the objective function

print(f"Status: {result.message}")
print(f"Minimum S value found: {S_min:.8f}\n")

# --- Optimized Parameter Output ---
print("Optimized Fit Parameters (P_fit):")
for label, value in zip(labels, P_fit):
    print(f"{label}: {value:.8f}")

print("\n" + "=" * 50)

# ==============================================================================
# 4. PRECISE ERROR ANALYSIS (HESSIAN, COVARIANCE, CORRELATION)
# ==============================================================================

print("### PRECISE ERROR ANALYSIS (Numdifftools) ###")

try:
    # 1. Numerical Calculation of the Hessian Matrix H (Curvature)
    H_calc = nd.Hessian(objective_function_s_wrapper)
    H_num = H_calc(P_fit)

    # 2. Covariance Matrix V = H^-1 (Estimates parameter uncertainty)
    V_num = np.linalg.inv(H_num)

    print("\nNumerical Hessian Matrix H (at S_min):")
    df_hess = pd.DataFrame(H_num, index=labels, columns=labels)
    print(df_hess.to_string(float_format='%.3e'))

    print("\nCovariance Matrix V (H^-1):")
    df_cov = pd.DataFrame(V_num, index=labels, columns=labels)
    print(df_cov.to_string(float_format='%.3e'))

    # 3. Standard Errors (Square root of the diagonal of V)
    std_err_num = np.sqrt(np.diag(V_num))
    print("\nStandard Errors of the Parameters:")
    for label, error in zip(labels, std_err_num):
        print(f"Error({label}): {error:.8f}")

    # 4. CORRELATION MATRIX R
    # R = V_ij / (std_i * std_j)
    std_product_matrix = np.outer(std_err_num, std_err_num)
    R_matrix = V_num / std_product_matrix

    print("\nCorrelation Matrix R:")
    df_corr = pd.DataFrame(R_matrix, index=labels, columns=labels)
    print(df_corr.to_string(float_format='%.4f'))

except ImportError:
    print("\nERROR: 'numdifftools' could not be imported. Please install it with 'pip install numdifftools'.")
except np.linalg.LinAlgError:
    print("\nERROR: The Hessian matrix is singular (non-invertible). Correlation matrix cannot be calculated.")
except Exception as e:
    print(f"\nAn error occurred during calculation: {e}")

# ==============================================================================
# 5. FINAL METRICS (LOG-LIKELIHOOD N2)
# ==============================================================================

N = N_data
term_konstante = (N / 2) * np.log(1 / (2 * np.pi))
ln_likelihood_python = term_konstante - 0.5 * S_min

print("\n--- FINAL METRICS ---")
print(f"S (Objective Function, minimized): {S_min:.8f}")
print(f"ln_P (Log-Likelihood from P, Maximum): {ln_likelihood_python:.8f}")

#OUTPUT:
# --- STARTING MINIMIZATION OF S with SCALED Y-DATA ---
# Initial Parameters P0 (Scaled): [6.59116979e+00 4.05318525e-01 5.43121992e+02 1.91985395e+01
#  4.12046976e-02 8.80316140e-02]
# Status: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
# Minimum S value found: -38.72893039
#
# Optimized Fit Parameters (P_fit):
# A1: 6.59434779
# A2: 0.40485439
# x0: 543.15386483
# dx: 19.11040551
# s0: 0.03926693
# s1: 0.07776550
#
# ==================================================
# ### PRECISE ERROR ANALYSIS (Numdifftools) ###
#
# Numerical Hessian Matrix H (at S_min):
#           A1        A2        x0        dx        s0        s1
# A1 3.059e+02 7.611e+01 5.015e+00 1.096e+00 6.829e+01 3.099e+02
# A2 7.611e+01 1.616e+03 1.964e+01 4.737e+01 1.853e+02 2.154e+02
# x0 5.015e+00 1.964e+01 8.839e-01 8.203e-01 7.346e+00 4.620e+00
# dx 1.096e+00 4.737e+01 8.203e-01 2.195e+00 1.362e+01 9.486e-01
# s0 6.829e+01 1.853e+02 7.346e+00 1.362e+01 2.397e+03 7.191e+03
# s1 3.099e+02 2.154e+02 4.620e+00 9.486e-01 7.191e+03 2.718e+04
#
# Covariance Matrix V (H^-1):
#           A1         A2         x0         dx         s0         s1
# A1  3.806e-03 -1.722e-04 -2.990e-02  1.178e-02  2.020e-04 -9.081e-05
# A2 -1.722e-04  1.929e-03 -6.487e-03 -4.373e-02  7.566e-04 -2.108e-04
# x0 -2.990e-02 -6.487e-03  2.010e+00 -5.388e-01 -9.473e-03  2.575e-03
# dx  1.178e-02 -4.373e-02 -5.388e-01  1.775e+00 -2.962e-02  8.078e-03
# s0  2.020e-04  7.566e-04 -9.473e-03 -2.962e-02  2.748e-03 -7.326e-04
# s1 -9.081e-05 -2.108e-04  2.575e-03  8.078e-03 -7.326e-04  2.326e-04
#
# Standard Errors of the Parameters:
# Error(A1): 0.06169493
# Error(A2): 0.04392264
# Error(x0): 1.41788109
# Error(dx): 1.33234383
# Error(s0): 0.05242177
# Error(s1): 0.01525072
#
# Correlation Matrix R:
#         A1      A2      x0      dx      s0      s1
# A1  1.0000 -0.0635 -0.3418  0.1433  0.0625 -0.0965
# A2 -0.0635  1.0000 -0.1042 -0.7472  0.3286 -0.3148
# x0 -0.3418 -0.1042  1.0000 -0.2852 -0.1274  0.1191
# dx  0.1433 -0.7472 -0.2852  1.0000 -0.4241  0.3976
# s0  0.0625  0.3286 -0.1274 -0.4241  1.0000 -0.9164
# s1 -0.0965 -0.3148  0.1191  0.3976 -0.9164  1.0000
#
# --- FINAL METRICS ---
# S (Objective Function, minimized): -38.72893039
# ln_P (Log-Likelihood from P, Maximum): -29.33927707
#
# Process finished with exit code 0