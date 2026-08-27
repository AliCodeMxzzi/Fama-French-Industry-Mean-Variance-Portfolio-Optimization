import pandas as pd
import numpy as np
from scipy.optimize import minimize

MONTHLY_INDUSTRY_VW = 1208 - 13 + 1  # lines 13-1208: monthly value-weighted industry portfolios
MONTHLY_FACTORS = 757 - 6 + 1  # lines 6-757: monthly factors

# Read and Load the data for industry portfolios and factors
# Rename the first column to "date"
df_monthly_industry_vw = pd.read_csv('17_Industry_Portfolios.csv', skiprows=11, nrows=MONTHLY_INDUSTRY_VW) # line 12 is the header
df_monthly_industry_vw = df_monthly_industry_vw.rename(columns={df_monthly_industry_vw.columns[0]: "date"})

df_monthly_factors = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv", skiprows=4, nrows=MONTHLY_FACTORS) # line 5 is the header
df_monthly_factors = df_monthly_factors.rename(columns={df_monthly_factors.columns[0]: "date"})


# TABLE 1: Annualized summary statistics: 17 industry portfolios' excess returns, 1963:07–2026:02
# extract the data from 1963:07 to 2026:02 and reset the index
df_monthly_industry_vw_1963_2026 = df_monthly_industry_vw[(df_monthly_industry_vw["date"] >= 196307) & (df_monthly_industry_vw["date"] <= 202602)]
df_monthly_industry_vw_1963_2026 = df_monthly_industry_vw_1963_2026.reset_index(drop=True)
industries = list(df_monthly_industry_vw_1963_2026.columns[1:]) # list of industry portfolios
industry_portfolio_excess_returns = df_monthly_industry_vw_1963_2026[industries].sub(df_monthly_factors["RF"], axis=0) # subtract the risk-free rate from the industry portfolios

# Monthly summary statistics for industry portfolios' excess returns
industry_portfolio_mean = industry_portfolio_excess_returns.mean()
industry_portfolio_std = industry_portfolio_excess_returns.std()
industry_portfolio_sharpe_ratio = industry_portfolio_mean / industry_portfolio_std

# Annualized summary statistics for industry portfolios' excess returns
industry_portfolio_annualized_mean = industry_portfolio_mean * 12
industry_portfolio_annualized_std = industry_portfolio_std * np.sqrt(12)
industry_portfolio_annualized_sharpe_ratio = industry_portfolio_sharpe_ratio * np.sqrt(12)

# industry summary statistics table
industry_summary_statistics = pd.DataFrame({
    "Mean(%)": industry_portfolio_annualized_mean.round(2),
    "Volatility(%)": industry_portfolio_annualized_std.round(2),
    "Sharpe Ratio": industry_portfolio_annualized_sharpe_ratio.round(2),
}).rename_axis("Industry").reset_index()

print("Table 1: Annualized summary statistics: 17 industry portfolios' excess returns, 1963:07-2026:02")
print(industry_summary_statistics)


# TABLE 2: Annualized summary statistics: Factor returns, 1963:07–2026:02
factors = list(df_monthly_factors.columns[1:6]) # list of factors
df_monthly_factors_only = df_monthly_factors[factors] # extract the factors from the dataframe

# Monthly summary statistics for factors
FF_factor_mean = df_monthly_factors_only.mean()
FF_factor_std = df_monthly_factors_only.std()
FF_factor_sharpe_ratio = FF_factor_mean / FF_factor_std

# Annualized summary statistics for factors
FF_factor_annualized_mean = FF_factor_mean * 12
FF_factor_annualized_std = FF_factor_std * np.sqrt(12)
FF_factor_annualized_sharpe_ratio = FF_factor_sharpe_ratio * np.sqrt(12)

# factor summary statistics table
FF_factor_summary_statistics = pd.DataFrame({
    "Mean(%)": FF_factor_annualized_mean.round(2),
    "Volatility(%)": FF_factor_annualized_std.round(2),
    "Sharpe Ratio": FF_factor_annualized_sharpe_ratio.round(2),
}).rename_axis("Factor").reset_index()

print("\nTable 2: Annualized summary statistics: Factor returns, 1963:07-2026:02")
print(FF_factor_summary_statistics)


# TABLE 3: Annualized summary statistics: portfolio excess returns, 1975:01–2026:02
# merge the industry portfolios and factors on the date column and reset the index
df_panel = df_monthly_industry_vw_1963_2026.merge(df_monthly_factors, on="date", how="inner").reset_index(drop=True)
all_dates = df_panel["date"].values # 1963:07 → 2026:02
rf_full = df_panel["RF"].values # risk-free rate

factor_returns_full  = df_panel[factors] # (T x 5) # extract the factors from the dataframe
industry_excess_full = df_panel[industries].sub(df_panel["RF"], axis=0) # (T x 17) # subtract the risk-free rate from the industry portfolios

oos_start_idx = np.where(all_dates == 197501)[0][0] # first OOS month index - 138
T_total = len(all_dates) # total number of months - 752
ROLLING_WINDOW = 120 # rolling window sizes
gamma = 5.0 # risk aversion coefficient

oos_dates = all_dates[oos_start_idx:] # 1975:01 → 2026:02
n_oos = len(oos_dates) # number of OOS months - 614
n_ind = len(industries) # number of industries - 17
n_factors = len(factors) # number of factors - 5

# initialize the arrays to store the results
mu_hat_store    = np.full((n_oos, n_ind), np.nan)         # (n_oos x 17) # expected excess returns
Sigma_hat_store = np.full((n_oos, n_ind, n_ind), np.nan)  # (n_oos x 17 x 17) # covariance matrix
weights_store = np.full((n_oos, n_ind), np.nan)  # (n_oos x 17) # weights
realized_excess_returns_store = np.full(n_oos, np.nan)  # (n_oos,) # realized excess returns
realized_total_returns_store  = np.full(n_oos, np.nan)  # (n_oos,) # realized total returns
ew_excess_returns_store = np.full(n_oos, np.nan)  # (n_oos,) # equal-weighted excess returns
ew_total_returns_store  = np.full(n_oos, np.nan)  # (n_oos,) # equal-weighted total returns

# loop through the OOS months
for i, date in enumerate(oos_dates): 
    # starts with 197501
    t = oos_start_idx + i

    avail_factors = factor_returns_full[:t] # (t,5) # extract the factors from the dataframe up to the current month
    avail_excess = industry_excess_full[:t] # (t,17) # extract the industry excess returns from the dataframe up to the current month

    # ── 1. EXPANDING WINDOW: μ̂f and Σ̂f ─────────────────────────────────────
    mu_hat_f = avail_factors.mean(axis=0)  # (5,) # mean of the factors
    Sigma_hat_f = np.cov(avail_factors, rowvar=False) # (5,5) # covariance matrix of the factors

    # ── 2. ROLLING 120-MONTH WINDOW: α̂, B̂, Σ̂ε ──────────────────────────────
    roll_factors = avail_factors[-ROLLING_WINDOW:] # (120,5) # extract the last 120 months of factors
    roll_excess = avail_excess[-ROLLING_WINDOW:] # (120,17) # extract the last 120 months of industry excess returns
    T_roll = len(roll_factors) # 120

    # Design matrix X: add intercept column of 1s → shape (T_roll x 6)
    X = np.column_stack([np.ones(T_roll), roll_factors])  # (120 x 6)
    # OLS: coefficients for all 17 industries at once
    coeffs, _, _, _ = np.linalg.lstsq(X, roll_excess, rcond=None)

    # coeffs shape: (6 x 17) — row 0 = alphas, rows 1-5 = betas
    alpha_hat = coeffs[0]   # (17,)   — intercepts
    B_hat = coeffs[1:]      # (5x17)  — factor loadings

    # Residuals and disturbance variances for Σ̂ε
    fitted = X @ coeffs                      # (120 x 17)
    resid = roll_excess - fitted             # (120 x 17)

    # Unbiased OLS variance: divide by (T - 6) degrees of freedom
    sigma_eps_sq = (resid ** 2).sum(axis=0) / (T_roll - 6)   # (17,)
    Sigma_eps    = np.diag(sigma_eps_sq)                      # (17 x 17) diagonal

    # ── 3. COMPUTE μ̂ AND Σ̂ ─────────────────────────────────────────────────
    # μ̂ = α̂ + B̂' μ̂f
    mu_hat    = alpha_hat + B_hat.T @ mu_hat_f          # (17,)
    mu_hat_dec = mu_hat / 100
    # Σ̂ = B̂' Σ̂f B̂ + Σ̂ε
    Sigma_hat = B_hat.T @ Sigma_hat_f @ B_hat + Sigma_eps  # (17 x 17)
    Sigma_hat_dec = Sigma_hat / 10000

    mu_hat_store[i] = mu_hat_dec
    Sigma_hat_store[i] = Sigma_hat_dec

    # ── 4. EXPECTED UTILITY MAXIMIZATION ────────────────────────────────────
    # max  r_f + w'μ̂ - 0.5γ w'Σ̂w
    # s.t. 0 ≤ w_i ≤ 0.25  for i = 1,...,17
    #
    # rf at time t-1 (last available rf before forecast month)
    rf_t = rf_full[t - 1]

    # Objective: NEGATIVE of utility (scipy minimizes, we want to maximize)
    # maximizing the utility function is the same as minimizing the negative of the utility function
    def neg_utility(w):
        return -(rf_t + w @ mu_hat_dec - 0.5 * gamma * w @ Sigma_hat_dec @ w)

    # Constraints: 0 ≤ w_i ≤ 0.25 for each of the 17 industries
    bounds = [(0.0, 0.25)] * n_ind

    # Initial guess: equal weight across all industries
    w0 = np.ones(n_ind) / n_ind * 0.25

    result = minimize(neg_utility, w0, method='SLSQP', bounds=bounds, options={'ftol': 1e-12})

    # Store results
    weights_store[i] = result.x

    # ── 5. REALIZED PORTFOLIO RETURNS ────────────────────────────────────────
    # Actual industry excess returns for the NEXT month (month t, the forecast month)
    actual_excess_t = industry_excess_full.iloc[t]   # (17,)

    # Realized portfolio excess return = w' * actual_excess_t
    realized_excess = weights_store[i] @ actual_excess_t                 # scalar

    # Realized total return = excess return + rf of that month
    rf_next = rf_full[t]                                                 # rf at forecast month t
    realized_total = realized_excess + rf_next                           # scalar

    realized_excess_returns_store[i] = realized_excess
    realized_total_returns_store[i]  = realized_total

    # ── 6. EQUAL-WEIGHTED (1/N) BENCHMARK PORTFOLIO ──────────────────────────
    w_ew = np.ones(n_ind) / n_ind                        # 1/N weights (17,)

    # Realized EW portfolio excess return = w_ew' * actual_excess_t
    ew_realized_excess = w_ew @ actual_excess_t          # scalar

    # Realized EW total return = excess return + rf of that month
    ew_realized_total  = ew_realized_excess + rf_next    # scalar

    ew_excess_returns_store[i] = ew_realized_excess
    ew_total_returns_store[i]  = ew_realized_total

# Monthly summary statistics for mean-variance optimal portfolio returns
mv_mean = realized_excess_returns_store.mean()
mv_std = realized_excess_returns_store.std(ddof=1)
mv_sharpe = mv_mean / mv_std

# Monthly summary statistics for equal-weighted portfolio returns
ew_mean = ew_excess_returns_store.mean()
ew_std = ew_excess_returns_store.std(ddof=1)
ew_sharpe = ew_mean / ew_std

# Annualized summary statistics for mean-variance optimal portfolio returns
mv_annualized_mean = mv_mean * 12
mv_annualized_std = mv_std * np.sqrt(12)
mv_annualized_sharpe = mv_sharpe * np.sqrt(12)

# Annualized summary statistics for equal-weighted portfolio returns
ew_annualized_mean = ew_mean * 12
ew_annualized_std = ew_std * np.sqrt(12)
ew_annualized_sharpe = ew_sharpe * np.sqrt(12)

# out-of-sample mean-variance optimal portfolio and equal-weighted portfolio summary statistics table
mv_ew_summary_statistics = pd.DataFrame({
    "Mean(%)": [ew_annualized_mean.round(2), mv_annualized_mean.round(2)],
    "Volatility(%)": [ew_annualized_std.round(2), mv_annualized_std.round(2)],
    "Sharpe Ratio": [ew_annualized_sharpe.round(2), mv_annualized_sharpe.round(2)],
}, index=["Benchmark (1/N)", "MV Optimal"])

print("\nTable 3: Annualized summary statistics: portfolio excess returns, 1975:01-2026:02")
print(mv_ew_summary_statistics)

# Out-of-sample annualized certainty equivalent return (CER): U = E[r] - (γ/2) Var[r]
# Industry returns are in percentage points (FF convention). Equivalent annual CER in %:
#   12*μ% - (γ/2)*12*Var%/100, since Var(decimal) = Var%/10000 and we report in %.
mv_var_oos = np.var(realized_excess_returns_store, ddof=1)
ew_var_oos = np.var(ew_excess_returns_store, ddof=1)
cer_mv_annualized = 12 * mv_mean - 0.5 * gamma * 12 * mv_var_oos / 100.0
cer_ew_annualized = 12 * ew_mean - 0.5 * gamma * 12 * ew_var_oos / 100.0
cer_gain_annualized = cer_mv_annualized - cer_ew_annualized

# CER values are in annual % (same scale as Table 3 means: monthly returns are in FF %).
print("\nOut-of-sample annualized CER (gamma = {:.1f}):".format(gamma))
print("  CER for MV optimal:   {:.4f}%".format(cer_mv_annualized))
print("  CER for Benchmark 1/N: {:.4f}%".format(cer_ew_annualized))
print("  Annualized CER gain (MV vs 1/N): {:.4f}%".format(cer_gain_annualized))