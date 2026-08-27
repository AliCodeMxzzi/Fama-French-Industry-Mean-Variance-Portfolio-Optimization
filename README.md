# Fama-French Industry Mean-Variance Portfolio Optimization

Out-of-sample mean-variance portfolio construction across the 17 Fama-French value-weighted industry portfolios, using a rolling/expanding Fama-French 5-factor forecasting model for expected returns and covariances, benchmarked against a naive 1/N equal-weighted strategy.

## Overview

This project simulates a real-time investor allocating monthly across 17 industry portfolios from 1975:01 to 2026:02 (613 out-of-sample months). At each month *t*, expected excess returns and the covariance matrix are forecast using **only information available through month t−1**, then used to solve a mean-variance expected-utility maximization problem. Performance is compared against a simple equal-weighted (1/N) benchmark using annualized mean, volatility, Sharpe ratio, and certainty-equivalent return (CER).

## Data

| File | Description |
|---|---|
| `17_Industry_Portfolios.csv` | Monthly value-weighted returns for 17 Fama-French industry portfolios (Ken French Data Library) |
| `F-F_Research_Data_5_Factors_2x3.csv` | Monthly Fama-French 5-factor returns (Mkt-RF, SMB, HML, RMW, CMA) and risk-free rate |

Full sample: 1963:07–2026:02 (751 months). In-sample window (1963:07–1974:12, 138 months) initializes the estimation; out-of-sample evaluation runs 1975:01–2026:02 (613 months).

## Methodology

1. **Factor moments (expanding window):** At each month *t*, estimate the factor mean vector μ̂f and covariance matrix Σ̂f using all factor data available through *t−1*.
2. **Alpha/beta/idiosyncratic risk (rolling 120-month window):** Run a multifactor OLS regression per industry on the trailing 120 months to obtain intercepts (α̂), factor loadings (B̂), and residual variances (Σ̂ε).
3. **Forecast construction:**
   \( \hat\mu = \hat\alpha + \hat B'\hat\mu_f \), \( \hat\Sigma = \hat B'\hat\Sigma_f\hat B + \hat\Sigma_\varepsilon \)
4. **Portfolio optimization:** Solve
   \( \max_w \; r_f + w'\hat\mu - \tfrac{1}{2}\gamma w'\hat\Sigma w \)
   subject to \( 0 \le w_i \le 0.25 \) for each of the 17 industries, with risk aversion γ = 5. Unallocated weight sits in the risk-free asset.
5. **Benchmark:** Equal-weighted portfolio with \( w_i = 1/17 \) each month.

## Key Results (1975:01–2026:02, out-of-sample)

| Portfolio | Mean (%) | Volatility (%) | Sharpe Ratio |
|---|---|---|---|
| Benchmark (1/N) | 9.49 | 16.27 | 0.58 |
| MV Optimal | 6.76 | 12.71 | 0.53 |

- Annualized CER (γ = 5): MV Optimal = 2.73%, Benchmark = 2.88%, gain = **−0.15%**.
- The MV strategy cuts annualized volatility by 3.56 points versus 1/N but gives up some mean return; on a utility-adjusted basis the two are nearly indistinguishable.
- Consistent with the literature: multifactor forecasting models tend to add value mainly through better risk control rather than superior return prediction.

## Repository Structure

```
.
├── industry_mean_variance_oos.py         # Main analysis script
├── 17_Industry_Portfolios.csv            # Industry portfolio returns data
├── F-F_Research_Data_5_Factors_2x3.csv   # Fama-French 5-factor data
├── industry_mean_variance_oos-write-up.pdf   # Full write-up (PDF)
├── industry_mean_variance_oos-write-up.docx  # Full write-up (Word)
├── requirements.txt
└── README.md
```

## Usage

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python industry_mean_variance_oos.py
```

The script prints three summary tables (industry excess returns, factor returns, and OOS portfolio performance) plus the annualized CER comparison.

## Requirements

- Python 3.9+
- pandas, numpy, scipy

## Data Source

Ken French Data Library: [17 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [Fama-French 5 Factors (2x3)](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Author

Prepared as part of the UCLA Master of Financial Engineering coursework.
