# Predicting Loan Defaults for Banca Massiccia

## Project Overview
In collaboration with Banca Massiccia, a fictitious leading Italian bank, this project aimed at enhancing the bank's loan underwriting process. Utilizing machine learning, I developed a model to predict the one-year Probability of Default (PD) for prospective borrowers, thereby enabling risk-based pricing and more informed loan decisions.

## Data Mining Solutions

### Approach
The approach involved training a logistic regression model on historical bank transaction data. The focus was on key financial indicators to forecast the probability of default for new borrowers.

### Techniques
- **Feature Engineering:** Extracted meaningful indicators such as debt ratio, leverage, and cash return on assets.
- **Modeling:** Employed logistic regression for default prediction.
- **Evaluation:** Applied finance-specific evaluation methods, including walk-forward testing.

### Outcome
The model provides a realistic prediction of default probability, assisting the bank in determining suitable interest rates and underwriting fees.

## Data Understanding and Preparation

### Problem Formulation
I applied a business-context-aware approach, considering financial factors such as profitability, leverage, and liquidity in our model.

### Data Imputation
Handled missing values by replacing them with related financial variables. For example, a variable called `roe` had some missing values, which were replaced by `profit` / `total equity`.

### Engineered Features
- Liquidity: Time Interest Earned, Current Ratio
- Size: Total Assets
- Debt Coverage: Leverage Ratio, Debt Ratio
- Profitability: Net Profit Margin, Cash Return on Assets
- Activity: Asset Turnover, Net Working Capital
- Leverage: (Long term debt + Short term debt) / Total Assets

## Model Evaluation and Interpretation

### Walk-Forward Analysis
Implemented walk-forward analysis for a realistic simulation of financial industry behavior, ensuring robust model performance.

### Calibration
Calibrated the model using a non-linear curve mapping, ensuring accurate default probability predictions.

### Interpreting Coefficients and P-value
* Verified the coefficients' signs aligned with our financial understanding, confirming the model's validity.
* Ensured that variables with p-values under 0.05 were selected as final variables.

## Conclusion and Future Work

