# Market Regime Analysis Using UMAP + GMM

This project explores **market regime discovery** using a combination of unsupervised learning techniques — **UMAP** for dimensionality reduction and **Gaussian Mixture Models (GMM)** for clustering. The aim is to identify distinct market conditions based on macroeconomic, volatility, and sentiment indicators, and to test whether these regimes hold predictive value for trading strategies.

## 🔍 Project Goals

- Uncover **latent market regimes** from raw financial data
- Analyze each regime's return characteristics, feature profile, and stability
- Evaluate clustering configurations using:
  - **Sharpe ratio** (return vs. risk)
  - **BIC** (model fit penalty)
  - **Entropy** (confidence in assignments)
  - **Stability** (robustness across random seeds)
- Build **simple regime-aware trading strategies** to validate the usefulness of regimes in a practical setting

## 🧠 Techniques Used

- **UMAP** (Uniform Manifold Approximation and Projection) for reducing high-dimensional features
- **Gaussian Mixture Models** for soft clustering of market states
- **Adjusted Rand Index**, entropy, and Sharpe for evaluation
- **Backtesting engine** with cash-based simulation, transaction costs, and slippage
- Visual analysis including:
  - Regime return profiles
  - Feature importance per regime (via Random Forest)
  - KS-statistic heatmaps to confirm distinct distributions

## 📈 Key Insights

- Regimes are statistically and visually distinct
- Some regimes exhibit consistently **higher Sharpe ratios**, while others signal elevated risk
- Even simple strategies — such as **only holding SPY during favorable regimes** — significantly outperform Buy & Hold, even after realistic costs
- Feature importance shows clear differentiation between regimes (volatility, sentiment, macro exposure)

## 📁 Files

- `market-regime-analysis.ipynb`: Main notebook containing the full pipeline, from data prep to backtesting
- `market-regime-analysis.html`: Exported notebook with interactive visualizations
- `models\gmm|scaler|umap.pkl`: Serialized models used in data pipeline
- `umap_gmm.csv`: Cached grid search results for UMAP+GMM configs (optional)
- `regime-features.json`: Cached feature importance results from RandomForest exploration
- `regime-observations.json`: LLM-interpretation of regimes after providing it with `regime-features.json`
- Plots and regime summaries saved within notebook outputs

## 📌 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually ensure the following libraries are available:

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `umap-learn`
- `plotly`
- `tqdm`
- `jinja2` (needed for displaying rich tables in Jupyter)

## 🚀 Getting Started

1. Open `market-regime-analysis.ipynb` in Jupyter or VSCode
2. Run all cells in order
3. Explore:
   - Regime visualizations
   - Feature breakdowns
   - Strategy backtests
   - Heatmaps and 3D plots of latent structure
