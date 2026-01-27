"""
Gold & Silver Price Forecasting Model
ООО «Альянс Алтын» - Internal Forecasting System

Production-ready code for multi-model ensemble forecasting with:
- SARIMAX time series model
- Ridge/ElasticNet regression with macro factors
- XGBoost non-linear learning
- Weighted ensemble with scenario analysis
- Confidence intervals (80%, 95%)

Author: Data Science Team
Date: January 2026
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Data sources
import yfinance as yf
import pandas_datareader as pdr

# Time series & statistical
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# ML models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Utilities
from scipy import stats
import pickle
import json

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecasting_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configuration
CONFIG = {
    'start_date': '2005-01-01',
    'end_date': '2026-01-27',  # Current date
    'forecast_months': 60,      # 5 years
    'test_size_months': 24,     # Last 2 years for OOS testing
    'random_state': 42,
    'ensemble_weights': {
        'sarimax': 0.35,
        'ridge': 0.33,
        'xgboost': 0.32
    }
}

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. DATA DOWNLOAD & CLEANING
# ============================================================================

def download_data(start_date, end_date):
    """
    Download gold, silver, and macroeconomic indicators.
    Uses yfinance, FRED, and public data sources.
    """
    logger.info("Starting data download...")

    try:
        # 1. Precious metals (yfinance)
        logger.info("Downloading gold and silver prices...")
        gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
        silver = yf.download('SI=F', start=start_date, end=end_date, progress=False)

        # Use adjusted close or close price
        xau = gold['Adj Close'].fillna(gold['Close'])
        xag = silver['Adj Close'].fillna(silver['Close'])

        # 2. Macroeconomic indicators (FRED)
        logger.info("Downloading macroeconomic indicators...")

        fred_keys = {
            'FFR': 'FEDFUNDS',           # Fed Funds Rate
            'DGS10': 'DGS10',            # 10Y Treasury yield
            'DGS2': 'DGS2',              # 2Y Treasury yield
            'T10Y2Y': 'T10Y2Y',          # 10Y-2Y spread
            'CPIAUCSL': 'CPIAUCSL',      # CPI
            'PPICMM': 'PPICMM',          # PPI
            'M2SL': 'M2SL',              # M2 Money Supply
            'DFEDTARU': 'DFEDTARU',      # Fed Rate Upper Bound (recent)
        }

        macro_data = pd.DataFrame()
        for name, fred_code in fred_keys.items():
            try:
                data = pdr.get_data_fred(fred_code, start=start_date, end=end_date)
                macro_data[name] = data
            except Exception as e:
                logger.warning(f"Could not fetch {name} ({fred_code}): {e}")

        # 3. Equity & commodity indices (yfinance)
        logger.info("Downloading equity and commodity data...")

        tickers = {
            'SPX': '^GSPC',    # S&P 500
            'VIX': '^VIX',     # Volatility Index
            'DXY': 'DX-Y.NYB', # USD Index
            'CRUDE': 'CL=F',   # WTI Oil
            'HG': 'HG=F',      # Copper
        }

        equity_data = pd.DataFrame()
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                equity_data[name] = data['Adj Close'].fillna(data['Close'])
            except Exception as e:
                logger.warning(f"Could not fetch {name} ({ticker}): {e}")

        # Combine all data
        df = pd.DataFrame()
        df['XAU'] = xau
        df['XAG'] = xag
        df = df.join(macro_data)
        df = df.join(equity_data)

        # Convert to monthly data (last business day of month)
        df = df.resample('M').last()
        df = df.sort_index()

        logger.info(f"Downloaded data from {df.index[0].date()} to {df.index[-1].date()}")
        logger.info(f"Total months: {len(df)}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")

        return df

    except Exception as e:
        logger.error(f"Data download failed: {e}")
        raise

def clean_data(df):
    """
    Clean and interpolate missing values.
    """
    logger.info("Cleaning data...")

    # Forward fill for missing values (max 3 periods)
    df = df.fillna(method='ffill', limit=3)

    # Linear interpolation for remaining small gaps
    df = df.interpolate(method='linear', limit=2)

    # Drop rows with any remaining NaNs
    df = df.dropna()

    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING
# ============================================================================

def exploratory_analysis(df):
    """
    Perform EDA: descriptive stats, correlations, trend decomposition.
    """
    logger.info("Performing exploratory analysis...")

    # Basic statistics
    stats_df = df.describe().T
    logger.info(f"\nDescriptive Statistics:\n{stats_df}")

    # Correlation matrix
    corr = df.corr()
    logger.info(f"\nCorrelation with Gold (XAU):\n{corr['XAU'].sort_values(ascending=False)}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gold price time series
    axes[0, 0].plot(df.index, df['XAU'], linewidth=2, color='gold')
    axes[0, 0].set_title('Gold Price (XAU/USD)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('USD/oz')
    axes[0, 0].grid(True, alpha=0.3)

    # Silver price time series
    axes[0, 1].plot(df.index, df['XAG'], linewidth=2, color='silver')
    axes[0, 1].set_title('Silver Price (XAG/USD)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('USD/oz')
    axes[0, 1].grid(True, alpha=0.3)

    # Gold correlation heatmap
    top_corr = corr['XAU'].drop('XAU').abs().nlargest(8)
    corr_subset = df[top_corr.index.tolist() + ['XAU']].corr()
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
    axes[1, 0].set_title('Gold Correlations (Top Indicators)', fontsize=12, fontweight='bold')

    # Distribution
    axes[1, 1].hist(df['XAU'].pct_change().dropna(), bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('Gold Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Monthly Return')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_eda_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("EDA plot saved: 01_eda_analysis.png")
    plt.close()

    # Trend decomposition
    if len(df) > 24:
        decomposition = seasonal_decompose(df['XAU'], model='additive', period=12)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        decomposition.observed.plot(ax=axes[0], color='gold', linewidth=2)
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Gold Price Decomposition (Additive)', fontsize=12, fontweight='bold')

        decomposition.trend.plot(ax=axes[1], color='steelblue', linewidth=2)
        axes[1].set_ylabel('Trend')

        decomposition.seasonal.plot(ax=axes[2], color='green', linewidth=1)
        axes[2].set_ylabel('Seasonal')

        decomposition.resid.plot(ax=axes[3], color='red', linewidth=1)
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')

        plt.tight_layout()
        plt.savefig('02_decomposition.png', dpi=300, bbox_inches='tight')
        logger.info("Decomposition plot saved: 02_decomposition.png")
        plt.close()

    return corr

def feature_engineering(df):
    """
    Create lagged, momentum, and interaction features for ML models.
    """
    logger.info("Engineering features...")

    df_feat = df.copy()

    # Lagged features (1, 6, 12 months)
    for col in df.columns:
        df_feat[f'{col}_lag1'] = df_feat[col].shift(1)
        df_feat[f'{col}_lag6'] = df_feat[col].shift(6)
        df_feat[f'{col}_lag12'] = df_feat[col].shift(12)

    # Momentum features (rate of change)
    for col in ['FFR', 'DGS10', 'SPX', 'DXY', 'CRUDE']:
        if col in df.columns:
            df_feat[f'{col}_momentum_3m'] = df_feat[col].pct_change(3)
            df_feat[f'{col}_momentum_12m'] = df_feat[col].pct_change(12)
            df_feat[f'{col}_volatility'] = df_feat[col].rolling(12).std()

    # Real interest rate proxy
    if 'DGS10' in df.columns and 'CPIAUCSL' in df.columns:
        inflation_rate = df['CPIAUCSL'].pct_change(12)
        df_feat['real_rate_10y'] = df['DGS10'] - inflation_rate * 100

    # Curve inversion indicator
    if 'DGS10' in df.columns and 'DGS2' in df.columns:
        df_feat['curve_inversion'] = df['DGS10'] - df['DGS2']
        df_feat['curve_inverted'] = (df_feat['curve_inversion'] < 0).astype(int)

    # Crisis dummy (historical)
    df_feat['crisis'] = 0
    crisis_periods = [
        ('2008-08-01', '2009-03-31'),  # Financial crisis
        ('2020-02-01', '2020-04-30'),  # COVID
        ('2022-02-01', '2022-09-30'),  # Geopolitical
    ]
    for start, end in crisis_periods:
        mask = (df_feat.index >= start) & (df_feat.index <= end)
        df_feat.loc[mask, 'crisis'] = 1

    # Drop NaNs from lagged features
    df_feat = df_feat.dropna()

    logger.info(f"Feature engineering complete. Total features: {len(df_feat.columns)}")
    logger.info(f"Remaining observations: {len(df_feat)}")

    return df_feat

# ============================================================================
# 3. MODEL TRAINING: SARIMAX, RIDGE, XGBOOST
# ============================================================================

def train_sarimax(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog_cols=None):
    """
    Train SARIMAX model for gold price.
    """
    logger.info("Training SARIMAX model...")

    if exog_cols is None:
        exog_cols = ['DXY', 'FFR', 'VIX', 'CRUDE']

    # Prepare data: remove NaNs from exogenous variables
    train_data = df[['XAU'] + exog_cols].dropna()

    endog = train_data['XAU']
    exog = train_data[exog_cols]

    # Standardize exogenous variables for numerical stability
    scaler_exog = StandardScaler()
    exog_scaled = scaler_exog.fit_transform(exog)
    exog_scaled = pd.DataFrame(exog_scaled, index=exog.index, columns=exog.columns)

    try:
        model = SARIMAX(
            endog,
            exog=exog_scaled,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=200)
        logger.info(f"SARIMAX AIC: {results.aic:.2f}")
        logger.info(f"SARIMAX Summary:\n{results.summary()}")

        return results, scaler_exog
    except Exception as e:
        logger.error(f"SARIMAX training failed: {e}. Falling back to simpler model.")
        model = SARIMAX(endog, exog=None, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
        results = model.fit(disp=False, maxiter=100)
        return results, None

def train_ridge_regression(df, test_size_months=24):
    """
    Train Ridge regression with macro factors.
    """
    logger.info("Training Ridge regression model...")

    # Select features (remove XAG, XAU price, keep only numeric)
    feature_cols = [col for col in df.columns if col not in ['XAU', 'XAG'] and df[col].dtype in [np.float64, np.float32, np.int64]]

    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'XAU']

    # Train-test split
    split_idx = len(X) - test_size_months
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test = ridge.predict(X_test_scaled)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    logger.info(f"Ridge Training RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
    logger.info(f"Ridge Testing RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': ridge.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    logger.info(f"\nTop 10 Ridge Features:\n{feature_importance.head(10)}")

    return ridge, scaler, feature_cols, (rmse_train, rmse_test, r2_train, r2_test)

def train_xgboost(df, test_size_months=24):
    """
    Train XGBoost model for non-linear patterns.
    """
    logger.info("Training XGBoost model...")

    # Prepare features (same as Ridge)
    feature_cols = [col for col in df.columns if col not in ['XAU', 'XAG'] and df[col].dtype in [np.float64, np.float32, np.int64]]

    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'XAU']

    # Train-test split
    split_idx = len(X) - test_size_months
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=CONFIG['random_state'],
        verbosity=0
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    logger.info(f"XGBoost Training RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
    logger.info(f"XGBoost Testing RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop 10 XGBoost Features:\n{feature_importance.head(10)}")

    return xgb_model, feature_cols, (rmse_train, rmse_test, r2_train, r2_test)

# ============================================================================
# 4. FORECAST GENERATION WITH ENSEMBLE
# ============================================================================

def forecast_sarimax(model, scaler_exog, months_forward=60, exog_cols=None):
    """
    Generate SARIMAX forecasts with confidence intervals.
    """
    logger.info(f"Generating SARIMAX forecasts for {months_forward} months...")

    # For future exogenous variables, use last values (assumption: no change)
    if scaler_exog is not None and exog_cols is not None:
        # Get last observed exog values
        last_exog = model.model.exog[-1:] if model.model.exog is not None else None
        exog_future = np.tile(last_exog, (months_forward, 1)) if last_exog is not None else None
    else:
        exog_future = None

    try:
        forecast_result = model.get_forecast(steps=months_forward, exog=exog_future)
        forecast = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI

        return forecast, conf_int
    except Exception as e:
        logger.warning(f"SARIMAX forecast failed: {e}. Using fallback.")
        # Fallback: naive forecast with last value
        forecast = np.full(months_forward, model.model.endog[-1])
        std_error = np.std(model.resid) if hasattr(model, 'resid') else 100
        conf_int = np.column_stack([forecast - 1.96*std_error, forecast + 1.96*std_error])
        return forecast, conf_int

def forecast_regression(ridge, scaler, feature_cols, df_current, months_forward=60):
    """
    Generate Ridge regression forecasts.
    Uses last-value assumption for macro features.
    """
    logger.info(f"Generating Ridge forecasts for {months_forward} months...")

    # Get last observed features
    X_last = df_current[feature_cols].iloc[-1:].values
    X_future = np.tile(X_last, (months_forward, 1))
    X_future_scaled = scaler.transform(X_future)

    forecast = ridge.predict(X_future_scaled)

    # Confidence intervals based on residual std
    residuals = ridge.predict(scaler.transform(df_current[feature_cols])) - df_current['XAU'].values
    std_error = np.std(residuals)

    ci_95 = np.column_stack([
        forecast - 1.96 * std_error,
        forecast + 1.96 * std_error
    ])

    return forecast, ci_95

def forecast_xgboost(xgb_model, feature_cols, df_current, months_forward=60):
    """
    Generate XGBoost forecasts.
    """
    logger.info(f"Generating XGBoost forecasts for {months_forward} months...")

    # Get last observed features
    X_last = df_current[feature_cols].iloc[-1:].values
    X_future = np.tile(X_last, (months_forward, 1))

    forecast = xgb_model.predict(X_future)

    # Confidence intervals based on residual std
    residuals = xgb_model.predict(df_current[feature_cols]) - df_current['XAU'].values
    std_error = np.std(residuals)

    ci_95 = np.column_stack([
        forecast - 1.96 * std_error,
        forecast + 1.96 * std_error
    ])

    return forecast, ci_95

def ensemble_forecast(forecast_sarimax, forecast_ridge, forecast_xgboost, weights=None):
    """
    Combine forecasts using weighted ensemble.
    """
    if weights is None:
        weights = CONFIG['ensemble_weights']

    ensemble = (
            weights['sarimax'] * forecast_sarimax +
            weights['ridge'] * forecast_ridge +
            weights['xgboost'] * forecast_xgboost
    )

    return ensemble

# ============================================================================
# 5. SCENARIO ANALYSIS
# ============================================================================

def scenario_analysis(forecast_base):
    """
    Generate optimistic and pessimistic scenarios.
    """
    logger.info("Running scenario analysis...")

    # Scenarios represent different macro regimes
    # Optimistic: Crisis/disinflation shock (similar to 2008, 2020)
    # Pessimistic: Sustained high real rates (inflation surprise continues)

    # Base case: +2-3% annually = 12% over 5 years
    # Optimistic: +5% annually = 28% over 5 years (flight to safety)
    # Pessimistic: -1% annually = -5% over 5 years (no demand)

    months_range = len(forecast_base)

    scenario_optimistic = forecast_base.copy()
    scenario_pessimistic = forecast_base.copy()

    for i in range(months_range):
        # Optimistic: gradual increase (crisis scenario)
        months_elapsed = i
        annual_growth_opt = 0.05
        scenario_optimistic[i] = forecast_base[0] * (1 + annual_growth_opt) ** (months_elapsed / 12)

        # Pessimistic: decline (sustained deflation)
        annual_growth_pess = -0.01
        scenario_pessimistic[i] = forecast_base[0] * (1 + annual_growth_pess) ** (months_elapsed / 12)

    return scenario_optimistic, scenario_pessimistic

def confidence_intervals(forecast, residuals, confidence=0.95):
    """
    Calculate confidence intervals from residual distribution.
    """
    # Assume normal distribution of residuals
    std_error = np.std(residuals)
    z_score = stats.norm.ppf((1 + confidence) / 2)

    ci_low = forecast - z_score * std_error
    ci_high = forecast + z_score * std_error

    return ci_low, ci_high

# ============================================================================
# 6. POST-PROCESSING & PLAUSIBILITY CHECKS
# ============================================================================

def post_process_forecast(forecast, scenario_opt, scenario_pess, df_original):
    """
    Apply post-processing: smoothing, outlier removal, economic plausibility.
    """
    logger.info("Post-processing forecasts...")

    # Store original for logging
    forecast_orig = forecast.copy()

    # 1. Statistical smoothing (near-term only)
    from scipy.ndimage import uniform_filter1d
    # Smooth first 12 months, gradually reduce smoothing
    for i in range(12):
        weight_smooth = 1.0 - (i / 12) * 0.7  # 100% → 30% smoothing
        window = 3
        smoothed = uniform_filter1d(forecast[:i+1], size=window, mode='nearest')[-1]
        forecast[i] = forecast[i] * (1 - weight_smooth) + smoothed * weight_smooth

    # 2. Economic plausibility constraints
    min_gold = 700    # Production cost floor
    max_gold = 5000   # Unrealistic without systemic collapse

    forecast = np.clip(forecast, min_gold, max_gold)
    scenario_opt = np.clip(scenario_opt, min_gold, max_gold)
    scenario_pess = np.clip(scenario_pess, min_gold, max_gold)

    # 3. Quarterly change cap (±15% absent major events)
    for i in range(3, len(forecast)):
        quarterly_change = (forecast[i] - forecast[i-3]) / forecast[i-3]
        if abs(quarterly_change) > 0.15:
            # Moderate extreme changes
            forecast[i] = forecast[i-3] * (1 + np.sign(quarterly_change) * 0.12)

    # 4. Geopolitical adjustment (add risk premium)
    # Assuming no major escalation, but add buffer
    geopolitical_buffer = 30  # $30/oz buffer
    forecast = forecast + geopolitical_buffer
    scenario_opt = scenario_opt + geopolitical_buffer * 0.5  # Less buffer in opt case

    # 5. Log adjustments
    logger.info(f"Original forecast range: ${forecast_orig.min():.0f}-${forecast_orig.max():.0f}")
    logger.info(f"Post-processed forecast range: ${forecast.min():.0f}-${forecast.max():.0f}")
    logger.info("Adjustments applied: smoothing (12M), plausibility clipping, quarterly caps, geopolitical buffer")

    return forecast, scenario_opt, scenario_pess

# ============================================================================
# 7. EXPORT & VISUALIZATION
# ============================================================================

def export_forecasts(forecast_dates, forecast_base, forecast_opt, forecast_pess,
                     ci80_low, ci80_high, ci95_low, ci95_high):
    """
    Export forecasts to CSV in required format.
    """
    logger.info("Exporting forecasts to CSV...")

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'gold_base': forecast_base,
        'gold_optimistic': forecast_opt,
        'gold_pessimistic': forecast_pess,
        'CI80_low': ci80_low,
        'CI80_high': ci80_high,
        'CI95_low': ci95_low,
        'CI95_high': ci95_high
    })

    # Add aggregated forecasts
    agg_forecasts = []
    for horizon, months in [('1Q', 3), ('1Y', 12), ('3Y', 36), ('5Y', 60)]:
        if months <= len(forecast_base):
            agg_forecasts.append({
                'date': f'Avg_{horizon}',
                'gold_base': forecast_base[:months].mean(),
                'gold_optimistic': forecast_opt[:months].mean(),
                'gold_pessimistic': forecast_pess[:months].mean(),
                'CI80_low': ci80_low[:months].mean(),
                'CI80_high': ci80_high[:months].mean(),
                'CI95_low': ci95_low[:months].mean(),
                'CI95_high': ci95_high[:months].mean(),
            })

    agg_df = pd.DataFrame(agg_forecasts)
    forecast_df = pd.concat([forecast_df, agg_df], ignore_index=True)

    forecast_df.to_csv('gold_silver_forecasts.csv', index=False)
    logger.info("Forecasts exported: gold_silver_forecasts.csv")
    logger.info(f"\n{forecast_df.head(10)}")

    return forecast_df

def plot_forecasts(df_original, forecast_dates, forecast_base, forecast_opt, forecast_pess,
                   ci95_low, ci95_high):
    """
    Plot historical prices and forecasts with scenarios and CI.
    """
    logger.info("Creating forecast visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Historical data
    ax = axes[0]
    ax.plot(df_original.index, df_original['XAU'], label='Historical Gold Price',
            color='gold', linewidth=2.5, zorder=3)

    # Forecasts
    ax.plot(forecast_dates, forecast_base, label='Base Case',
            color='steelblue', linewidth=2, linestyle='--', zorder=2)
    ax.fill_between(forecast_dates, forecast_opt, forecast_pess,
                    alpha=0.15, color='green', label='Optimistic to Pessimistic')
    ax.fill_between(forecast_dates, ci95_low, ci95_high,
                    alpha=0.1, color='red', label='95% Confidence Interval')

    # Formatting
    ax.set_title('Gold Price Forecast (5-Year Horizon)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD/oz)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Silver (secondary axis)
    ax2 = axes[1]
    ax2.plot(df_original.index, df_original['XAG'], label='Historical Silver Price',
             color='silver', linewidth=2.5, zorder=3)
    ax2.set_title('Silver Price (Informational)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD/oz)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('03_forecast_base_case.png', dpi=300, bbox_inches='tight')
    logger.info("Forecast plot saved: 03_forecast_base_case.png")
    plt.close()

    # Scenario comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(forecast_dates, forecast_base, label='Base Case',
            color='steelblue', linewidth=2.5, marker='o', markersize=3)
    ax.plot(forecast_dates, forecast_opt, label='Optimistic (Crisis/QE)',
            color='green', linewidth=2, linestyle='--', alpha=0.8)
    ax.plot(forecast_dates, forecast_pess, label='Pessimistic (Deflation)',
            color='red', linewidth=2, linestyle='--', alpha=0.8)

    # Shaded regions
    ax.fill_between(forecast_dates, forecast_opt, forecast_base,
                    alpha=0.1, color='green')
    ax.fill_between(forecast_dates, forecast_base, forecast_pess,
                    alpha=0.1, color='red')

    ax.set_title('Gold Price Scenario Analysis', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD/oz)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=500)

    plt.tight_layout()
    plt.savefig('04_scenario_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Scenario plot saved: 04_scenario_analysis.png")
    plt.close()

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute complete forecasting pipeline.
    """
    logger.info("="*80)
    logger.info("GOLD PRICE FORECASTING MODEL - FULL EXECUTION")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*80)

    try:
        # Step 1: Data
        logger.info("\n[STEP 1] DATA DOWNLOAD & PREPARATION")
        df_raw = download_data(CONFIG['start_date'], CONFIG['end_date'])
        df_clean = clean_data(df_raw)

        # Step 2: EDA
        logger.info("\n[STEP 2] EXPLORATORY ANALYSIS")
        corr_matrix = exploratory_analysis(df_clean)

        # Step 3: Feature Engineering
        logger.info("\n[STEP 3] FEATURE ENGINEERING")
        df_features = feature_engineering(df_clean)

        # Step 4: Model Training
        logger.info("\n[STEP 4] MODEL TRAINING")

        # SARIMAX
        sarimax_results, scaler_exog = train_sarimax(df_clean)

        # Ridge
        ridge_model, ridge_scaler, ridge_cols, ridge_metrics = train_ridge_regression(df_features)

        # XGBoost
        xgb_model, xgb_cols, xgb_metrics = train_xgboost(df_features)

        # Step 5: Forecasting
        logger.info("\n[STEP 5] FORECAST GENERATION")

        # Generate individual forecasts
        forecast_sarimax_vals, ci_sarimax = forecast_sarimax(
            sarimax_results, scaler_exog, CONFIG['forecast_months'], ['DXY', 'FFR', 'VIX', 'CRUDE']
        )

        forecast_ridge_vals, ci_ridge = forecast_regression(
            ridge_model, ridge_scaler, ridge_cols, df_features, CONFIG['forecast_months']
        )

        forecast_xgb_vals, ci_xgb = forecast_xgboost(
            xgb_model, xgb_cols, df_features, CONFIG['forecast_months']
        )

        # Ensemble
        forecast_ensemble = ensemble_forecast(
            forecast_sarimax_vals, forecast_ridge_vals, forecast_xgb_vals
        )

        logger.info(f"Ensemble forecast created (weights: {CONFIG['ensemble_weights']})")
        logger.info(f"Forecast range: ${forecast_ensemble.min():.0f} - ${forecast_ensemble.max():.0f}")

        # Step 6: Scenarios
        logger.info("\n[STEP 6] SCENARIO ANALYSIS")
        scenario_opt, scenario_pess = scenario_analysis(forecast_ensemble.copy())

        # Confidence intervals
        all_residuals = np.concatenate([
            sarimax_results.resid.values if hasattr(sarimax_results, 'resid') else np.array([]),
            ridge_model.predict(ridge_scaler.transform(df_features[ridge_cols])) - df_features['XAU'].values
        ])
        ci80_low, ci80_high = confidence_intervals(forecast_ensemble, all_residuals, confidence=0.80)
        ci95_low, ci95_high = confidence_intervals(forecast_ensemble, all_residuals, confidence=0.95)

        # Step 7: Post-processing
        logger.info("\n[STEP 7] POST-PROCESSING & ADJUSTMENTS")
        forecast_final, scenario_opt_final, scenario_pess_final = post_process_forecast(
            forecast_ensemble, scenario_opt, scenario_pess, df_features
        )

        # Step 8: Export
        logger.info("\n[STEP 8] EXPORT & VISUALIZATION")

        # Forecast dates (monthly from next month)
        last_date = df_clean.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=CONFIG['forecast_months'], freq='M')

        # Export CSV
        forecast_table = export_forecasts(
            forecast_dates, forecast_final, scenario_opt_final, scenario_pess_final,
            ci80_low, ci80_high, ci95_low, ci95_high
        )

        # Plots
        plot_forecasts(df_clean, forecast_dates, forecast_final, scenario_opt_final,
                       scenario_pess_final, ci95_low, ci95_high)

        # Save models & scalers
        logger.info("\n[STEP 9] MODEL PERSISTENCE")
        models = {
            'sarimax': sarimax_results,
            'ridge': ridge_model,
            'ridge_scaler': ridge_scaler,
            'ridge_cols': ridge_cols,
            'xgboost': xgb_model,
            'xgb_cols': xgb_cols,
            'config': CONFIG
        }

        with open('trained_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        logger.info("Models saved: trained_models.pkl")

        # Summary statistics
        logger.info("\n[SUMMARY STATISTICS]")
        logger.info(f"Forecast 1Y average: ${forecast_final[:12].mean():.0f}")
        logger.info(f"Forecast 3Y average: ${forecast_final[:36].mean():.0f}")
        logger.info(f"Forecast 5Y average: ${forecast_final[:60].mean():.0f}")
        logger.info(f"Optimistic 5Y: ${scenario_opt_final[:60].mean():.0f}")
        logger.info(f"Pessimistic 5Y: ${scenario_pess_final[:60].mean():.0f}")
        logger.info(f"95% CI width (1Y): ${(ci95_high[:12].mean() - ci95_low[:12].mean()):.0f}")

        logger.info("\n" + "="*80)
        logger.info("EXECUTION COMPLETE")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*80)

        return {
            'forecast': forecast_table,
            'models': models,
            'df_clean': df_clean,
            'dates': forecast_dates
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    results = main()
