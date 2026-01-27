import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. КОНФИГУРАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ==========================================
class CommoditiesForecaster:
    def __init__(self, target_ticker='GC=F', start_date='2005-01-01'):
        self.target_ticker = target_ticker
        self.start_date = start_date
        self.models = {}
        self.scalers = {}
        self.data = None
        self.residuals = None

    def get_data(self):
        print("📥 Загрузка макроэкономических данных...")
        # Тикеры: Золото, Серебро, S&P500, Доходность 10Y, DXY, Нефть, VIX
        tickers = {
            'Gold': 'GC=F',
            'Silver': 'SI=F',
            'SP500': '^GSPC',
            '10Y_Yield': '^TNX',
            'DXY': 'DX-Y.NYB',
            'Oil': 'CL=F',
            'VIX': '^VIX'
        }

        df = yf.download(list(tickers.values()), start=self.start_date)['Adj Close']
        df.columns = list(tickers.keys())

        # Ресемплинг до среднемесячных значений (убираем шум, фокусируемся на трендах)
        df_monthly = df.resample('M').mean()

        # Заполнение пропусков (forward fill), удаление оставшихся NaN
        df_monthly = df_monthly.ffill().dropna()
        self.data = df_monthly
        return df_monthly

    # ==========================================
    # 2. ГЕНЕРАЦИЯ ПРИЗНАКОВ (FEATURE ENGINEERING)
    # ==========================================
    def engineer_features(self, df):
        data = df.copy()

        # Макро-трансформации
        # Простой прокси реальной ставки: Доходность - 2% (таргет инфляции)
        data['Real_Rates_Proxy'] = data['10Y_Yield'] - 2.0
        data['Gold_Returns'] = data['Gold'].pct_change()

        # Лаговые признаки (Критически важны для временных рядов)
        for lag in [1, 3, 6, 12]:
            data[f'Gold_Lag_{lag}'] = data['Gold'].shift(lag)
            data[f'DXY_Lag_{lag}'] = data['DXY'].shift(lag)
            data[f'Yield_Lag_{lag}'] = data['10Y_Yield'].shift(lag)

        # Технические индикаторы
        data['MA_12'] = data['Gold'].rolling(window=12).mean()
        data['Volatility_12'] = data['Gold'].rolling(window=12).std()

        # Удаляем NaN, возникшие из-за лагов
        data = data.dropna()
        return data

    # ==========================================
    # 3. ОБУЧЕНИЕ МОДЕЛЕЙ (АНСАМБЛЬ)
    # ==========================================
    def train_ensemble(self, df):
        print("⚙️ Обучение ансамбля моделей...")

        # Определение целевой переменной и признаков
        target = 'Gold'
        features = [c for c in df.columns if c not in ['Gold', 'Silver', 'Gold_Returns']]

        X = df[features]
        y = df[target]

        # Разделение (Обучаем на прошлом, тестируем на последних 24 месяцах)
        split_idx = int(len(df) * 0.9)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # --- Модель 1: ElasticNet (Регрессия) ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        enet.fit(X_train_sc, y_train)

        # --- Модель 2: Random Forest (ML для нелинейности) ---
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)

        # --- Модель 3: SARIMAX (Временной ряд) ---
        # Простая ARIMA на цене (порядок p,d,q выбран для устойчивости)
        sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0,0,0,0))
        sarima_res = sarima.fit(disp=False)

        # Сохраняем модели
        self.models['ElasticNet'] = enet
        self.models['RandomForest'] = rf
        self.models['SARIMA'] = sarima_res
        self.scalers['X'] = scaler
        self.features = features

        # Оценка ансамбля на тесте для получения ошибки (Sigma) для доверительных интервалов
        pred_enet = enet.predict(X_test_sc)
        pred_rf = rf.predict(X_test)
        # Прим: Прогноз SARIMA требует динамического добавления, здесь упрощено
        pred_sarima = sarima_res.forecast(steps=len(X_test))

        # Взвешенное среднее (40% ML, 40% TS, 20% Regression)
        ensemble_pred = (0.2 * pred_enet) + (0.4 * pred_rf) + (0.4 * pred_sarima.values)

        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        self.sigma = rmse # Сохраняем ошибку для расчета интервалов

        print(f"✅ Обучение завершено. Test RMSE: ${rmse:.2f}")
        return

    # ==========================================
    # 4. ГЕНЕРАЦИЯ СЦЕНАРИЕВ И ПРОГНОЗОВ
    # ==========================================
    def generate_forecasts(self, last_row, months=60):
        """
        Генерирует будущие макро-допущения и прогнозирует цену на золото.
        """
        future_dates = pd.date_range(start=self.data.index[-1], periods=months+1, freq='M')[1:]

        scenarios = ['Base', 'Optimistic', 'Pessimistic']
        results = pd.DataFrame(index=future_dates)

        for scen in scenarios:
            # 1. Генерация макро-допущений (Драйверы)
            future_X = pd.DataFrame(index=future_dates, columns=self.features)

            # Логика дрейфа (Drift) для демонстрации
            # Base: Флэт/Тренд; Opt: DXY вниз, Ставки вниз; Pess: DXY вверх, Ставки вверх
            last_vals = last_row[self.features]

            for i in range(len(future_X)):
                drift_factor = (i / 12) # Годы вперед

                if scen == 'Base':
                    # Возврат к среднему / слабый тренд
                    future_X.iloc[i] = last_vals
                elif scen == 'Optimistic':
                    # Бычий кейс: Доллар слабеет, ставки падают
                    future_X.iloc[i] = last_vals
                    future_X.iloc[i]['DXY'] = last_vals['DXY'] * (1 - 0.02 * drift_factor)
                    future_X.iloc[i]['10Y_Yield'] = last_vals['10Y_Yield'] * (1 - 0.05 * drift_factor)
                elif scen == 'Pessimistic':
                    # Медвежий кейс: Доллар крепнет, ставки растут
                    future_X.iloc[i] = last_vals
                    future_X.iloc[i]['DXY'] = last_vals['DXY'] * (1 + 0.02 * drift_factor)
                    future_X.iloc[i]['10Y_Yield'] = last_vals['10Y_Yield'] * (1 + 0.05 * drift_factor)

            # 2. Прогноз с использованием Ансамбля
            # ElasticNet
            X_sc = self.scalers['X'].transform(future_X.fillna(method='ffill'))
            pred_enet = self.models['ElasticNet'].predict(X_sc)

            # Random Forest
            pred_rf = self.models['RandomForest'].predict(future_X.fillna(method='ffill'))

            # SARIMA (Прогноз от последней точки)
            pred_sarima = self.models['SARIMA'].forecast(steps=months).values

            # Взвешенный ансамбль
            raw_forecast = (0.2 * pred_enet) + (0.4 * pred_rf) + (0.4 * pred_sarima)

            # 3. Пост-обработка: Управленческая надстройка и Сглаживание
            # Геополитическая премия (Base + $50/год)
            geo_premium = np.linspace(0, 200, months) # $200 премии за 5 лет из-за дефицита
            final_forecast = raw_forecast + geo_premium

            results[f'Gold_{scen}'] = final_forecast

        # Добавление Доверительных Интервалов (используя исторический RMSE и затухание во времени)
        z_95 = 1.96
        z_80 = 1.28

        # Неопределенность растет со временем (правило квадратного корня из времени)
        time_decay = np.sqrt(np.arange(1, months + 1))
        std_error = self.sigma * time_decay * 0.5 # Коэффициент тюнинга

        results['CI80_low'] = results['Gold_Base'] - (z_80 * std_error)
        results['CI80_high'] = results['Gold_Base'] + (z_80 * std_error)
        results['CI95_low'] = results['Gold_Base'] - (z_95 * std_error)
        results['CI95_high'] = results['Gold_Base'] + (z_95 * std_error)

        return results

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    # Инициализация
    forecaster = CommoditiesForecaster()

    # 1. Получение данных
    df = forecaster.get_data()

    # 2. Инжиниринг признаков
    df_eng = forecaster.engineer_features(df)

    # 3. Обучение
    forecaster.train_ensemble(df_eng)

    # 4. Прогноз на 5 лет (60 месяцев)
    last_known_row = df_eng.iloc[-1]
    forecast_df = forecaster.generate_forecasts(last_known_row, months=60)

    # 5. Экспорт и Визуализация
    print("\n📊 Forecast Head (Следующие 5 месяцев):")
    print(forecast_df[['Gold_Base', 'Gold_Optimistic', 'Gold_Pessimistic']].head())

    forecast_df.to_csv("Alliance_Altyn_Gold_Forecast.csv")

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-24:], df['Gold'].iloc[-24:], label='История', color='black')
    plt.plot(forecast_df.index, forecast_df['Gold_Base'], label='Базовый прогноз', color='blue')
    plt.plot(forecast_df.index, forecast_df['Gold_Optimistic'], label='Оптимистичный', color='green', linestyle='--')
    plt.plot(forecast_df.index, forecast_df['Gold_Pessimistic'], label='Пессимистичный', color='red', linestyle='--')
    plt.fill_between(forecast_df.index, forecast_df['CI80_low'], forecast_df['CI80_high'], color='blue', alpha=0.1, label='80% CI')
    plt.title('Alliance Altyn: 5-летняя модель прогноза цены на золото')
    plt.xlabel('Год')
    plt.ylabel('Цена (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()