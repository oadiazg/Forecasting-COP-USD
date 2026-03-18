"""
RandomWalk.py
=============
Modelo de Caminata Aleatoria (Random Walk) para simulación de tasas de cambio.

El Random Walk es un modelo de referencia (benchmark) ampliamente utilizado en
finanzas para predecir tasas de cambio. Se basa en la hipótesis de mercado
eficiente: el mejor predictor del precio de mañana es el precio de hoy más
un componente aleatorio (ruido).

Fórmula:
    S(t+1) = S(t) * exp(mu * dt + sigma * sqrt(dt) * Z)

donde:
    S(t)  = tasa de cambio en el tiempo t
    mu    = drift (tendencia promedio)
    sigma = volatilidad
    dt    = paso de tiempo (por defecto = 1)
    Z     ~ N(0,1) = número aleatorio normal estándar

Uso:
    python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv \
                                  --pred_len 30 --num_simulations 1000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


class RandomWalkModel:
    """
    Modelo de Caminata Aleatoria (Geometric Brownian Motion) para
    predicción de tasas de cambio COP/USD.
    """

    def __init__(self, pred_len: int = 30, num_simulations: int = 1000,
                 confidence_level: float = 0.95, seed: int = 42):
        """
        Args:
            pred_len:          Número de pasos a predecir hacia el futuro
            num_simulations:   Número de trayectorias de Monte Carlo
            confidence_level:  Nivel de confianza para el intervalo de predicción
            seed:              Semilla aleatoria para reproducibilidad
        """
        self.pred_len = pred_len
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.seed = seed
        self.mu = None
        self.sigma = None
        self.last_price = None
        self.fitted = False

    def fit(self, series: np.ndarray):
        """
        Estima los parámetros del modelo (drift y volatilidad) a partir de
        los datos históricos.

        Args:
            series: Array 1D con la serie histórica de precios/tasas de cambio
        """
        series = np.array(series, dtype=float)
        if len(series) < 2:
            raise ValueError("Se requieren al menos 2 observaciones para ajustar el modelo.")

        # Calcular retornos logarítmicos
        log_returns = np.diff(np.log(series))

        # Estimar parámetros
        self.mu = np.mean(log_returns)       # Drift diario
        self.sigma = np.std(log_returns)     # Volatilidad diaria
        self.last_price = series[-1]
        self.fitted = True

        print(f"Modelo RandomWalk ajustado:")
        print(f"  Último precio:  {self.last_price:.4f}")
        print(f"  Drift (mu):     {self.mu:.6f} por período")
        print(f"  Volatilidad:    {self.sigma:.6f} por período")
        return self

    def simulate(self) -> np.ndarray:
        """
        Genera múltiples trayectorias de simulación usando el método de
        Monte Carlo (Geometric Brownian Motion).

        Returns:
            simulations: Array de forma (num_simulations, pred_len) con
                         todas las trayectorias simuladas
        """
        if not self.fitted:
            raise RuntimeError("Primero debe ajustar el modelo con .fit()")

        np.random.seed(self.seed)
        dt = 1  # Un paso de tiempo por período

        # Generar ruido aleatorio: (num_simulations, pred_len)
        random_shocks = np.random.normal(0, 1, (self.num_simulations, self.pred_len))

        # Calcular retornos para cada trayectoria
        daily_returns = np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * dt
            + self.sigma * np.sqrt(dt) * random_shocks
        )

        # Calcular precios acumulados
        simulations = np.zeros((self.num_simulations, self.pred_len))
        simulations[:, 0] = self.last_price * daily_returns[:, 0]

        for t in range(1, self.pred_len):
            simulations[:, t] = simulations[:, t - 1] * daily_returns[:, t]

        return simulations

    def predict(self) -> dict:
        """
        Calcula estadísticas de predicción a partir de las simulaciones.

        Returns:
            dict con:
                - 'mean':       Predicción media en cada paso
                - 'median':     Predicción mediana en cada paso
                - 'lower':      Límite inferior del intervalo de confianza
                - 'upper':      Límite superior del intervalo de confianza
                - 'simulations': Todas las trayectorias simuladas
        """
        simulations = self.simulate()
        alpha = 1 - self.confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100

        return {
            'mean':        np.mean(simulations, axis=0),
            'median':      np.median(simulations, axis=0),
            'lower':       np.percentile(simulations, lower_pct, axis=0),
            'upper':       np.percentile(simulations, upper_pct, axis=0),
            'simulations': simulations,
        }

    def evaluate(self, true_values: np.ndarray) -> dict:
        """
        Evalúa el desempeño del modelo comparando predicciones con valores reales.

        Args:
            true_values: Array con los valores reales del período de predicción

        Returns:
            dict con métricas MAE, RMSE, MAPE, RSE
        """
        predictions = self.predict()
        pred = predictions['mean']
        true = np.array(true_values, dtype=float)

        n = min(len(pred), len(true))
        pred = pred[:n]
        true = true[:n]

        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mape = np.mean(np.abs((pred - true) / true)) * 100
        rse = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

        metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'RSE': rse}

        print("\nMétricas de evaluación (Random Walk):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        return metrics

    def plot(self, historical_series: np.ndarray = None,
             true_values: np.ndarray = None,
             output_path: str = None,
             title: str = "Random Walk - Predicción Tasa COP/USD"):
        """
        Genera gráfico con las simulaciones y el intervalo de confianza.

        Args:
            historical_series: Serie histórica para mostrar contexto
            true_values:       Valores reales (si están disponibles)
            output_path:       Ruta para guardar el gráfico (ej: 'resultado.png')
            title:             Título del gráfico
        """
        predictions = self.predict()

        fig, ax = plt.subplots(figsize=(12, 6))

        t_pred = np.arange(1, self.pred_len + 1)

        # Mostrar serie histórica
        if historical_series is not None:
            t_hist = np.arange(-len(historical_series) + 1, 1)
            ax.plot(t_hist, historical_series, color='black', label='Histórico', linewidth=1.5)

        # Mostrar algunas trayectorias simuladas (máx. 100 para no saturar)
        n_show = min(100, self.num_simulations)
        for i in range(n_show):
            ax.plot(t_pred, predictions['simulations'][i], color='lightblue', alpha=0.1, linewidth=0.5)

        # Intervalo de confianza
        ax.fill_between(
            t_pred,
            predictions['lower'],
            predictions['upper'],
            color='steelblue', alpha=0.3,
            label=f'IC {int(self.confidence_level * 100)}%'
        )

        # Predicción media
        ax.plot(t_pred, predictions['mean'], color='steelblue', linewidth=2, label='Media predicha')

        # Valores reales (si disponibles)
        if true_values is not None:
            n = min(self.pred_len, len(true_values))
            ax.plot(t_pred[:n], true_values[:n], color='red', linewidth=2,
                    linestyle='--', label='Valor real')

        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Períodos hacia el futuro')
        ax.set_ylabel('Tasa COP/USD')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=150)
            print(f"Gráfico guardado en: {output_path}")

        plt.show()
        return fig


def load_exchange_rate_series(data_path: str, target_col: str = None,
                               date_col: str = 'date') -> tuple:
    """
    Carga una serie de tasas de cambio desde un archivo CSV.

    Args:
        data_path:  Ruta al archivo CSV
        target_col: Nombre de la columna objetivo (si None, usa la última columna)
        date_col:   Nombre de la columna de fechas

    Returns:
        (series, dates) - array de precios y array de fechas
    """
    df = pd.read_csv(data_path)

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
    else:
        dates = pd.RangeIndex(len(df))

    if target_col is None:
        # Usar la última columna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No se encontraron columnas numéricas en el archivo.")
        target_col = numeric_cols[-1]
        print(f"Columna objetivo seleccionada automáticamente: '{target_col}'")

    series = df[target_col].values
    return series, dates


def main():
    parser = argparse.ArgumentParser(
        description='Random Walk - Modelo de simulación para tasas de cambio COP/USD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Simulación básica con 30 días de predicción
  python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv --pred_len 30

  # Con 5000 simulaciones y columna objetivo específica
  python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv \\
    --target_col COP_USD --pred_len 60 --num_simulations 5000

  # Guardar gráfico
  python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv \\
    --pred_len 30 --output_plot resultados/random_walk.png
        """
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al archivo CSV con los datos históricos')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Columna objetivo (por defecto: última columna numérica)')
    parser.add_argument('--date_col', type=str, default='date',
                        help='Columna de fechas (por defecto: date)')
    parser.add_argument('--pred_len', type=int, default=30,
                        help='Número de períodos a predecir (por defecto: 30)')
    parser.add_argument('--num_simulations', type=int, default=1000,
                        help='Número de trayectorias Monte Carlo (por defecto: 1000)')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                        help='Nivel de confianza del intervalo (por defecto: 0.95)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proporción datos de entrenamiento (por defecto: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para reproducibilidad (por defecto: 42)')
    parser.add_argument('--output_plot', type=str, default=None,
                        help='Ruta para guardar el gráfico (ej: resultados/plot.png)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Desactivar visualización del gráfico')

    args = parser.parse_args()

    # Cargar datos
    print(f"\nCargando datos desde: {args.data_path}")
    series, dates = load_exchange_rate_series(args.data_path, args.target_col, args.date_col)
    print(f"Total de observaciones: {len(series)}")
    print(f"Rango: {series.min():.4f} - {series.max():.4f}")

    # Dividir en entrenamiento y prueba
    n_train = int(len(series) * args.train_ratio)
    train_series = series[:n_train]
    test_series = series[n_train:n_train + args.pred_len] if n_train < len(series) else None

    print(f"Observaciones de entrenamiento: {len(train_series)}")
    if test_series is not None:
        print(f"Observaciones de prueba disponibles: {len(test_series)}")

    # Ajustar y simular
    model = RandomWalkModel(
        pred_len=args.pred_len,
        num_simulations=args.num_simulations,
        confidence_level=args.confidence_level,
        seed=args.seed
    )
    model.fit(train_series)

    # Evaluar si hay datos de prueba
    if test_series is not None and len(test_series) > 0:
        model.evaluate(test_series)

    # Guardar predicciones en CSV
    predictions = model.predict()
    results_df = pd.DataFrame({
        'periodo': np.arange(1, args.pred_len + 1),
        'prediccion_media':   predictions['mean'],
        'prediccion_mediana': predictions['median'],
        'limite_inferior':    predictions['lower'],
        'limite_superior':    predictions['upper'],
    })
    results_path = 'resultados_random_walk.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nPredicciones guardadas en: {results_path}")
    print(results_df.head(10).to_string(index=False))

    # Visualización
    if not args.no_plot:
        model.plot(
            historical_series=train_series[-60:],
            true_values=test_series,
            output_path=args.output_plot
        )


if __name__ == '__main__':
    main()
