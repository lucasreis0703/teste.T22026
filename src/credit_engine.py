from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .preprocessing import DataPreprocessor


class CreditEngine:
    """
    Encapsula o treinamento do modelo de crédito,
    clustering e comparação de diferentes algoritmos.
    """

    def __init__(
        self,
        base_path: Path,
        random_state: int = 42,
    ) -> None:
        self.base_path = base_path
        self.preprocessor = DataPreprocessor()
        self.random_state = random_state

        self.models: Dict[str, object] = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=random_state,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                tree_method="hist",
            ),
            "CatBoost": CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="RMSE",
                verbose=False,
                random_seed=random_state,
            ),
        }

        self.kmeans: Optional[KMeans] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None

    def _read_csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.base_path / name)

    def load_and_prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        df1 = self._read_csv("base_infos_pessoais.csv")
        df2 = self._read_csv("base_regional.csv")
        df3 = self._read_csv("base_bens.csv")
        df4 = self._read_csv("base_financeiro.csv")
        df5 = self._read_csv("base_scores.csv")
        df_y = self._read_csv("base_target.csv")

        df_treino = (
            df1.merge(df2, on="SK_ID_CURR")
            .merge(df3, on="SK_ID_CURR")
            .merge(df4, on="SK_ID_CURR")
            .merge(df5, on="SK_ID_CURR")
        )

        y_treino = df_y["TARGET_CREDIT_LIMIT"]
        X_treino, _ = self.preprocessor.clean(df_treino)

        self._X_train = X_treino
        self._y_train = y_treino

        return X_treino, y_treino

    def train_all_models(self) -> Dict[str, Dict[str, float]]:
        """
        Treina RandomForest, XGBoost e CatBoost e
        devolve métricas de treino (MSE, RMSE, MAE) para comparação.
        """
        X, y = self.load_and_prepare_training_data()

        metrics: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            model.fit(X, y)
            preds = model.predict(X)
            mse = mean_squared_error(y, preds)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y, preds)
            metrics[name] = {
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
            }

        return metrics

    def fit_clusters(self, n_clusters: int = 5) -> None:
        """
        Ajusta KMeans sobre a base de treino para gerar clusters de clientes.
        """
        if self._X_train is None:
            self.load_and_prepare_training_data()

        assert self._X_train is not None

        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        )
        self.kmeans.fit(self._X_train)

    def predict_for_clients(
        self, df_clients: pd.DataFrame, model_name: str = "RandomForest"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if model_name not in self.models:
            raise ValueError(f"Modelo desconhecido: {model_name}")

        X, _ = self.preprocessor.clean(df_clients)
        model = self.models[model_name]
        preds = model.predict(X)

        df_resultados = df_clients.copy()
        df_resultados[f"LIMITE_{model_name}"] = np.round(preds, 2)

        if self.kmeans is not None:
            clusters = self.kmeans.predict(X)
            df_resultados["CLUSTER"] = clusters

        return df_resultados, preds

    def predict_with_all_models(
        self, df_clients: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Gera previsões para a base enviada usando todos os modelos.
        """
        X, _ = self.preprocessor.clean(df_clients)

        df_resultados = df_clients.copy()
        all_preds: Dict[str, np.ndarray] = {}

        for name, model in self.models.items():
            preds = model.predict(X)
            df_resultados[f"LIMITE_{name}"] = np.round(preds, 2)
            all_preds[name] = preds

        if self.kmeans is not None:
            clusters = self.kmeans.predict(X)
            df_resultados["CLUSTER"] = clusters

        return df_resultados, all_preds

    def evaluate_with_key(
        self, df_resultados: pd.DataFrame, coluna_limite: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Se o arquivo de gabarito existir e os IDs coincidirem,
        devolve (mse, rmse, mae) para a coluna de limite informada.
        Caso contrário, retorna None.
        """
        chave_path = self.base_path / "CHAVE_SOLUCAO_SURPRESA_5K.csv"
        if not chave_path.exists():
            return None

        df_gabarito = pd.read_csv(chave_path)

        total = len(df_resultados)
        if len(df_gabarito) != total:
            return None

        if set(df_gabarito["SK_ID_CURR"]) != set(df_resultados["SK_ID_CURR"]):
            return None

        df_gabarito = (
            df_gabarito.set_index("SK_ID_CURR")
            .loc[df_resultados["SK_ID_CURR"]]
            .reset_index()
        )

        mse = mean_squared_error(
            df_gabarito["TARGET_CREDIT_LIMIT"], df_resultados[coluna_limite]
        )
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(
            df_gabarito["TARGET_CREDIT_LIMIT"], df_resultados[coluna_limite]
        )

        return float(mse), float(rmse), float(mae)
