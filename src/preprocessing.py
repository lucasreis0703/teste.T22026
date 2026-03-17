from __future__ import annotations

import pandas as pd


class DataPreprocessor:
    """
    Responsável por todo o pré-processamento usado tanto no treino
    quanto na predição.
    """

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        df_clean = df.copy()

        ids = df_clean["SK_ID_CURR"] if "SK_ID_CURR" in df_clean.columns else None
        if "SK_ID_CURR" in df_clean.columns:
            df_clean = df_clean.drop(columns=["SK_ID_CURR"])

        if "TARGET_CREDIT_LIMIT" in df_clean.columns:
            df_clean = df_clean.drop(columns=["TARGET_CREDIT_LIMIT"])

        if "DAYS_EMPLOYED" in df_clean.columns:
            df_clean["DAYS_EMPLOYED"] = df_clean["DAYS_EMPLOYED"].replace(365243, 0)
            df_clean["ANOS_EMPREGO"] = abs(df_clean["DAYS_EMPLOYED"]) / 365
            df_clean = df_clean.drop(columns=["DAYS_EMPLOYED"])

        if "DAYS_BIRTH" in df_clean.columns:
            df_clean["IDADE_ANOS"] = abs(df_clean["DAYS_BIRTH"]) / 365
            df_clean = df_clean.drop(columns=["DAYS_BIRTH"])

        num_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(-1)

        cat_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df_clean[col] = (
                df_clean[col].astype(str).astype("category").cat.codes
            )

        return df_clean, ids

