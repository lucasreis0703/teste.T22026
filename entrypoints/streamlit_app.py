from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.credit_engine import CreditEngine


@st.cache_resource(
    show_spinner="Treinando e comparando modelos (isso acontece só na primeira vez)..."
)
def get_engine_and_metrics():
    base_path = Path(__file__).resolve().parent.parent
    engine = CreditEngine(base_path=base_path)
    model_metrics = engine.train_all_models()
    engine.fit_clusters(n_clusters=5)
    return engine, model_metrics


def main() -> None:
    st.set_page_config(
        page_title="Motor de Crédito Inteligente",
        page_icon="🏦",
        layout="wide",
    )

    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Poli-USP.svg/1200px-Poli-USP.svg.png",
        width=150,
    )
    st.sidebar.title("🏦 Sistema de Varejo")
    st.sidebar.markdown(
        "Faça o upload de uma base de novos clientes (sem a target) "
        "para o Motor de Crédito analisar e gerar os limites na hora."
    )

    engine, model_metrics = get_engine_and_metrics()

    st.title("💳 Motor de Aprovação Automática")
    st.markdown(
        "Os modelos internos já estão treinados e comparados. "
        "Anexe uma base de clientes (sem a coluna de target) para avaliá-los."
    )

    arquivo_upload = st.file_uploader(
        "Anexe a Base de Clientes (CSV)", type=["csv"]
    )

    if arquivo_upload is None or engine is None:
        return

    st.success("Arquivo recebido! Processando clientes...")

    df_novos_clientes = pd.read_csv(arquivo_upload)

    # 1) Gera previsões com TODOS os modelos e clusters
    df_resultados, all_preds = engine.predict_with_all_models(df_novos_clientes)

    # Define um modelo "principal" para usar no painel de varejo
    coluna_limite_referencia = "LIMITE_RandomForest"

    np.random.seed(99)
    df_resultados["VALOR_CARRINHO"] = (
        df_resultados["AMT_INCOME_TOTAL"]
        * np.random.uniform(0.05, 0.60, len(df_resultados))
    )
    df_resultados["COMPRA_APROVADA"] = (
        df_resultados[coluna_limite_referencia] >= df_resultados["VALOR_CARRINHO"]
    )

    st.markdown("---")

    st.subheader("📊 Resumo Operacional da Loja (usando RandomForest)")
    total = len(df_resultados)
    aprovados = df_resultados["COMPRA_APROVADA"].sum()
    taxa = (aprovados / total) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes Analisados", f"{total:,}")
    c2.metric("Vendas Aprovadas", f"{aprovados:,}")
    c3.metric("Taxa de Conversão", f"{taxa:.1f}%")

    st.markdown("---")

    # 2) Comparação de modelos no treino
    st.subheader("🧠 Comparação de Modelos (base de treino interna)")
    df_metrics = pd.DataFrame(model_metrics).T.reset_index(names="modelo")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.dataframe(df_metrics.style.format({"mse": "{:,.2f}", "rmse": "{:,.2f}", "mae": "{:,.2f}"}))
    with col_m2:
        fig_bar = px.bar(
            df_metrics,
            x="modelo",
            y="rmse",
            title="RMSE por modelo (quanto menor, melhor)",
            text="rmse",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3) Auditoria opcional com gabarito, se a base de 5k estiver presente e compatível
    st.markdown("---")
    st.subheader("🎯 Auditoria dos modelos com gabarito (se disponível)")
    audit_rows = []
    for name in engine.models.keys():
        col_limite = f"LIMITE_{name}"
        metrics = engine.evaluate_with_key(df_resultados, coluna_limite=col_limite)
        if metrics is not None:
            mse, rmse, mae = metrics
            audit_rows.append(
                {"modelo": name, "mse": mse, "rmse": rmse, "mae": mae}
            )
    if audit_rows:
        df_audit = pd.DataFrame(audit_rows)
        st.dataframe(
            df_audit.style.format(
                {"mse": "{:,.2f}", "rmse": "{:,.2f}", "mae": "{:,.2f}"}
            )
        )
        fig_audit = px.bar(
            df_audit, x="modelo", y="rmse", title="RMSE com gabarito real"
        )
        st.plotly_chart(fig_audit, use_container_width=True)
    else:
        st.info(
            "Gabarito (`CHAVE_SOLUCAO_SURPRESA_5K.csv`) não encontrado ou incompatível. "
            "Rodando em modo produção sem auditoria de target."
        )

    st.markdown("---")

    st.subheader("🔍 Consultar Veredito por Cliente")
    lista_ids = df_resultados["SK_ID_CURR"].astype(str).tolist()
    cliente_id = st.selectbox(
        "Digite ou selecione o ID do cliente:", lista_ids
    )

    if not cliente_id:
        return

    cliente = df_resultados[df_resultados["SK_ID_CURR"] == int(cliente_id)].iloc[0]

    st.write(
        f"**Renda Anual:** R$ {cliente['AMT_INCOME_TOTAL']:,.2f} | "
        f"**Score:** {cliente.get('EXT_SOURCE_2', np.nan):.2f} | "
        f"**Cluster:** {int(cliente['CLUSTER']) if 'CLUSTER' in cliente else 'N/A'}"
    )

    col_a, col_b = st.columns(2)
    col_a.info(
        "🛒 **Valor da Compra (Carrinho):**\n"
        f"R$ {cliente['VALOR_CARRINHO']:,.2f}"
    )
    col_b.warning(
        "🏦 **Limites Liberados (Modelos):**\n"
        + "\n".join(
            [
                f"{name}: R$ {cliente[f'LIMITE_{name}']:,.2f}"
                for name in engine.models.keys()
            ]
        )
    )

    if cliente["COMPRA_APROVADA"]:
        st.success("✅ **Status:** APROVADO! Imprimindo carnê...")
    else:
        st.error(
            "❌ **Status:** REPROVADO! "
            "Limite insuficiente para o valor da compra."
        )

    # 4) Gráficos de clusters e distribuição de clientes
    st.markdown("---")
    st.subheader("🧩 Visualização de Clusters e Clientes")

    if "CLUSTER" in df_resultados.columns:
        # Scatter simples em 2D usando duas features principais, se existirem
        feature_x = "AMT_INCOME_TOTAL" if "AMT_INCOME_TOTAL" in df_resultados.columns else None
        feature_y = "IDADE_ANOS" if "IDADE_ANOS" in df_resultados.columns else None

        if feature_x and feature_y:
            fig_cluster = px.scatter(
                df_resultados,
                x=feature_x,
                y=feature_y,
                color="CLUSTER",
                hover_data=["SK_ID_CURR"],
                title="Clusters de Clientes (renda x idade)",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info(
                "Não foi possível encontrar colunas padrão para plotar clusters "
                "(esperado algo como `AMT_INCOME_TOTAL` e `IDADE_ANOS`)."
            )

    # Distribuição de limites previstos por modelo
    st.subheader("📈 Distribuição de Limites por Modelo")
    df_long = df_resultados.melt(
        id_vars=["SK_ID_CURR"],
        value_vars=[f"LIMITE_{name}" for name in engine.models.keys()],
        var_name="modelo",
        value_name="limite",
    )
    fig_hist = px.histogram(
        df_long,
        x="limite",
        color="modelo",
        barmode="overlay",
        nbins=40,
        title="Distribuição de limites previstos por modelo",
        opacity=0.6,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


if __name__ == "__main__":
    main()

