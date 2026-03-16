import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Motor de Crédito Inteligente", page_icon="🏦", layout="wide")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Poli-USP.svg/1200px-Poli-USP.svg.png", width=150)
st.sidebar.title("🏦 Sistema de Varejo")
st.sidebar.markdown("Faça o upload de uma base de novos clientes para o Motor de Crédito analisar e gerar os limites na hora.")

# ==========================================
# 1. FUNÇÃO DE PRÉ-PROCESSAMENTO UNIVERSAL
# ==========================================
def limpar_dados(df):
    """Função que prepara os dados tanto para treino quanto para as previsões"""
    df_clean = df.copy()
    
    # Guarda o ID e remove para não treinar com ele
    ids = df_clean['SK_ID_CURR'] if 'SK_ID_CURR' in df_clean.columns else None
    if 'SK_ID_CURR' in df_clean.columns:
        df_clean = df_clean.drop(columns=['SK_ID_CURR'])
        
    if 'TARGET_CREDIT_LIMIT' in df_clean.columns:
        df_clean = df_clean.drop(columns=['TARGET_CREDIT_LIMIT'])
        
    # Tratando o tempo de emprego (Armadilha)
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].replace(365243, 0)
        df_clean['ANOS_EMPREGO'] = abs(df_clean['DAYS_EMPLOYED']) / 365
        df_clean = df_clean.drop(columns=['DAYS_EMPLOYED'])
        
    # Convertendo dias negativos de nascimento
    if 'DAYS_BIRTH' in df_clean.columns:
        df_clean['IDADE_ANOS'] = abs(df_clean['DAYS_BIRTH']) / 365
        df_clean = df_clean.drop(columns=['DAYS_BIRTH'])
    
    # Preenchendo Nulos com -1 (Estratégia segura para Random Forest)
    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(-1)
    
    # Transformando texto em números usando códigos de categoria
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).astype('category').cat.codes
        
    return df_clean, ids

# ==========================================
# 2. O CÉREBRO: TREINAMENTO INTERNO DO MODELO
# ==========================================
# O @st.cache_resource garante que o modelo treine SÓ UMA VEZ quando o app liga
@st.cache_resource(show_spinner="Treinando o Motor de Crédito (Isso acontece só na primeira vez)...")
def inicializar_motor():
    try:
        # 1. Carrega as tabelas de treino originais
        df1 = pd.read_csv('base_infos_pessoais.csv')
        df2 = pd.read_csv('base_regional.csv')
        df3 = pd.read_csv('base_bens.csv')
        df4 = pd.read_csv('base_financeiro.csv')
        df5 = pd.read_csv('base_scores.csv')
        df_y = pd.read_csv('base_target.csv')
        
        # 2. Une tudo
        df_treino = df1.merge(df2, on='SK_ID_CURR').merge(df3, on='SK_ID_CURR')\
                       .merge(df4, on='SK_ID_CURR').merge(df5, on='SK_ID_CURR')
                       
        y_treino = df_y['TARGET_CREDIT_LIMIT']
        
        # 3. Limpa e treina
        X_treino, _ = limpar_dados(df_treino)
        
        modelo = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        modelo.fit(X_treino, y_treino)
        
        return modelo
    except Exception as e:
        st.error("Erro ao treinar o modelo base. Certifique-se de que os 6 arquivos CSV de treino estão nesta pasta.")
        return None

# Inicializa o cérebro
modelo_rf = inicializar_motor()

# ==========================================
# 3. INTERFACE DE USUÁRIO (UPLOAD DA BASE NOVA)
# ==========================================
st.title("💳 Motor de Aprovação Automática")
st.markdown("O modelo interno já está treinado e pronto. Anexe uma base de clientes (ex: `teste_surpresa_5k_cego.csv`) para avaliá-los.")

arquivo_upload = st.file_uploader("Anexe a Base de Clientes (CSV)", type=['csv'])

if arquivo_upload is not None and modelo_rf is not None:
    st.success("Arquivo recebido! Processando clientes...")
    
    # 1. Carrega o arquivo do usuário
    df_novos_clientes = pd.read_csv(arquivo_upload)
    
    # 2. Limpa e gera as previsões usando o modelo interno
    X_novos, ids_novos = limpar_dados(df_novos_clientes)
    previsoes = modelo_rf.predict(X_novos)
    
    # 3. Monta o DataFrame final
    df_resultados = df_novos_clientes.copy()
    df_resultados['LIMITE_LIBERADO'] = np.round(previsoes, 2)
    
    # Simulando o carrinho de compras para avaliar aprovação
    np.random.seed(99)
    df_resultados['VALOR_CARRINHO'] = df_resultados['AMT_INCOME_TOTAL'] * np.random.uniform(0.05, 0.60, len(df_resultados))
    df_resultados['COMPRA_APROVADA'] = df_resultados['LIMITE_LIBERADO'] >= df_resultados['VALOR_CARRINHO']
    
    st.markdown("---")
    
    # ==========================================
    # 4. PAINEL DE NEGÓCIOS (VAREJO)
    # ==========================================
    st.subheader("📊 Resumo Operacional da Loja")
    total = len(df_resultados)
    aprovados = df_resultados['COMPRA_APROVADA'].sum()
    taxa = (aprovados / total) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes Analisados", f"{total:,}")
    c2.metric("Vendas Aprovadas", f"{aprovados:,}")
    c3.metric("Taxa de Conversão", f"{taxa:.1f}%")
    
    # ==========================================
    # 5. CÁLCULO DE MSE (Se for uma base de teste nossa)
    # ==========================================
    # O app tenta carregar o gabarito secretamente para ver se a base anexada é a de 5k
    try:
        df_gabarito = pd.read_csv("CHAVE_SOLUCAO_SURPRESA_5K.csv")
        # Confere se os IDs batem
        if len(df_gabarito) == total and set(df_gabarito['SK_ID_CURR']) == set(df_resultados['SK_ID_CURR']):
            st.markdown("---")
            st.subheader("🎯 Auditoria do Modelo (Métricas Reais)")
            
            # Ordena para garantir que a comparação seja 1 para 1
            df_gabarito = df_gabarito.set_index('SK_ID_CURR').loc[df_resultados['SK_ID_CURR']].reset_index()
            
            mse = mean_squared_error(df_gabarito['TARGET_CREDIT_LIMIT'], df_resultados['LIMITE_LIBERADO'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(df_gabarito['TARGET_CREDIT_LIMIT'], df_resultados['LIMITE_LIBERADO'])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("MSE", f"{mse:,.2f}")
            m2.metric("RMSE (Erro Médio)", f"R$ {rmse:,.2f}")
            m3.metric("MAE", f"R$ {mae:,.2f}")
            
    except FileNotFoundError:
        pass # Se não achar o gabarito, não mostra o bloco de MSE, apenas atua como produção real.

    st.markdown("---")

    # ==========================================
    # 6. BUSCA INDIVIDUAL (Raio-X do Cliente)
    # ==========================================
    st.subheader("🔍 Consultar Veredito por Cliente")
    lista_ids = df_resultados['SK_ID_CURR'].astype(str).tolist()
    cliente_id = st.selectbox("Digite ou selecione o ID do cliente:", lista_ids)
    
    if cliente_id:
        cliente = df_resultados[df_resultados['SK_ID_CURR'] == int(cliente_id)].iloc[0]
        
        st.write(f"**Renda Anual:** R$ {cliente['AMT_INCOME_TOTAL']:,.2f} | **Score:** {cliente['EXT_SOURCE_2']:.2f}")
        
        col_a, col_b = st.columns(2)
        col_a.info(f"🛒 **Valor da Compra (Carrinho):**\nR$ {cliente['VALOR_CARRINHO']:,.2f}")
        col_b.warning(f"🏦 **Limite Liberado pela IA:**\nR$ {cliente['LIMITE_LIBERADO']:,.2f}")
        
        if cliente['COMPRA_APROVADA']:
            st.success("✅ **Status:** APROVADO! Imprimindo carnê...")
        else:
            st.error("❌ **Status:** REPROVADO! Limite insuficiente para o valor da compra.")