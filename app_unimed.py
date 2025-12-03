import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Configuração da página
st.set_page_config(
    page_title="Análise de Dados Hospitalares",
    page_icon="🏥",
    layout="wide"
)

# Título da aplicação
st.title("🏥 Análise de Dados Hospitalares - Ciência de Dados")

# Aviso sobre configuração de limite de upload
with st.expander("⚠️ IMPORTANTE: Configuração para arquivos grandes", expanded=False):
    st.markdown("""
    ### Para arquivos maiores que 200MB:
    
    **Opção 1: Configurar via linha de comando:**
    ```bash
    streamlit run app.py --server.maxUploadSize=1024
    ```
    
    **Opção 2: Criar arquivo `config.toml` na pasta `.streamlit`:**
    ```toml
    [server]
    maxUploadSize = 1024  # Em MB (1GB = 1024MB)
    ```
    
    **Opção 3: Usar amostragem do app:**
    - Configure para usar apenas uma porcentagem dos dados
    - Ou limite o número máximo de linhas
    - Ideal para análise exploratória
    """)
    st.info("💡 **Dica:** Para arquivos muito grandes (>500MB), recomendo usar a opção de amostragem mesmo com o limite aumentado.")

st.markdown("---")

# Função para carregar dados de diferentes formatos com amostragem otimizada
@st.cache_data(ttl=3600, show_spinner=True)
def load_data(file, file_type, sample_percentage=100, max_rows=None, use_sample=True):
    """Carrega dados de arquivos CSV ou Parquet com opção de amostragem otimizada"""
    try:
        if file_type == 'csv':
            if use_sample and (sample_percentage < 100 or max_rows):
                if max_rows:
                    df = pd.read_csv(file, nrows=max_rows)
                elif sample_percentage < 100:
                    chunk_size = 10000
                    chunks = []
                    total_read = 0
                    
                    for chunk in pd.read_csv(file, chunksize=chunk_size):
                        sample_size = max(1, int(len(chunk) * sample_percentage / 100))
                        chunks.append(chunk.sample(n=sample_size, random_state=42))
                        total_read += len(chunk)
                        
                        if total_read > 1000000:
                            break
                    
                    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                else:
                    df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
                
        elif file_type == 'parquet':
            if use_sample and (sample_percentage < 100 or max_rows):
                import pyarrow.parquet as pq
                
                parquet_file = pq.ParquetFile(file)
                total_rows = parquet_file.metadata.num_rows
                
                if max_rows:
                    sample_size = min(max_rows, total_rows)
                else:
                    sample_size = int(total_rows * (sample_percentage / 100))
                
                df = parquet_file.read_row_groups(
                    row_groups=np.random.choice(
                        parquet_file.metadata.num_row_groups,
                        size=min(parquet_file.metadata.num_row_groups, 
                               max(1, int(parquet_file.metadata.num_row_groups * sample_percentage / 100))),
                        replace=False
                    )
                ).to_pandas()
                
                if len(df) > sample_size * 1.5:
                    df = df.sample(n=sample_size, random_state=42)
            else:
                df = pd.read_parquet(file)
        else:
            st.error("Formato de arquivo não suportado")
            return pd.DataFrame()
        
        # Converter colunas de data
        date_columns = ['dt_movimento_estoque', 'dt_referencia', 'dt_cadastramento']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Converter ds_operacao para categoria
        if 'ds_operacao' in df.columns:
            df['ds_operacao'] = df['ds_operacao'].astype('category')
        
        gc.collect()
        return df
    except MemoryError:
        st.error("❌ Erro de memória! Use amostragem menor.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return pd.DataFrame()

# Função para aplicar filtros
def aplicar_filtros(df, filtros):
    """Aplica filtros ao DataFrame sem modificar o original"""
    df_filtrado = df.copy()
    
    # Filtro por data
    if filtros.get('data_inicio') and filtros.get('data_fim'):
        if 'dt_movimento_estoque' in df_filtrado.columns:
            df_filtrado = df_filtrado[
                (df_filtrado['dt_movimento_estoque'].dt.date >= filtros['data_inicio']) & 
                (df_filtrado['dt_movimento_estoque'].dt.date <= filtros['data_fim'])
            ]
    
    # Filtro por estabelecimento
    if filtros.get('estabelecimento') and filtros['estabelecimento'] != 'Todos':
        if 'ds_estabelecimento' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['ds_estabelecimento'] == filtros['estabelecimento']]
    
    # Filtro por grupo de material
    if filtros.get('grupo_material') and filtros['grupo_material'] != 'Todos':
        if 'ds_grupo_material' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['ds_grupo_material'] == filtros['grupo_material']]
    
    # Filtro por operação
    if filtros.get('operacao') and filtros['operacao'] != 'Todos':
        if 'ds_operacao' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['ds_operacao'] == filtros['operacao']]
    
    # Filtro especial para desperdícios
    if filtros.get('filtrar_desperdicios') and filtros.get('tipos_desperdicio'):
        if 'ds_operacao' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['ds_operacao'].isin(filtros['tipos_desperdicio'])]
    
    return df_filtrado

# Função para ordenar opções com "Todos" no final
def sort_options_with_all_last(options_list):
    """Ordena uma lista de opções colocando 'Todos' no final"""
    all_options = [opt for opt in options_list if str(opt) != 'Todos']
    if 'Todos' in options_list:
        all_options.append('Todos')
    return all_options

# Função para obter valor seguro para selectbox
def get_safe_default(options_list, default_value, fallback_value='Todos'):
    """Obtém um valor padrão seguro para selectbox"""
    if default_value in options_list:
        return default_value
    elif fallback_value in options_list:
        return fallback_value
    elif options_list:
        return options_list[0]
    else:
        return None

# Inicializar estado da sessão
if 'filtros' not in st.session_state:
    st.session_state.filtros = {
        'data_inicio': None,
        'data_fim': None,
        'estabelecimento': 'Todos',
        'grupo_material': 'Todos',
        'operacao': 'Todos',
        'filtrar_desperdicios': False,
        'tipos_desperdicio': []
    }

if 'df_original' not in st.session_state:
    st.session_state.df_original = pd.DataFrame()

if 'analise_desperdicios' not in st.session_state:
    st.session_state.analise_desperdicios = {
        'tipos_selecionados': [],
        'anos_selecionados': [],
        'tipo_analise': "Quantidade de Movimentações"
    }

# Upload de arquivo
st.sidebar.header("📁 Carregar Dados")

option = st.sidebar.radio(
    "Selecione a fonte de dados:",
    ["📤 Upload de arquivo", "📂 Usar arquivo local", "🔄 Dados de demonstração"]
)

df_original = st.session_state.df_original
sample_percentage = 100
max_rows = None
use_sample = True

if option == "📤 Upload de arquivo":
    uploaded_file = st.sidebar.file_uploader(
        "Escolha um arquivo", 
        type=['csv', 'parquet'],
        help="Carregue arquivos CSV ou Parquet"
    )
    
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            file_type = 'csv'
        elif file_name.endswith('.parquet'):
            file_type = 'parquet'
        else:
            st.error("Formato de arquivo não suportado")
            st.stop()
        
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        with st.sidebar.expander("⚙️ Configuração de Amostragem", expanded=file_size_mb > 50):
            st.write(f"📏 Tamanho estimado: {file_size_mb:.1f} MB")
            
            if file_size_mb > 50:
                st.warning("⚠️ Arquivo grande detectado. Recomendo usar amostragem.")
                use_sample = st.checkbox("Usar amostragem para melhor performance", value=True)
            else:
                use_sample = st.checkbox("Usar amostragem", value=False)
            
            if use_sample:
                sample_option = st.radio(
                    "Tipo de amostragem:",
                    ["Porcentagem dos dados", "Número máximo de linhas"]
                )
                
                if sample_option == "Porcentagem dos dados":
                    sample_percentage = st.slider(
                        "Porcentagem dos dados a usar:",
                        min_value=1,
                        max_value=100,
                        value=5 if file_size_mb > 200 else 100,
                        help=f"Use {5 if file_size_mb > 200 else 10}% para arquivos grandes"
                    )
                else:
                    max_rows = st.number_input(
                        "Número máximo de linhas:",
                        min_value=100,
                        max_value=10000000,
                        value=50000 if file_size_mb > 100 else 100000,
                        step=1000,
                        help="Limite o número de linhas para análise mais rápida"
                    )
            else:
                if file_size_mb > 200:
                    st.error("⚠️ Arquivo muito grande para carregar sem amostragem!")
                    st.stop()
        
        if st.sidebar.button("🚀 Carregar Dados", type="primary"):
            with st.spinner(f"Carregando dados ({sample_percentage if use_sample and sample_percentage < 100 else 100}%)..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    df_loaded = load_data(temp_path, file_type, sample_percentage, max_rows, use_sample)
                    
                    if not df_loaded.empty:
                        st.session_state.df_original = df_loaded
                        df_original = df_loaded
                        
                        # Resetar filtros com valores válidos
                        min_date = df_loaded['dt_movimento_estoque'].min().date() if 'dt_movimento_estoque' in df_loaded.columns else date.today()
                        max_date = df_loaded['dt_movimento_estoque'].max().date() if 'dt_movimento_estoque' in df_loaded.columns else date.today()
                        
                        st.session_state.filtros = {
                            'data_inicio': min_date,
                            'data_fim': max_date,
                            'estabelecimento': 'Todos',
                            'grupo_material': 'Todos',
                            'operacao': 'Todos',
                            'filtrar_desperdicios': False,
                            'tipos_desperdicio': []
                        }
                        
                        # Resetar análise de desperdícios
                        st.session_state.analise_desperdicios = {
                            'tipos_selecionados': [],
                            'anos_selecionados': [2023, 2024] if 'ano' in df_loaded.columns and 2023 in df_loaded['ano'].unique() and 2024 in df_loaded['ano'].unique() else [],
                            'tipo_analise': "Quantidade de Movimentações"
                        }
                        
                        st.sidebar.success(f"✅ {len(df_loaded):,} linhas carregadas com sucesso!")
                        if use_sample and sample_percentage < 100:
                            st.sidebar.info(f"📊 Amostra: {sample_percentage}% do arquivo original")
                        elif use_sample and max_rows:
                            st.sidebar.info(f"📊 Limite: {max_rows:,} linhas")
                    else:
                        st.sidebar.error("❌ Falha ao carregar dados")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

elif option == "📂 Usar arquivo local":
    available_files = []
    
    if os.path.exists("base_tratada.csv"):
        available_files.append(("base_tratada.csv", "csv"))
    
    for file in os.listdir("."):
        if file.lower().endswith('.parquet'):
            available_files.append((file, "parquet"))
    
    if available_files:
        file_options = [f"{name} ({type.upper()})" for name, type in available_files]
        selected_file = st.sidebar.selectbox("Selecione um arquivo local:", file_options)
        
        if selected_file:
            file_name = selected_file.split(" (")[0]
            file_type = selected_file.split("(")[1].replace(")", "").lower()
            
            with st.sidebar.expander("⚙️ Configuração de Amostragem", expanded=False):
                sample_option = st.radio(
                    "Tipo de amostragem:",
                    ["Usar todos os dados", "Porcentagem dos dados", "Número máximo de linhas"],
                    key="local_sample"
                )
                
                if sample_option == "Porcentagem dos dados":
                    sample_percentage = st.slider(
                        "Porcentagem dos dados a usar:",
                        min_value=1,
                        max_value=100,
                        value=100,
                        help="Use uma porcentagem menor para datasets muito grandes",
                        key="local_percentage"
                    )
                    use_sample = True
                elif sample_option == "Número máximo de linhas":
                    max_rows = st.number_input(
                        "Número máximo de linhas:",
                        min_value=100,
                        max_value=10000000,
                        value=100000,
                        step=1000,
                        help="Limite o número de linhas para análise mais rápida",
                        key="local_rows"
                    )
                    use_sample = True
                else:
                    use_sample = False
            
            if st.sidebar.button("🚀 Carregar Dados Local", type="primary"):
                with st.spinner(f"Carregando dados ({sample_percentage if use_sample and sample_percentage < 100 else 100}%)..."):
                    df_loaded = load_data(file_name, file_type, sample_percentage, max_rows, use_sample)
                    if not df_loaded.empty:
                        st.session_state.df_original = df_loaded
                        df_original = df_loaded
                        
                        # Resetar filtros com valores válidos
                        min_date = df_loaded['dt_movimento_estoque'].min().date() if 'dt_movimento_estoque' in df_loaded.columns else date.today()
                        max_date = df_loaded['dt_movimento_estoque'].max().date() if 'dt_movimento_estoque' in df_loaded.columns else date.today()
                        
                        st.session_state.filtros = {
                            'data_inicio': min_date,
                            'data_fim': max_date,
                            'estabelecimento': 'Todos',
                            'grupo_material': 'Todos',
                            'operacao': 'Todos',
                            'filtrar_desperdicios': False,
                            'tipos_desperdicio': []
                        }
                        
                        # Resetar análise de desperdícios
                        st.session_state.analise_desperdicios = {
                            'tipos_selecionados': [],
                            'anos_selecionados': [2023, 2024] if 'ano' in df_loaded.columns and 2023 in df_loaded['ano'].unique() and 2024 in df_loaded['ano'].unique() else [],
                            'tipo_analise': "Quantidade de Movimentações"
                        }
                        
                        st.sidebar.success(f"✅ {len(df_loaded):,} linhas carregadas com sucesso!")
                        if use_sample and sample_percentage < 100:
                            st.sidebar.info(f"📊 Amostra: {sample_percentage}% do arquivo original")
                        elif use_sample and max_rows:
                            st.sidebar.info(f"📊 Limite: {max_rows:,} linhas")
    else:
        st.sidebar.warning("Nenhum arquivo CSV ou Parquet encontrado no diretório.")

else:  # Dados de demonstração
    st.sidebar.info("Usando dados de demonstração")
    
    with st.sidebar.expander("⚙️ Configuração do Dataset Demo", expanded=False):
        sample_size = st.slider(
            "Tamanho do dataset demo:",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000,
            help="Ajuste o tamanho do dataset de demonstração"
        )
    
    # Criar dados de demonstração
    demo_dates = pd.date_range(start='2023-01-01', end='2024-10-31', freq='D')
    demo_materials = [f'Material {chr(65+i)}' for i in range(10)]
    demo_operations = ['Quebras e Contaminações', 'Produtos vencidos', 'Perdas e Quebras', 
                      'Quebras/Contaminação Med Controlados', 'Perdas por estabilidade', 
                      'Medicamentos Controlados Vencidos', 'Consumo']
    
    np.random.seed(42)
    
    demo_data = {
        'ds_centro_custo': np.random.choice(['Posto Carambeí', 'SADT - RADIOLOGIA', 'Coleta Ambulatorial', 
                                           '3º ANDAR - UNIDADE DE INTERNAÇÃO', 'MANUTENÇÃO', 'UNIMED 24 HORAS'], 
                                          sample_size),
        'ds_estabelecimento': np.random.choice(['Laboratorio Unimed Ponta Grossa', 'Hospital Geral Unimed', 
                                              'SADT - TOMOGRAFIA UNIMED'], sample_size),
        'cd_material': np.random.randint(10000, 99999, sample_size),
        'dt_movimento_estoque': pd.to_datetime(np.random.choice(demo_dates, sample_size)),
        'ds_operacao': np.random.choice(demo_operations, sample_size, 
                                       p=[0.05, 0.1, 0.15, 0.05, 0.02, 0.08, 0.55]),
        'dt_referencia': pd.to_datetime(np.random.choice(demo_dates, sample_size)),
        'qt_estoque': np.random.randint(1, 100, sample_size),
        'vl_movimento': np.random.exponential(500, sample_size),
        'vl_consumo': np.random.exponential(500, sample_size),
        'qt_consumo': np.random.poisson(10, sample_size),
        'ds_material_hospital': np.random.choice(demo_materials, sample_size),
        'ie_ativo': np.random.choice([True, False], sample_size, p=[0.8, 0.2]),
        'ds_grupo_material': np.random.choice(['Impressos e Material de Expediente', 'Materiais Hospitalares',
                                             'Bens e Materiais de Manutenção e Conservação', 'Medicamentos'], 
                                            sample_size),
        'ano': np.random.choice([2023, 2024], sample_size, p=[0.4, 0.6]),
        'mes': np.random.randint(1, 13, sample_size)
    }
    
    df_loaded = pd.DataFrame(demo_data)
    st.session_state.df_original = df_loaded
    st.session_state.demo_data_size = sample_size
    
    # Resetar filtros com valores válidos
    min_date = df_loaded['dt_movimento_estoque'].min().date()
    max_date = df_loaded['dt_movimento_estoque'].max().date()
    
    st.session_state.filtros = {
        'data_inicio': min_date,
        'data_fim': max_date,
        'estabelecimento': 'Todos',
        'grupo_material': 'Todos',
        'operacao': 'Todos',
        'filtrar_desperdicios': False,
        'tipos_desperdicio': []
    }
    
    # Resetar análise de desperdícios
    st.session_state.analise_desperdicios = {
        'tipos_selecionados': ['Quebras e Contaminações', 'Produtos vencidos', 'Perdas e Quebras'][:min(3, len(demo_operations))],
        'anos_selecionados': [2023, 2024],
        'tipo_analise': "Quantidade de Movimentações"
    }
    
    df_original = df_loaded
    st.sidebar.warning(f"⚠️ Modo demonstração: {sample_size:,} linhas para teste")

# Verificar se temos dados
if df_original.empty:
    st.warning("""
    ⚠️ Nenhum dado carregado. Por favor:
    1. Carregue um arquivo CSV ou Parquet usando o menu lateral, OU
    2. Selecione 'Usar arquivo local' se tiver arquivos no diretório, OU
    3. Use os dados de demonstração para testar
    """)
    
    with st.expander("📋 Instruções para uso", expanded=True):
        st.markdown("""
        ## Como usar este dashboard:
        
        1. **📤 Upload de arquivo** (menu lateral):
           - Clique em "Browse files"
           - Selecione um arquivo CSV ou Parquet
           - Configure a amostragem se necessário
           - Clique em "Carregar Dados"
        
        2. **📂 Usar arquivo local**:
           - Coloque seus arquivos no mesmo diretório do script
           - Nomes aceitos: `base_tratada.csv` ou qualquer `.parquet`
           - Selecione esta opção no menu lateral
        
        3. **🔄 Dados de demonstração**:
           - Dados de exemplo para teste rápido
           - Ajuste o tamanho do dataset
        """)
    
    st.stop()

# Sidebar para filtros
st.sidebar.header("🔍 Filtros")

# Mostrar informações da amostragem
if use_sample and sample_percentage < 100:
    st.sidebar.info(f"📊 Amostra: {sample_percentage}% dos dados")
if use_sample and max_rows:
    st.sidebar.info(f"📊 Limite: {max_rows:,} linhas")

# Filtro por data
if 'dt_movimento_estoque' in df_original.columns and not df_original['dt_movimento_estoque'].isnull().all():
    min_date = df_original['dt_movimento_estoque'].min().date()
    max_date = df_original['dt_movimento_estoque'].max().date()
    
    # Garantir valores válidos para o date_input
    default_start = st.session_state.filtros.get('data_inicio')
    default_end = st.session_state.filtros.get('data_fim')
    
    if default_start is None or not isinstance(default_start, date):
        default_start = min_date
    if default_end is None or not isinstance(default_end, date):
        default_end = max_date
    
    date_range = st.sidebar.date_input(
        "Período de Movimentação",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        st.session_state.filtros['data_inicio'] = date_range[0]
        st.session_state.filtros['data_fim'] = date_range[1]

# Filtro por estabelecimento
if 'ds_estabelecimento' in df_original.columns:
    estabelecimentos = list(df_original['ds_estabelecimento'].unique())
    estabelecimentos = sort_options_with_all_last(['Todos'] + estabelecimentos)
    
    default_estab = get_safe_default(
        estabelecimentos, 
        st.session_state.filtros.get('estabelecimento', 'Todos'),
        'Todos'
    )
    
    selected_estab = st.sidebar.selectbox(
        "Estabelecimento", 
        estabelecimentos, 
        index=estabelecimentos.index(default_estab) if default_estab in estabelecimentos else 0
    )
    
    st.session_state.filtros['estabelecimento'] = selected_estab

# Filtro por grupo de material
if 'ds_grupo_material' in df_original.columns:
    grupos = list(df_original['ds_grupo_material'].unique())
    grupos = sort_options_with_all_last(['Todos'] + grupos)
    
    default_grupo = get_safe_default(
        grupos,
        st.session_state.filtros.get('grupo_material', 'Todos'),
        'Todos'
    )
    
    selected_grupo = st.sidebar.selectbox(
        "Grupo de Material", 
        grupos, 
        index=grupos.index(default_grupo) if default_grupo in grupos else 0
    )
    
    st.session_state.filtros['grupo_material'] = selected_grupo

# Filtro por operação
if 'ds_operacao' in df_original.columns:
    operacoes = list(df_original['ds_operacao'].unique())
    operacoes = sort_options_with_all_last(['Todos'] + operacoes)
    
    default_op = get_safe_default(
        operacoes,
        st.session_state.filtros.get('operacao', 'Todos'),
        'Todos'
    )
    
    selected_op = st.sidebar.selectbox(
        "Operação", 
        operacoes, 
        index=operacoes.index(default_op) if default_op in operacoes else 0
    )
    
    st.session_state.filtros['operacao'] = selected_op

# Filtro especial para análise de desperdícios
filtrar_desperdicios = st.sidebar.checkbox(
    "🔍 Filtrar apenas desperdícios", 
    value=st.session_state.filtros.get('filtrar_desperdicios', False)
)
st.session_state.filtros['filtrar_desperdicios'] = filtrar_desperdicios

if filtrar_desperdicios:
    desperdicios_lista = [
        "Quebras e Contaminações",
        "Produtos vencidos",
        "Perdas e Quebras",
        "Quebras/Contaminação Med Controlados",
        "Perdas por estabilidade",
        "Medicamentos Controlados Vencidos"
    ]
    
    # Verificar quais desperdícios existem nos dados
    desperdicios_disponiveis = [op for op in desperdicios_lista if op in df_original['ds_operacao'].unique()]
    
    if desperdicios_disponiveis:
        desperdicios_disponiveis = sort_options_with_all_last(['Todos'] + desperdicios_disponiveis)
        
        # Obter valores salvos e filtrar apenas os válidos
        tipos_salvos = st.session_state.filtros.get('tipos_desperdicio', [])
        tipos_validos = [tipo for tipo in tipos_salvos if tipo in desperdicios_disponiveis]
        
        # Se não temos valores válidos, usar alguns padrões
        if not tipos_validos:
            # Remover 'Todos' da lista se existir
            opcoes_sem_todos = [op for op in desperdicios_disponiveis if op != 'Todos']
            if opcoes_sem_todos:
                # Usar até 3 opções disponíveis
                tipos_validos = opcoes_sem_todos[:min(3, len(opcoes_sem_todos))]
        
        desperdicios_selecionados = st.sidebar.multiselect(
            "Tipos de desperdício:",
            desperdicios_disponiveis,
            default=tipos_validos
        )
        
        # Se "Todos" foi selecionado, usar todos os desperdícios
        if 'Todos' in desperdicios_selecionados:
            desperdicios_selecionados = [op for op in desperdicios_disponiveis if op != 'Todos']
        
        st.session_state.filtros['tipos_desperdicio'] = desperdicios_selecionados
    else:
        st.sidebar.info("Nenhum tipo de desperdício encontrado nos dados")
        st.session_state.filtros['tipos_desperdicio'] = []

# Botão para limpar filtros
if st.sidebar.button("🧹 Limpar Filtros"):
    min_date = df_original['dt_movimento_estoque'].min().date() if 'dt_movimento_estoque' in df_original.columns else date.today()
    max_date = df_original['dt_movimento_estoque'].max().date() if 'dt_movimento_estoque' in df_original.columns else date.today()
    
    st.session_state.filtros = {
        'data_inicio': min_date,
        'data_fim': max_date,
        'estabelecimento': 'Todos',
        'grupo_material': 'Todos',
        'operacao': 'Todos',
        'filtrar_desperdicios': False,
        'tipos_desperdicio': []
    }
    st.rerun()

st.sidebar.markdown("---")

# Aplicar filtros ao DataFrame
df_filtrado = aplicar_filtros(df_original, st.session_state.filtros)

# Mostrar estatísticas
st.sidebar.info(f"📊 Total de registros: {len(df_original):,}")
if len(df_original) > 0:
    st.sidebar.info(f"🔍 Após filtros: {len(df_filtrado):,} ({len(df_filtrado)/len(df_original)*100:.1f}%)")

# Layout principal
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Visão Geral", "📊 Análises Detalhadas", "📋 Dados Brutos", "🚨 Análise de Desperdícios", "📚 Sobre"])

# Usar df_filtrado para todas as análises
df = df_filtrado

# Aba 1: Visão Geral
with tab1:
    st.header("Visão Geral dos Dados")
    
    if len(df) > 0:
        info_cols = st.columns([2, 1, 1])
        
        with info_cols[0]:
            if option == "📤 Upload de arquivo" and 'uploaded_file' in locals():
                st.caption(f"📁 Arquivo: {uploaded_file.name}")
            elif option == "📂 Usar arquivo local" and 'file_name' in locals():
                st.caption(f"📁 Arquivo: {file_name}")
            elif option == "🔄 Dados de demonstração":
                st.caption("🔄 Dados de demonstração")
        
        with info_cols[1]:
            if use_sample and sample_percentage < 100:
                st.caption(f"📊 Amostra: {sample_percentage}%")
        
        with info_cols[2]:
            if len(df_original) > 0:
                st.caption(f"🔍 Filtros ativos: {len(df)/len(df_original)*100:.1f}% dos dados")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_consumo = df['qt_consumo'].sum() if 'qt_consumo' in df.columns else 0
            st.metric("Total Consumido", f"{total_consumo:,.0f}")
        
        with col2:
            valor_total = df['vl_consumo'].sum() if 'vl_consumo' in df.columns else 0
            st.metric("Valor Total Consumo (R$)", f"R$ {valor_total:,.2f}")
        
        with col3:
            if 'vl_movimento' in df.columns:
                valor_movimento = df['vl_movimento'].sum()
                st.metric("Valor Total Movimentado (R$)", f"R$ {valor_movimento:,.2f}")
            else:
                st.metric("Materiais Únicos", df['cd_material'].nunique() if 'cd_material' in df.columns else 0)
        
        with col4:
            estabelecimentos_unicos = df['ds_estabelecimento'].nunique() if 'ds_estabelecimento' in df.columns else 0
            st.metric("Estabelecimentos", estabelecimentos_unicos)
        
        # Gráficos principais
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'ds_grupo_material' in df.columns and 'vl_consumo' in df.columns:
                grupo_consumo = df.groupby('ds_grupo_material')['vl_consumo'].sum().reset_index()
                grupo_consumo = grupo_consumo.sort_values('vl_consumo', ascending=False)
                
                fig = px.bar(grupo_consumo, 
                            x='ds_grupo_material', 
                            y='vl_consumo',
                            title="Valor Consumido por Grupo de Material",
                            labels={'ds_grupo_material': 'Grupo de Material', 'vl_consumo': 'Valor Total (R$)'},
                            color='vl_consumo')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            if 'ds_estabelecimento' in df.columns and 'qt_consumo' in df.columns:
                estab_consumo = df.groupby('ds_estabelecimento')['qt_consumo'].sum().reset_index()
                estab_consumo = estab_consumo.sort_values('qt_consumo', ascending=False)
                
                fig = px.pie(estab_consumo, 
                            values='qt_consumo', 
                            names='ds_estabelecimento',
                            title="Distribuição de Consumo por Estabelecimento",
                            hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        
        # Evolução temporal
        if 'dt_movimento_estoque' in df.columns:
            st.subheader("Evolução Temporal")
            
            col_evo1, col_evo2 = st.columns([3, 1])
            
            with col_evo2:
                metrica_evo = st.radio(
                    "Selecione a métrica:",
                    ["Consumo (vl_consumo)", "Movimentação (vl_movimento)"],
                    key="evo_metrica"
                )
            
            df['data'] = df['dt_movimento_estoque'].dt.date
            
            if metrica_evo == "Consumo (vl_consumo)" and 'vl_consumo' in df.columns:
                evolucao = df.groupby('data')['vl_consumo'].sum().reset_index()
                titulo = "Consumo Diário (Valor)"
                y_label = "Valor (R$)"
                coluna = 'vl_consumo'
            elif metrica_evo == "Movimentação (vl_movimento)" and 'vl_movimento' in df.columns:
                evolucao = df.groupby('data')['vl_movimento'].sum().reset_index()
                titulo = "Movimentação Diária (Valor)"
                y_label = "Valor (R$)"
                coluna = 'vl_movimento'
            else:
                evolucao = pd.DataFrame()
            
            if not evolucao.empty:
                fig = px.line(evolucao, 
                             x='data', 
                             y=coluna,
                             title=titulo,
                             markers=True)
                fig.update_layout(xaxis_title="Data", yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum dado disponível após aplicar os filtros. Tente ajustar os filtros.")

# Aba 2: Análises Detalhadas
with tab2:
    st.header("Análises Detalhadas")
    
    if len(df) > 0:
        # Análise por material
        st.subheader("Top 10 Materiais")
        
        col_mat1, col_mat2 = st.columns([3, 1])
        
        with col_mat2:
            metrica_material = st.radio(
                "Selecione a métrica:",
                ["Consumo (vl_consumo)", "Movimentação (vl_movimento)"],
                key="mat_metrica"
            )
        
        if 'ds_material_hospital' in df.columns:
            if metrica_material == "Consumo (vl_consumo)" and 'vl_consumo' in df.columns:
                top_materiais = df.groupby('ds_material_hospital').agg({
                    'vl_consumo': 'sum',
                    'qt_consumo': 'sum'
                }).reset_index()
                
                top_materiais = top_materiais.sort_values('vl_consumo', ascending=False).head(10)
                
                fig = px.bar(top_materiais,
                            x='ds_material_hospital',
                            y='vl_consumo',
                            hover_data=['qt_consumo'],
                            title="Top 10 Materiais por Valor Consumido",
                            labels={'ds_material_hospital': 'Material', 'vl_consumo': 'Valor Total (R$)', 'qt_consumo': 'Quantidade'},
                            color='vl_consumo')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif metrica_material == "Movimentação (vl_movimento)" and 'vl_movimento' in df.columns:
                top_materiais = df.groupby('ds_material_hospital').agg({
                    'vl_movimento': 'sum',
                    'qt_consumo': 'sum'
                }).reset_index()
                
                top_materiais = top_materiais.sort_values('vl_movimento', ascending=False).head(10)
                
                fig = px.bar(top_materiais,
                            x='ds_material_hospital',
                            y='vl_movimento',
                            hover_data=['qt_consumo'],
                            title="Top 10 Materiais por Valor Movimentado",
                            labels={'ds_material_hospital': 'Material', 'vl_movimento': 'Valor Total Movimentado (R$)', 'qt_consumo': 'Quantidade'},
                            color='vl_movimento')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Análise por centro de custo
        st.subheader("Análise por Centro de Custo")
        
        col_cc1, col_cc2 = st.columns([3, 1])
        
        with col_cc2:
            metrica_cc = st.radio(
                "Selecione a métrica:",
                ["Consumo (vl_consumo)", "Movimentação (vl_movimento)"],
                key="cc_metrica"
            )
        
        if 'ds_centro_custo' in df.columns:
            if metrica_cc == "Consumo (vl_consumo)" and 'vl_consumo' in df.columns:
                centro_custo = df.groupby('ds_centro_custo')['vl_consumo'].sum().reset_index()
                centro_custo = centro_custo.sort_values('vl_consumo', ascending=False)
                titulo_cc = "Distribuição por Centro de Custo (Consumo)"
                coluna_cc = 'vl_consumo'
            
            elif metrica_cc == "Movimentação (vl_movimento)" and 'vl_movimento' in df.columns:
                centro_custo = df.groupby('ds_centro_custo')['vl_movimento'].sum().reset_index()
                centro_custo = centro_custo.sort_values('vl_movimento', ascending=False)
                titulo_cc = "Distribuição por Centro de Custo (Movimentação)"
                coluna_cc = 'vl_movimento'
            else:
                centro_custo = pd.DataFrame()
            
            if not centro_custo.empty:
                col_analise1, col_analise2 = st.columns(2)
                
                with col_analise1:
                    fig = px.treemap(centro_custo,
                                    path=['ds_centro_custo'],
                                    values=coluna_cc,
                                    title=titulo_cc,
                                    color=coluna_cc)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_analise2:
                    if 'ds_classe_material' in df.columns:
                        if metrica_cc == "Consumo (vl_consumo)" and 'vl_consumo' in df.columns:
                            classe_material = df.groupby('ds_classe_material')['vl_consumo'].sum().reset_index()
                            classe_material = classe_material.sort_values('vl_consumo', ascending=False).head(10)
                            titulo_classe = "Top Classes de Material por Consumo"
                            coluna_classe = 'vl_consumo'
                        elif metrica_cc == "Movimentação (vl_movimento)" and 'vl_movimento' in df.columns:
                            classe_material = df.groupby('ds_classe_material')['vl_movimento'].sum().reset_index()
                            classe_material = classe_material.sort_values('vl_movimento', ascending=False).head(10)
                            titulo_classe = "Top Classes de Material por Movimentação"
                            coluna_classe = 'vl_movimento'
                        else:
                            classe_material = pd.DataFrame()
                        
                        if not classe_material.empty:
                            fig = px.bar(classe_material,
                                        y='ds_classe_material',
                                        x=coluna_classe,
                                        orientation='h',
                                        title=titulo_classe,
                                        labels={'ds_classe_material': 'Classe', coluna_classe: 'Valor (R$)'},
                                        color=coluna_classe)
                            st.plotly_chart(fig, use_container_width=True)
        
        # Análise estatística
        st.subheader("Estatísticas Descritivas")
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            if 'vl_consumo' in df.columns:
                st.metric("Média Consumo", f"R$ {df['vl_consumo'].mean():,.2f}")
        
        with col_stats2:
            if 'vl_movimento' in df.columns:
                st.metric("Média Movimentação", f"R$ {df['vl_movimento'].mean():,.2f}")
        
        with col_stats3:
            if 'vl_consumo' in df.columns:
                st.metric("Mediana Consumo", f"R$ {df['vl_consumo'].median():,.2f}")
        
        with col_stats4:
            if 'vl_consumo' in df.columns:
                st.metric("Desvio Padrão Consumo", f"R$ {df['vl_consumo'].std():,.2f}")
        
        # Box plots
        col_box1, col_box2 = st.columns(2)
        
        with col_box1:
            if 'vl_consumo' in df.columns:
                fig = px.box(df, y='vl_consumo', title="Distribuição dos Valores de Consumo")
                st.plotly_chart(fig, use_container_width=True)
        
        with col_box2:
            if 'vl_movimento' in df.columns:
                fig = px.box(df, y='vl_movimento', title="Distribuição dos Valores de Movimentação")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum dado disponível após aplicar os filtros. Tente ajustar os filtros.")

# Aba 3: Dados Brutos
with tab3:
    st.header("Dados Brutos")
    
    if len(df) > 0:
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Total de Linhas", f"{len(df):,}")
        
        with col_info2:
            st.metric("Total de Colunas", len(df.columns))
        
        with col_info3:
            if len(df_original) > 0:
                st.metric("Dados Filtrados", f"{len(df)/len(df_original)*100:.1f}%")
        
        # Configuração de visualização
        with st.expander("⚙️ Configuração de Visualização", expanded=True):
            col_view1, col_view2 = st.columns(2)
            
            with col_view1:
                rows_to_show = st.slider(
                    "Linhas para mostrar:",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    help="Ajuste o número de linhas visíveis na tabela"
                )
            
            with col_view2:
                show_full_data = st.checkbox(
                    "Mostrar dados completos (pode ser lento)",
                    value=False,
                    help="Desmarque para melhor performance com datasets grandes"
                )
        
        # Mostrar dados
        if show_full_data:
            st.dataframe(df, use_container_width=True, height=600)
        else:
            st.dataframe(df.head(rows_to_show), use_container_width=True)
            st.caption(f"Mostrando {rows_to_show} de {len(df):,} linhas. Use a configuração acima para ver mais.")
        
        # Estatísticas
        with st.expander("📊 Estatísticas descritivas"):
            if not df.empty:
                st.dataframe(df.describe(), use_container_width=True)
        
        # Download
        st.subheader("Exportar Dados")
        
        col_download1, col_download2, col_download3 = st.columns(3)
        
        with col_download1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV (filtrado)",
                data=csv,
                file_name="dados_filtrados.csv",
                mime="text/csv"
            )
        
        with col_download2:
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Parquet (filtrado)",
                data=buffer,
                file_name="dados_filtrados.parquet",
                mime="application/octet-stream"
            )
        
        with col_download3:
            if not df.empty:
                desc_stats = df.describe()
                csv_stats = desc_stats.to_csv()
                st.download_button(
                    label="📊 Download Estatísticas",
                    data=csv_stats,
                    file_name="estatisticas.csv",
                    mime="text/csv"
                )
        
        # Informações das colunas
        st.subheader("📋 Informações das Colunas")
        
        info_df = pd.DataFrame({
            'Coluna': df.columns,
            'Tipo': df.dtypes.astype(str),
            'Valores Únicos': [df[col].nunique() for col in df.columns],
            'Valores Nulos': [df[col].isnull().sum() for col in df.columns]
        })
        
        st.dataframe(info_df, use_container_width=True)
    else:
        st.warning("Nenhum dado disponível após aplicar os filtros. Tente ajustar os filtros.")

# Aba 4: Análise de Desperdícios
with tab4:
    st.header("🚨 Análise de Desperdícios")
    
    if len(df) > 0:
        # Definir tipos de desperdícios
        desperdicios = [
            "Quebras e Contaminações",
            "Produtos vencidos", 
            "Perdas e Quebras",
            "Quebras/Contaminação Med Controlados",
            "Perdas por estabilidade",
            "Medicamentos Controlados Vencidos"
        ]
        
        # Verificar quais existem nos dados
        desperdicios_existentes = [op for op in desperdicios if op in df['ds_operacao'].unique()]
        
        if not desperdicios_existentes:
            st.warning("⚠️ Nenhum tipo de desperdício encontrado nos dados.")
            st.write("Tipos de desperdício esperados:")
            for desperdicio in desperdicios:
                st.write(f"- {desperdicio}")
        else:
            st.subheader("Configuração da Análise")
            
            col_filtro1, col_filtro2, col_filtro3 = st.columns(3)
            
            with col_filtro1:
                # Obter tipos salvos e filtrar apenas os válidos
                tipos_salvos = st.session_state.analise_desperdicios.get('tipos_selecionados', [])
                tipos_validos = [tipo for tipo in tipos_salvos if tipo in desperdicios_existentes]
                
                # Se não temos valores válidos, usar os primeiros disponíveis
                if not tipos_validos:
                    tipos_validos = desperdicios_existentes[:min(3, len(desperdicios_existentes))]
                
                desperdicios_selecionados = st.multiselect(
                    "Selecione os tipos de desperdício para análise:",
                    desperdicios_existentes,
                    default=tipos_validos
                )
                st.session_state.analise_desperdicios['tipos_selecionados'] = desperdicios_selecionados
            
            with col_filtro2:
                if 'ano' in df.columns:
                    anos_disponiveis = sorted(df['ano'].unique())
                    
                    # Obter anos salvos e filtrar apenas os válidos
                    anos_salvos = st.session_state.analise_desperdicios.get('anos_selecionados', [])
                    anos_validos = [ano for ano in anos_salvos if ano in anos_disponiveis]
                    
                    # Se não temos valores válidos, usar os últimos 2 anos
                    if not anos_validos and len(anos_disponiveis) >= 2:
                        anos_validos = anos_disponiveis[-2:]
                    elif not anos_validos and anos_disponiveis:
                        anos_validos = anos_disponiveis
                    
                    anos_selecionados = st.multiselect(
                        "Selecione os anos para análise:",
                        anos_disponiveis,
                        default=anos_validos
                    )
                    st.session_state.analise_desperdicios['anos_selecionados'] = anos_selecionados
            
            with col_filtro3:
                tipo_analise_salvo = st.session_state.analise_desperdicios.get('tipo_analise', "Quantidade de Movimentações")
                tipo_analise = st.radio(
                    "Tipo de análise:",
                    ["Quantidade de Movimentações", "Valor Movimentado"],
                    index=0 if tipo_analise_salvo == "Quantidade de Movimentações" else 1,
                    help="Escolha entre analisar pelo número de ocorrências ou pelo valor financeiro"
                )
                st.session_state.analise_desperdicios['tipo_analise'] = tipo_analise
            
            # Botão para executar análise
            if st.button("🔍 Executar Análise de Desperdícios", type="primary"):
                if not desperdicios_selecionados:
                    st.error("Selecione pelo menos um tipo de desperdício.")
                    st.stop()
                
                # Filtrar dados
                df_desperdicios = df[
                    (df["ds_operacao"].isin(desperdicios_selecionados))
                ].copy()
                
                if 'ano' in df.columns and anos_selecionados:
                    df_desperdicios = df_desperdicios[df_desperdicios["ano"].isin(anos_selecionados)]
                
                if df_desperdicios.empty:
                    st.error("Nenhum dado encontrado com os filtros selecionados.")
                    st.stop()
                
                df_desperdicios["ds_operacao"] = df_desperdicios["ds_operacao"].cat.remove_unused_categories()
                
                st.success(f"✅ Análise realizada com {len(df_desperdicios):,} registros de desperdício.")
                
                # Métricas
                st.subheader("📊 Métricas de Desperdício")
                
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                
                with col_met1:
                    total_desperdicios = len(df_desperdicios)
                    st.metric("Total de Ocorrências", f"{total_desperdicios:,}")
                
                with col_met2:
                    if tipo_analise == "Valor Movimentado" and 'vl_movimento' in df_desperdicios.columns:
                        valor_total_desperdicio = df_desperdicios['vl_movimento'].sum()
                        st.metric("Valor Total Movimentado", f"R$ {valor_total_desperdicio:,.2f}")
                    else:
                        valor_total_desperdicio = df_desperdicios['vl_consumo'].sum() if 'vl_consumo' in df_desperdicios.columns else 0
                        st.metric("Valor Total Perdido", f"R$ {valor_total_desperdicio:,.2f}")
                
                with col_met3:
                    tipos_desperdicio = df_desperdicios['ds_operacao'].nunique()
                    st.metric("Tipos de Desperdício", tipos_desperdicio)
                
                with col_met4:
                    if 'ds_estabelecimento' in df_desperdicios.columns:
                        locais_afetados = df_desperdicios['ds_estabelecimento'].nunique()
                        st.metric("Locais Afetados", locais_afetados)
                
                # Análise temporal
                st.subheader("📈 Análise Temporal de Desperdícios")
                
                try:
                    if tipo_analise == "Quantidade de Movimentações":
                        freq = df_desperdicios.groupby(['ds_operacao', "ano", "mes"]).size().reset_index(name="movimentacoes")
                        y_col = "movimentacoes"
                        titulo_grafico = "Frequência de Desperdícios por Mês e Ano"
                        y_label = "Número de Ocorrências"
                    else:
                        if 'vl_movimento' in df_desperdicios.columns:
                            freq = df_desperdicios.groupby(['ds_operacao', "ano", "mes"])['vl_movimento'].sum().reset_index()
                            freq = freq.rename(columns={'vl_movimento': 'valor_movimentado'})
                            y_col = "valor_movimentado"
                            titulo_grafico = "Valor Movimentado de Desperdícios por Mês e Ano"
                            y_label = "Valor Movimentado (R$)"
                        else:
                            st.error("Coluna 'vl_movimento' não encontrada para análise de valor.")
                            freq = pd.DataFrame()
                    
                    if not freq.empty:
                        freq["ano_mes"] = freq["ano"].astype(str) + '-' + freq["mes"].astype(str).str.zfill(2)
                        freq = freq.sort_values(by=["ano_mes", "ds_operacao"])
                        
                        col_viz1, col_viz2 = st.columns([3, 1])
                        
                        with col_viz2:
                            viz_option = st.radio(
                                "Tipo de visualização:",
                                ["Plotly (interativo)", "Matplotlib (estático)"],
                                key="viz_desperdicio"
                            )
                        
                        if viz_option == "Matplotlib (estático)":
                            fig, ax = plt.subplots(figsize=(16, 8))
                            
                            for operacao in freq['ds_operacao'].unique():
                                dados_op = freq[freq['ds_operacao'] == operacao]
                                ax.plot(dados_op['ano_mes'], dados_op[y_col], marker='o', label=operacao)
                            
                            ax.set_title(titulo_grafico)
                            ax.set_xlabel("Mês e Ano")
                            ax.set_ylabel(y_label)
                            ax.tick_params(axis='x', rotation=90)
                            ax.grid(True, linestyle='--', alpha=0.6)
                            ax.legend(title="Tipo de Desperdício", bbox_to_anchor=(1.05, 1), loc='upper left')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        else:
                            fig = px.line(freq, 
                                        x='ano_mes', 
                                        y=y_col,
                                        color='ds_operacao',
                                        markers=True,
                                        title=titulo_grafico,
                                        labels={'ano_mes': 'Mês e Ano', y_col: y_label, 'ds_operacao': 'Tipo de Desperdício'})
                            
                            fig.update_layout(
                                xaxis_tickangle=-90,
                                xaxis_title="Mês e Ano",
                                yaxis_title=y_label,
                                hovermode='x unified',
                                legend_title="Tipo de Desperdício",
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Erro ao criar análise temporal: {str(e)}")
                
                # Distribuição por tipo
                st.subheader("🏥 Distribuição por Tipo de Desperdício")
                
                col_viz3, col_viz4 = st.columns(2)
                
                with col_viz3:
                    if tipo_analise == "Quantidade de Movimentações":
                        dist_tipo = df_desperdicios.groupby('ds_operacao').size().reset_index(name='quantidade')
                        fig = px.pie(dist_tipo, 
                                    values='quantidade', 
                                    names='ds_operacao',
                                    title="Distribuição por Tipo de Desperdício (Quantidade)",
                                    hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        if 'vl_movimento' in df_desperdicios.columns:
                            valor_tipo = df_desperdicios.groupby('ds_operacao')['vl_movimento'].sum().reset_index()
                            fig = px.pie(valor_tipo, 
                                        values='vl_movimento', 
                                        names='ds_operacao',
                                        title="Distribuição por Tipo de Desperdício (Valor Movimentado)",
                                        hole=0.4)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col_viz4:
                    if tipo_analise == "Quantidade de Movimentações":
                        quant_tipo = df_desperdicios.groupby('ds_operacao').size().reset_index(name='quantidade')
                        quant_tipo = quant_tipo.sort_values('quantidade', ascending=False)
                        
                        fig = px.bar(quant_tipo,
                                    x='ds_operacao',
                                    y='quantidade',
                                    title="Quantidade de Ocorrências por Tipo de Desperdício",
                                    labels={'ds_operacao': 'Tipo de Desperdício', 'quantidade': 'Número de Ocorrências'},
                                    color='quantidade')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        if 'vl_movimento' in df_desperdicios.columns:
                            valor_tipo = df_desperdicios.groupby('ds_operacao')['vl_movimento'].sum().reset_index()
                            valor_tipo = valor_tipo.sort_values('vl_movimento', ascending=False)
                            
                            fig = px.bar(valor_tipo,
                                        x='ds_operacao',
                                        y='vl_movimento',
                                        title="Valor Movimentado por Tipo de Desperdício",
                                        labels={'ds_operacao': 'Tipo de Desperdício', 'vl_movimento': 'Valor Total (R$)'},
                                        color='vl_movimento')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Análise por estabelecimento
                if 'ds_estabelecimento' in df_desperdicios.columns:
                    st.subheader("🏥 Desperdícios por Estabelecimento")
                    
                    if tipo_analise == "Quantidade de Movimentações":
                        desperdicio_estab = df_desperdicios.groupby(['ds_estabelecimento', 'ds_operacao']).size().reset_index(name='quantidade')
                        y_col_estab = 'quantidade'
                        titulo_estab = "Desperdícios por Estabelecimento e Tipo (Quantidade)"
                        y_label_estab = 'Número de Ocorrências'
                    else:
                        if 'vl_movimento' in df_desperdicios.columns:
                            desperdicio_estab = df_desperdicios.groupby(['ds_estabelecimento', 'ds_operacao'])['vl_movimento'].sum().reset_index()
                            desperdicio_estab = desperdicio_estab.rename(columns={'vl_movimento': 'valor_movimentado'})
                            y_col_estab = 'valor_movimentado'
                            titulo_estab = "Desperdícios por Estabelecimento e Tipo (Valor Movimentado)"
                            y_label_estab = 'Valor Movimentado (R$)'
                        else:
                            desperdicio_estab = pd.DataFrame()
                    
                    if not desperdicio_estab.empty:
                        fig = px.bar(desperdicio_estab,
                                    x='ds_estabelecimento',
                                    y=y_col_estab,
                                    color='ds_operacao',
                                    barmode='stack',
                                    title=titulo_estab,
                                    labels={'ds_estabelecimento': 'Estabelecimento', y_col_estab: y_label_estab, 'ds_operacao': 'Tipo de Desperdício'})
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Tabela detalhada
                st.subheader("📋 Detalhamento dos Desperdícios")
                
                col_det1, col_det2 = st.columns([2, 1])
                
                with col_det1:
                    if tipo_analise == "Quantidade de Movimentações":
                        resumo_desperdicios = df_desperdicios.groupby('ds_operacao').agg({
                            'qt_consumo': 'sum' if 'qt_consumo' in df_desperdicios.columns else None,
                            'vl_consumo': 'sum' if 'vl_consumo' in df_desperdicios.columns else None,
                            'cd_material': 'nunique'
                        }).reset_index()
                        
                        resumo_desperdicios.columns = ['Tipo de Desperdício', 'Quantidade Total', 'Valor Total (R$)', 'Materiais Únicos']
                        if 'Valor Total (R$)' in resumo_desperdicios.columns:
                            resumo_desperdicios['Valor Total (R$)'] = resumo_desperdicios['Valor Total (R$)'].round(2)
                    else:
                        if 'vl_movimento' in df_desperdicios.columns:
                            resumo_desperdicios = df_desperdicios.groupby('ds_operacao').agg({
                                'vl_movimento': 'sum',
                                'cd_material': 'nunique',
                                'qt_consumo': 'sum' if 'qt_consumo' in df_desperdicios.columns else None
                            }).reset_index()
                            
                            resumo_desperdicios.columns = ['Tipo de Desperdício', 'Valor Movimentado (R$)', 'Materiais Únicos', 'Quantidade Total']
                            resumo_desperdicios['Valor Movimentado (R$)'] = resumo_desperdicios['Valor Movimentado (R$)'].round(2)
                    
                    if 'resumo_desperdicios' in locals() and not resumo_desperdicios.empty:
                        coluna_ordenacao = 'Valor Total (R$)' if tipo_analise == "Quantidade de Movimentações" else 'Valor Movimentado (R$)'
                        st.dataframe(resumo_desperdicios.sort_values(coluna_ordenacao, ascending=False), use_container_width=True)
                
                with col_det2:
                    if tipo_analise == "Quantidade de Movimentações":
                        if total_desperdicios > 0:
                            media_valor = valor_total_desperdicio/total_desperdicios if valor_total_desperdicio > 0 else 0
                            st.metric("Média por Ocorrência", f"R$ {media_valor:,.2f}")
                        
                        if 'vl_consumo' in df_desperdicios.columns:
                            st.metric("Máxima Ocorrência", f"R$ {df_desperdicios['vl_consumo'].max():,.2f}")
                    else:
                        if total_desperdicios > 0 and 'vl_movimento' in df_desperdicios.columns:
                            media_movimento = df_desperdicios['vl_movimento'].sum()/total_desperdicios
                            st.metric("Média por Movimentação", f"R$ {media_movimento:,.2f}")
                        
                        if 'vl_movimento' in df_desperdicios.columns:
                            st.metric("Máxima Movimentação", f"R$ {df_desperdicios['vl_movimento'].max():,.2f}")
                    
                    if 'freq' in locals() and not freq.empty:
                        st.metric("Mês com Maior Valor", freq.loc[freq[y_col].idxmax(), 'ano_mes'])
                
                # Recomendações
                with st.expander("💡 Recomendações para Redução de Desperdícios", expanded=True):
                    st.markdown("""
                    ### Baseado na análise realizada:
                    
                    1. **Gestão de Estoque:**
                       - Implementar sistema FIFO (First In, First Out)
                       - Revisar níveis de estoque mínimo/máximo
                       - Monitorar datas de validade regularmente
                    
                    2. **Controle de Qualidade:**
                       - Treinamento de equipe sobre manipulação adequada
                       - Protocolos para evitar contaminação
                       - Sistema de identificação de produtos próximos ao vencimento
                    
                    3. **Medicamentos Controlados:**
                       - Auditoria regular dos controles
                       - Sistema de alerta para vencimentos
                       - Gestão específica por lote
                    
                    4. **Monitoramento Contínuo:**
                       - Dashboard de acompanhamento mensal
                       - Metas de redução de desperdício
                       - Análise de causas raízes
                    """)
                
                # Exportação
                st.subheader("📥 Exportar Análise de Desperdícios")
                
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    csv_desperdicios = df_desperdicios.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Dados Filtrados (CSV)",
                        data=csv_desperdicios,
                        file_name="desperdicios_analisados.csv",
                        mime="text/csv"
                    )
                
                with col_exp2:
                    buffer = io.BytesIO()
                    df_desperdicios.to_parquet(buffer, index=False)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Dados Filtrados (Parquet)",
                        data=buffer,
                        file_name="desperdicios_analisados.parquet",
                        mime="application/octet-stream"
                    )
            
            else:
                st.info("👆 Configure os filtros acima e clique em 'Executar Análise de Desperdícios' para iniciar.")
    else:
        st.warning("Nenhum dado disponível após aplicar os filtros. Tente ajustar os filtros.")

# Aba 5: Sobre
with tab5:
    st.header("📚 Sobre este Projeto")
    
    st.markdown("""
    ## 🎓 Projeto de Ciência de Dados - Análise Hospitalar
    
    **Correções Implementadas:**
    
    ### ✅ **Sistema de Verificação de Valores Padrão**
    - Função `get_safe_default()` para garantir valores válidos em selectboxes
    - Filtragem de valores salvos antes de usá-los como padrão
    - Fallbacks inteligentes quando valores não estão disponíveis
    
    ### ✅ **Tratamento Robustecido de Multiselect**
    - Valores padrão são sempre filtrados para incluir apenas opções disponíveis
    - Reset adequado do estado quando dados mudam
    - Validação em tempo real de valores
    
    ### ✅ **Sistema de Estado Aprimorado**
    - Inicialização adequada de todos os estados
    - Reset completo ao carregar novos dados
    - Persistência segura entre interações
    
    ### ✅ **Tratamento de Erros Abrangente**
    - Verificação de dados antes de cada análise
    - Mensagens de erro claras e informativas
    - Fallbacks para todos os cenários possíveis
    
    **Funcionalidades Principais:**
    1. **📈 Visão Geral:** Métricas e gráficos com dados filtrados
    2. **📊 Análises Detalhadas:** Top materiais, análise estatística
    3. **📋 Dados Brutos:** Visualização completa com exportação
    4. **🚨 Análise de Desperdícios:** Duas perspectivas (quantidade vs valor)
    5. **📚 Sobre:** Documentação do projeto
    
    **Vantagens do Sistema Corrigido:**
    - ✅ Sem erros de valores padrão inválidos
    - ✅ Filtros persistentes e funcionais
    - ✅ Interface responsiva e fluida
    - ✅ Performance otimizada
    - ✅ Código robusto e manutenível
    """)

# Rodapé
st.markdown("---")
st.caption("""
Desenvolvido para a disciplina de Ciência de Dados | Dashboard Analítico Hospitalar
🔄 Filtros persistentes | ⚡ Performance otimizada | 🏥 Análise especializada
""")

# Limpar memória
gc.collect()