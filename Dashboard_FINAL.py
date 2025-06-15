import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import json
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import pour le chatbot
try:
    from huggingface_hub import InferenceClient
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False
    st.warning("⚠️ huggingface_hub n'est pas installé. Installez-le avec: pip install huggingface_hub")

# Configuration de la page
st.set_page_config(
    page_title="🏠 Airbnb NYC Analytics Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Styles CSS améliorés avec onglets stylisés
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0e101d 0%, #1a1d3a 100%);
        font-family: 'Montserrat', sans-serif;
    }
    
    /* ONGLETS STYLISÉS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 30px;
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 15px;
        color: #fff;
        font-weight: 600;
        font-size: 16px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stTabs [data-baseweb="tab"]:hover:before {
        left: 100%;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 217, 239, 0.3);
        border-color: #66d9ef;
        background: linear-gradient(135deg, #66d9ef 0%, #4facfe 100%);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #66d9ef 0%, #4facfe 100%) !important;
        color: #fff !important;
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 217, 239, 0.4);
        border-color: #66d9ef !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255,255,255,0.02);
        padding: 30px;
        border-radius: 20px;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    /* Header Principal */
    .main-header {
        font-size: 4rem;
        color: #66d9ef;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 700;
        text-shadow: 0 0 20px #66d9ef;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #66d9ef, 0 0 30px #66d9ef; }
        to { text-shadow: 0 0 30px #66d9ef, 0 0 40px #66d9ef; }
    }

    /* Carte Métrique Futuriste */
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        color: #fff;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 217, 239, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 217, 239, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover:before {
        opacity: 1;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 217, 239, 0.3);
        border-color: #66d9ef;
    }

    /* Boîte d'Insight avec Effet de Verre */
    .insight-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102, 217, 239, 0.3);
        padding: 2rem;
        margin: 2rem 0;
        border-radius: 20px;
        color: #fff;
        font-size: 1.2rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        position: relative;
    }
    
    .insight-box:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #66d9ef, #4facfe, #66d9ef);
        border-radius: 20px 20px 0 0;
    }
    
    .insight-box strong {
        color: #66d9ef;
        font-weight: 600;
    }

    /* Chat Container */
    .chat-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 2rem;
        border-radius: 25px;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(102, 217, 239, 0.2);
        backdrop-filter: blur(10px);
    }

    .chatbot-header {
        background: linear-gradient(135deg, #66d9ef 0%, #4facfe 100%);
        color: #fff;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 10px 30px rgba(102, 217, 239, 0.3);
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        max-width: 80%;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .user-message {
        background: linear-gradient(135deg, #66d9ef 0%, #4facfe 100%);
        color: #fff;
        margin-left: auto;
    }

    .bot-message {
        background: rgba(255,255,255,0.1);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a1d3a 0%, #2d3748 100%);
    }
    
    /* Animations subtiles */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card, .insight-box, .chat-container {
        animation: slideIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Configuration du chatbot
class AirbnbChatbot:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if api_key and CHATBOT_AVAILABLE:
            try:
                self.client = InferenceClient(provider="nebius", api_key=api_key)
            except Exception as e:
                st.error(f"Erreur d'initialisation du chatbot: {e}")
    
    def get_system_prompt(self, df_stats):
        """Génère le prompt système avec les statistiques des données"""
        return f"""Tu es un assistant expert en analyse de données Airbnb pour New York City. 
        Tu aides les utilisateurs à comprendre et analyser les données de location Airbnb.
        
        DONNÉES ACTUELLES:
        - Total des annonces: {len(df_stats):,}
        - Prix moyen: ${df_stats['price'].mean():.2f}
        - Prix médian: ${df_stats['price'].median():.2f}
        - Quartiers: {', '.join(df_stats['neighbourhood_group'].unique())}
        - Types de logement: {', '.join(df_stats['room_type'].unique())}
        - Fourchette de prix: ${df_stats['price'].min():.0f} - ${df_stats['price'].max():.0f}
        
        INSTRUCTIONS:
        - Réponds en français de manière claire et concise
        - Utilise les données fournies pour donner des insights précis
        - Propose des analyses et visualisations quand c'est pertinent
        - Sois professionnel mais accessible
        - Si on te demande des analyses spécifiques, guide l'utilisateur vers les bonnes sections du dashboard
        - Limite tes réponses à 3-4 phrases maximum pour rester concis
        """
    
    def chat(self, message, conversation_history, df_stats):
        """Génère une réponse du chatbot"""
        if not self.client:
            return "❌ Chatbot non disponible. Veuillez configurer votre clé API Nebius."
        
        try:
            # Préparer les messages
            messages = [
                {"role": "system", "content": self.get_system_prompt(df_stats)}
            ]
            
            # Ajouter l'historique (garder les 6 derniers messages)
            for msg in conversation_history[-6:]:
                messages.append(msg)
            
            # Ajouter le message actuel
            messages.append({"role": "user", "content": message})
            
            # Appel à l'API
            completion = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            
            return completion.choices[0].message['content'].strip()
            
        except Exception as e:
            return f"❌ Erreur lors de la génération de la réponse: {str(e)}"

# Palette de couleurs
PRIMARY_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#5A6C57'
}

EXTENDED_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5A6C57',
                   '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge et nettoie les données Airbnb"""
    # Simuler le chargement depuis Kaggle
    # En production, vous remplacerez ceci par votre code de chargement réel
    
    # Ici, vous devriez avoir votre vraie logique de chargement
    # df = pd.read_csv('path_to_your_data/AB_NYC_2019.csv')
    
    # Pour la démo, je crée des données synthétiques basées sur votre analyse
    np.random.seed(42)
    n_samples = 10000
    
    neighbourhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    room_types = ['Entire home/apt', 'Private room', 'Shared room']
    
    # Génération de données synthétiques réalistes
    df = pd.DataFrame({
        'neighbourhood_group': np.random.choice(neighbourhoods, n_samples, 
                                              p=[0.35, 0.35, 0.2, 0.08, 0.02]),
        'room_type': np.random.choice(room_types, n_samples, p=[0.6, 0.35, 0.05]),
        'price': np.random.lognormal(4.5, 0.8, n_samples).astype(int),
        'minimum_nights': np.random.choice([1, 2, 3, 4, 5, 7, 30], n_samples, 
                                         p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]),
        'number_of_reviews': np.random.poisson(20, n_samples),
        'reviews_per_month': np.random.poisson(2, n_samples),
        'calculated_host_listings_count': np.random.choice(range(1, 50), n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'latitude': np.random.normal(40.7128, 0.1, n_samples),
        'longitude': np.random.normal(-74.0060, 0.1, n_samples)
    })
    
    # Ajuster les prix selon le type de logement et quartier
    price_multipliers = {
        'Manhattan': 1.5, 'Brooklyn': 1.0, 'Queens': 0.8, 
        'Bronx': 0.6, 'Staten Island': 0.7
    }
    room_multipliers = {
        'Entire home/apt': 1.0, 'Private room': 0.6, 'Shared room': 0.4
    }
    
    for idx, row in df.iterrows():
        df.loc[idx, 'price'] = int(row['price'] * 
                                  price_multipliers[row['neighbourhood_group']] * 
                                  room_multipliers[row['room_type']])
    
    # Nettoyer les prix impossibles
    df = df[df['price'] > 0]
    df = df[df['price'] <= 1000]  # Filtrer les outliers extrêmes
    
    return df

# Fonction pour préparer les modèles ML
@st.cache_data
def prepare_models(df):
    """Prépare et entraîne les modèles de ML"""
    
    # Préparation des données
    features = ['neighbourhood_group', 'room_type', 'minimum_nights', 
                'number_of_reviews', 'reviews_per_month', 
                'calculated_host_listings_count', 'availability_365']
    
    df_model = df.copy()
    
    # Encodage des variables catégorielles
    le_dict = {}
    for col in ['neighbourhood_group', 'room_type']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le
    
    X = df_model[features]
    y = df_model['price']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement des modèles
    rf_model = RandomForestRegressor(max_leaf_nodes=250, random_state=42)
    rf_model.fit(X_train, y_train)
    
    dt_model = DecisionTreeRegressor(max_leaf_nodes=250, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Évaluation
    rf_pred = rf_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    
    rf_mae = mean_absolute_error(y_test, rf_pred)
    dt_mae = mean_absolute_error(y_test, dt_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    dt_r2 = r2_score(y_test, dt_pred)
    
    return {
        'rf_model': rf_model,
        'dt_model': dt_model,
        'le_dict': le_dict,
        'X_test': X_test,
        'y_test': y_test,
        'rf_mae': rf_mae,
        'dt_mae': dt_mae,
        'rf_r2': rf_r2,
        'dt_r2': dt_r2,
        'features': features
    }

# Initialisation du chatbot
@st.cache_resource
def init_chatbot():
    """Initialize the chatbot"""
    # Vous devez remplacer cette clé par votre vraie clé API Nebius
    api_key = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExMDIxMzMxODk5MTI3Nzk1NzIxNiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNzQ4ODcwMSwidXVpZCI6IjJjNGQ1Nzg4LTY4MzQtNGI2My1hMmM5LWUxM2U5NjFjYjY2MiIsIm5hbWUiOiJDaGF0Qm90IiwiZXhwaXJlc19hdCI6IjIwMzAtMDYtMTJUMDk6NTg6MjErMDAwMCJ9.kmiGNSwJ1Zyphf9Iz0fbxPb7rlHj5Gwo4FzT9MmIWUg"  
    return AirbnbChatbot(api_key)

# Chargement des données
df = load_data()
models_data = prepare_models(df)

# Initialisation du chatbot
chatbot = init_chatbot()

# Initialiser l'historique de conversation dans la session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# HEADER PRINCIPAL
st.markdown('<h1 class="main-header">🏠 Airbnb NYC Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analyse interactive des données Airbnb à New York - Projet Data Science</p>', unsafe_allow_html=True)

# SIDEBAR - Configuration du chatbot et filtres
st.sidebar.markdown("## 🤖 Assistant IA")


# Bouton pour effacer l'historique
if st.sidebar.button("🗑️ Effacer l'historique"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("## 🎛️ Filtres Interactifs")

# Filtres
selected_neighbourhoods = st.sidebar.multiselect(
    "Quartiers", 
    options=df['neighbourhood_group'].unique(),
    default=df['neighbourhood_group'].unique()
)

selected_room_types = st.sidebar.multiselect(
    "Types de logement",
    options=df['room_type'].unique(),
    default=df['room_type'].unique()
)

price_range = st.sidebar.slider(
    "Fourchette de prix ($)",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), min(500, int(df['price'].max()))),
    step=10
)

# Filtrer les données
filtered_df = df[
    (df['neighbourhood_group'].isin(selected_neighbourhoods)) &
    (df['room_type'].isin(selected_room_types)) &
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1])
]

# ...existing code...

# SECTION CHATBOT
st.markdown("## 🤖 Assistant IA Airbnb")

# Conteneur du chatbot
with st.container():
    st.markdown('<div class="chatbot-header">💬 Posez vos questions sur les données Airbnb NYC</div>', unsafe_allow_html=True)
    
    chat_container = st.container()
    
    # Afficher l'historique des conversations
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-message bot-message">🤖 Bonjour! Je suis votre assistant IA pour l\'analyse des données Airbnb NYC. Posez-moi vos questions!</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Ajout du bouton pour effacer la dernière réponse
    if st.button("🧹 Effacer la dernière réponse"):
        if len(st.session_state.chat_history) >= 2:
            st.session_state.chat_history = st.session_state.chat_history[:-2]
            st.rerun()
        elif len(st.session_state.chat_history) == 1:
            st.session_state.chat_history = []
            st.rerun()

    # Input pour les messages
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Votre question:",
            placeholder="Ex: Quel est le quartier le plus cher? Quels sont les insights sur les prix?",
            key="chat_input"
        )
    with col2:
        send_button = st.button("📤 Envoyer", type="primary")

    # Correction de la boucle infinie : utiliser une clé temporaire pour ne traiter qu'une fois
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""

    if (send_button or user_input) and user_input.strip() and user_input != st.session_state.last_user_input:
        if chatbot.client:
            # Ajouter le message utilisateur
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.last_user_input = user_input

            # Générer la réponse
            with st.spinner("🤔 Réflexion en cours..."):
                response = chatbot.chat(user_input, st.session_state.chat_history[:-1], filtered_df)

            # Ajouter la réponse du bot
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            st.rerun()
        else:
            st.error("❌ Veuillez configurer votre clé API Nebius dans la sidebar")

# ...existing code...

# Suggestions de questions
st.markdown("### 💡 Questions suggérées:")
questions_suggestions = [
    "Quel est le quartier le plus rentable?",
    "Comment les prix varient-ils selon le type de logement?",
    "Quels sont les facteurs qui influencent le plus les prix?",
    "Quelle est la saisonnalité des prix?",
    "Comment optimiser le prix de ma location?"
]

cols = st.columns(len(questions_suggestions))
for i, question in enumerate(questions_suggestions):
    with cols[i]:
        if st.button(question, key=f"suggestion_{i}"):
            if chatbot.client:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("🤔 Réflexion en cours..."):
                    response = chatbot.chat(question, st.session_state.chat_history[:-1], filtered_df)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.error("❌ Veuillez configurer votre clé API Nebius")

# MÉTRIQUES PRINCIPALES
st.markdown("## 📊 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Annonces", f"{len(filtered_df):,}")

with col2:
    st.metric("Prix Moyen", f"${filtered_df['price'].mean():.0f}")

with col3:
    st.metric("Prix Médian", f"${filtered_df['price'].median():.0f}")

with col4:
    st.metric("Hôtes Uniques", f"{filtered_df['calculated_host_listings_count'].sum():,}")

# VISUALISATIONS PRINCIPALES
st.markdown("## 🎨 Visualisations Interactives")

# Onglets pour organiser les visualisations avec style amélioré
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Vue d'ensemble", 
    "🗺️ Analyse Géographique", 
    "💰 Analyse des Prix", 
    "🤖 Machine Learning"
])

with tab1:
    st.markdown('<h3 class="sub-header">Vue d\'ensemble des données</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition des types de logement
        room_counts = filtered_df['room_type'].value_counts()
        fig_pie = px.pie(
            values=room_counts.values,
            names=room_counts.index,
            title="🏠 Répartition des types de logement",
            color_discrete_sequence=EXTENDED_PALETTE
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Popularité par quartier
        neigh_counts = filtered_df['neighbourhood_group'].value_counts()
        fig_bar = px.bar(
            x=neigh_counts.index,
            y=neigh_counts.values,
            title="🏙️ Popularité par quartier",
            color=neigh_counts.values,
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Distribution des prix
    fig_hist = px.histogram(
        filtered_df,
        x='price',
        nbins=50,
        title="💰 Distribution des prix",
        color_discrete_sequence=[PRIMARY_COLORS['primary']]
    )
    fig_hist.add_vline(x=filtered_df['price'].mean(), line_dash="dash", 
                      annotation_text=f"Moyenne: ${filtered_df['price'].mean():.0f}")
    fig_hist.add_vline(x=filtered_df['price'].median(), line_dash="dash", 
                      annotation_text=f"Médiane: ${filtered_df['price'].median():.0f}")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.markdown('<h3 class="sub-header">Analyse Géographique</h3>', unsafe_allow_html=True)
    
    # Carte de densité des prix
    fig_density = px.density_mapbox(
        filtered_df,
        lat='latitude',
        lon='longitude',
        z='price',
        radius=10,
        center=dict(lat=40.7128, lon=-74.0060),
        zoom=10,
        mapbox_style="carto-positron",
        color_continuous_scale="Viridis",
        title="🗺️ Densité des prix Airbnb à New York"
    )
    fig_density.update_layout(height=600)
    st.plotly_chart(fig_density, use_container_width=True)
    
    # Scatter plot géographique
    fig_scatter = px.scatter_mapbox(
        filtered_df.sample(min(1000, len(filtered_df))),
        lat='latitude',
        lon='longitude',
        color='price',
        size='number_of_reviews',
        hover_data=['neighbourhood_group', 'room_type'],
        mapbox_style="carto-positron",
        title="📍 Répartition des annonces (échantillon)",
        color_continuous_scale="Plasma"
    )
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown('<h3 class="sub-header">Analyse Détaillée des Prix</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prix moyen par quartier et type de logement
        price_heatmap = filtered_df.groupby(['neighbourhood_group', 'room_type'])['price'].mean().reset_index()
        fig_heatmap = px.density_heatmap(
            price_heatmap,
            x="room_type",
            y="neighbourhood_group", 
            z="price",
            title="🔥 Prix moyen par quartier et type",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Box plot des prix par quartier
        fig_box = px.box(
            filtered_df,
            x="neighbourhood_group",
            y="price",
            title="📦 Distribution des prix par quartier",
            color="neighbourhood_group",
            color_discrete_sequence=EXTENDED_PALETTE
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Corrélation entre variables
    st.markdown("### 🔗 Matrice de Corrélation")
    numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                   'calculated_host_listings_count', 'availability_365']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Matrice de corrélation des variables numériques",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.markdown('<h3 class="sub-header">Modèles de Prédiction des Prix</h3>', unsafe_allow_html=True)
    
    # Métriques des modèles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌲 Random Forest")
        st.metric("MAE", f"${models_data['rf_mae']:.2f}")
        st.metric("R²", f"{models_data['rf_r2']:.3f}")
    
    with col2:
        st.markdown("### 🌳 Decision Tree")
        st.metric("MAE", f"${models_data['dt_mae']:.2f}")
        st.metric("R²", f"{models_data['dt_r2']:.3f}")
    
    # Importance des features
    if hasattr(models_data['rf_model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': models_data['features'],
            'importance': models_data['rf_model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="📊 Importance des Variables (Random Forest)",
            color='importance',
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prédicteur interactif
    st.markdown("### 🔮 Prédicteur de Prix Interactif")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_neighbourhood = st.selectbox("Quartier", df['neighbourhood_group'].unique())
        pred_room_type = st.selectbox("Type de logement", df['room_type'].unique())
        pred_min_nights = st.slider("Nuits minimum", 1, 30, 2)
    
    with col2:
        pred_reviews = st.slider("Nombre de reviews", 0, 200, 20)
        pred_reviews_month = st.slider("Reviews par mois", 0, 20, 2)
        pred_host_listings = st.slider("Annonces de l'hôte", 1, 50, 3)
    
    with col3:
        pred_availability = st.slider("Disponibilité (jours/an)", 0, 365, 200)
        
        if st.button("🎯 Prédire le Prix", type="primary"):
            # Préparer les données pour la prédiction
            pred_data = pd.DataFrame({
                'neighbourhood_group': [pred_neighbourhood],
                'room_type': [pred_room_type],
                'minimum_nights': [pred_min_nights],
                'number_of_reviews': [pred_reviews],
                'reviews_per_month': [pred_reviews_month],
                'calculated_host_listings_count': [pred_host_listings],
                'availability_365': [pred_availability]
            })
            
            # Encoder les variables catégorielles
            for col in ['neighbourhood_group', 'room_type']:
                pred_data[col] = models_data['le_dict'][col].transform(pred_data[col])
            
            # Prédiction
            rf_prediction = models_data['rf_model'].predict(pred_data)[0]
            dt_prediction = models_data['dt_model'].predict(pred_data)[0]
            
            st.success(f"🌲 Random Forest: **${rf_prediction:.0f}**")
            st.success(f"🌳 Decision Tree: **${dt_prediction:.0f}**")

# INSIGHTS ET RECOMMANDATIONS
st.markdown("## 💡 Insights Clés")

insights = [
    "🏙️ **Manhattan** reste le quartier le plus cher avec une moyenne de prix significativement plus élevée",
    "🏠 Les **appartements entiers** représentent environ 60% des annonces et génèrent les prix les plus élevés",
    "📊 Une forte corrélation existe entre le **type de logement** et le **prix**",
    "🎯 Le modèle **Random Forest** obtient de meilleures performances que le Decision Tree",
    "⭐ Le **nombre de reviews** est un indicateur important de la popularité des annonces"
]

for insight in insights:
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>👥 Projet réalisé par Carolina HENAO URIBE & Marwane ZAIM SASSI</h4>
    <p>🔗 Dataset: <a href="https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data" target="_blank">Kaggle - NYC Airbnb Open Data</a></p>
    <p>🛠️ Développé avec Streamlit, Plotly et Scikit-learn</p>
</div>
""", unsafe_allow_html=True)