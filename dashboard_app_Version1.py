# %%
#%pip install kagglehub pandas streamlit

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("dgomonov/new-york-city-airbnb-open-data")

print("Path to dataset files:", path)


# %%

import pandas as pd

df = pd.read_csv(f'{path}/AB_NYC_2019.csv')


df = df.dropna(axis=0)

# Convertir reviews per month en int64
df['reviews_per_month'] = df['reviews_per_month'].astype(int)

# Convertir la colonne 'last_review' en type datetime
df['last_review'] = pd.to_datetime(df['last_review'])

# Traiter le price = 0 qui serait impossible dans la vraie vie
df = df[df['price'] > 0]





# %%
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit app to explore the New York Airbnb dataset
st.title("New York Airbnb Dataset Explorer")
st.write("This dataset contains information about the New York City Airbnb listings, including their prices, locations, and various attributes.")
st.write("You can explore the dataset below:")
st.dataframe(df)
st.write("### Dataset Information")
st.write("This dataset includes various personality traits and behaviors of individuals, which can be used for analysis and modeling.")



st.write("### Répartition des types de logement")

# Dropdown pour filtrer par groupe de quartier
selected_group = st.selectbox(
    "Choisir un groupe de quartier (neighbourhood_group)",
    options=["Tous"] + sorted(df['neighbourhood_group'].unique())
)

# Filtrage du DataFrame
if selected_group != "Tous":
    filtered_df = df[df['neighbourhood_group'] == selected_group]
else:
    filtered_df = df

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=filtered_df, x='room_type', order=filtered_df['room_type'].value_counts().index, ax=ax)
ax.set_title("Répartition des types de logement")
ax.set_xlabel("Type de chambre")
ax.set_ylabel("Nombre d'annonces")
st.pyplot(fig)


# --- Quartiers les plus populaires ---
st.write("### Quartiers les plus populaires")
fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.countplot(data=df, x='neighbourhood_group', order=df['neighbourhood_group'].value_counts().index, ax=ax1)
ax1.set_title("Quartiers les plus populaires")
ax1.set_xlabel("Quartier")
ax1.set_ylabel("Nombre d'annonces")
st.pyplot(fig1)


# --- Matrice de corrélation ---
st.write("### Matrice de corrélation")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
ax2.set_title("Matrice de corrélation")
st.pyplot(fig2)

# --- Distribution des prix ---
st.write("### Distribution des prix limite à 3000$")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.histplot(df[df['price'] < 3000]['price'], bins=50, kde=True, ax=ax3)
ax3.set_title("Distribution des prix < $3000")
ax3.set_xlabel("Prix ($)")
ax3.set_ylabel("Nombre d'annonces")
mean_price = df['price'].mean()
median_price = df['price'].median()
ax3.axvline(mean_price, color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: ${mean_price:.2f}')
ax3.axvline(median_price, color='green', linestyle='dashed', linewidth=1, label=f'Médiane: ${median_price:.2f}')
ax3.legend()
st.pyplot(fig3)

# --- Top hôtes par nombre de reviews ---
st.write("### Top 20 des hôtes avec le plus d'avis")
top_hosts = df.groupby(['host_id', 'host_name'])['number_of_reviews'].sum().sort_values(ascending=False).head(20)
top_hosts_df = top_hosts.reset_index()
fig4, ax4 = plt.subplots(figsize=(12, 6))
sns.barplot(x='number_of_reviews', y='host_name', data=top_hosts_df, palette='viridis', ax=ax4)
ax4.set_title("Top 20 des hôtes (host_id) avec le plus d'avis")
ax4.set_xlabel("Nombre total de reviews")
ax4.set_ylabel("Nom de l'hôte")
st.pyplot(fig4)

# --- Pairplot interactif (optionnel car lourd) ---
st.write("### Visualisation croisee des variables numériques (Pairplot)")
if st.checkbox("Afficher le Pairplot (peut être long)"):
    numerical_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                      'calculated_host_listings_count', 'availability_365']
    fig5 = sns.pairplot(df[numerical_cols])
    st.pyplot(fig5)

st.write("---")
st.write("Dashboard interactif construit avec Streamlit et Seaborn")
