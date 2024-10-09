import streamlit as st
import pymongo
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
import time

# Descargar stopwords de NLTK
nltk.download('stopwords')

# Configurar la página
st.set_page_config(page_title="Real-Time Analysis of Candidate Posts", layout="wide", initial_sidebar_state="expanded")

# Configuración de colores
candidate_colors = {
    "Kamala Harris": "#1f77b4",  # Azul
    "Donald Trump": "#d62728"    # Rojo
}

# Connect to MongoDB using st.secrets
mongo_uri = st.secrets["mongo"]["uri"]
client = MongoClient(mongo_uri)
db = client['elections']
collection = db['monitoring']

# Función para cargar datos desde MongoDB
@st.cache_data(ttl=60)
def get_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    
    # Convertir la columna de fecha a datetime con formato explícito
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['date'])

    # Unificar nombres de candidatos
    df['candidates'] = df['candidates'].apply(lambda x: 'Kamala Harris' if any(c in ['Kamala', 'Kamala Harris'] for c in x) 
                                              else 'Donald Trump' if any(c in ['Trump', 'Donald Trump'] for c in x) 
                                              else x)
    return df

# Función para procesar y extraer texto de comentarios
def process_comments(df):
    comments_list = []
    for item in df['comments']:
        if isinstance(item, list):
            # Extraer el texto de cada comentario dentro de la lista
            for comment in item:
                if isinstance(comment, dict) and 'comment_body' in comment:
                    comments_list.append(comment['comment_body'])
        elif isinstance(item, dict):
            # Si el comentario es un diccionario y tiene 'comment_body', extraer el texto
            if 'comment_body' in item:
                comments_list.append(item['comment_body'])
        elif isinstance(item, str):
            # Si ya es un string, agregarlo directamente
            comments_list.append(item)
    return comments_list

# Cargar datos
df = get_data()
last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

# Mostrar última actualización
st.sidebar.markdown(f"**Última actualización:** {last_updated}")
st.sidebar.button("Actualizar ahora")  # Botón para forzar actualización manual

# Filtros de fecha y candidato
st.sidebar.header("Filtros")
start_date = st.sidebar.date_input("Fecha de inicio", min_value=df['date'].min(), value=df['date'].min())
end_date = st.sidebar.date_input("Fecha de fin", max_value=df['date'].max(), value=df['date'].max())
selected_candidate = st.sidebar.selectbox("Selecciona un candidato", ["Todos", "Kamala Harris", "Donald Trump"])
df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]

if selected_candidate != "Todos":
    df = df[df['candidates'] == selected_candidate]

# Título del Dashboard
st.title("Real-Time Analysis of Candidate Posts")
st.write("This dashboard provides real-time insights into social media sentiment, interactions, and trends regarding political candidates.")

# Análisis de Sentimientos y Tendencias
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiment Over Time", "Total Interactions", "Word Cloud", "Most Influential Posts", "Sentiment Distribution"])

with tab1:
    st.subheader("Sentiment Comparison Over Time")
    sentiment_counts = df.groupby([df['date'].dt.date, 'candidates', 'sentiment']).size().reset_index(name='counts')
    fig = px.line(sentiment_counts, x='date', y='counts', color='candidates', line_dash='sentiment', color_discrete_map=candidate_colors)
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig)

with tab2:
    st.subheader("Total Interactions")
    interactions = df.groupby('candidates').agg({'upvotes': 'sum', 'comments': 'size'}).reset_index()
    fig = px.bar(interactions, x='candidates', y=['upvotes', 'comments'], barmode='stack', color='candidates', color_discrete_map=candidate_colors)
    st.plotly_chart(fig)

with tab3:
    st.subheader("Word Cloud of Comments")
    stop_words = set(stopwords.words('english')) | {"Trump", "Donald", "Kamala", "Harris", "https", "www", "would"}
    
    # Extraer y procesar los comentarios
    all_comments = " ".join(process_comments(df))
    
    # Generar y mostrar la nube de palabras
    if all_comments:
        wordcloud = WordCloud(width=800, height=400, stopwords=stop_words).generate(all_comments)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No comments available for the selected data.")

with tab4:
    st.subheader("Most Influential Posts")
    influential_posts = df[['title', 'upvotes', 'subreddit']].sort_values(by='upvotes', ascending=False).head(10)
    st.table(influential_posts)

with tab5:
    st.subheader("Sentiment Distribution by Candidate")
    sentiment_by_candidate = df.groupby(['candidates', 'sentiment']).size().reset_index(name='count')
    fig = px.bar(sentiment_by_candidate, x='candidates', y='count', color='sentiment', barmode='stack', color_discrete_map={"Positive": "green", "Negative": "red"})
    st.plotly_chart(fig)

