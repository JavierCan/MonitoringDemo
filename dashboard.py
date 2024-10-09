import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import time

# Download NLTK stopwords if necessary
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Set up the page configuration
st.set_page_config(page_title="Real-Time Analysis of Candidate Posts", layout="wide", initial_sidebar_state="expanded")

# Candidate color configuration
candidate_colors = {
    "Kamala Harris": "#1f77b4",  # Blue
    "Donald Trump": "#d62728"    # Red
}

# Connect to MongoDB using st.secrets
mongo_uri = st.secrets["mongo"]["uri"]
client = MongoClient(mongo_uri)
db = client['elections']
collection = db['monitoring']

# Function to load data from MongoDB
@st.cache_data(ttl=60)
def get_data():
    data = list(collection.find({}, {
        'date': 1,
        'candidates': 1,
        'sentiment': 1,
        'upvotes': 1,
        'comments': 1,
        'title': 1,
        'subreddit': 1
    }))
    df = pd.DataFrame(data)

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # Unify candidate names
    def unify_candidate_names(candidates):
        if isinstance(candidates, list):
            if any(c in ['Kamala', 'Kamala Harris'] for c in candidates):
                return 'Kamala Harris'
            elif any(c in ['Trump', 'Donald Trump'] for c in candidates):
                return 'Donald Trump'
        return 'Others'
    df['candidates'] = df['candidates'].apply(unify_candidate_names)
    return df

# Function to process and extract text from comments
def process_comments(df):
    comments_list = []
    for comments in df['comments']:
        if isinstance(comments, list):
            for comment in comments:
                if isinstance(comment, dict) and 'comment_body' in comment:
                    comments_list.append(comment['comment_body'])
        elif isinstance(comments, dict):
            if 'comment_body' in comments:
                comments_list.append(comments['comment_body'])
        elif isinstance(comments, str):
            comments_list.append(comments)
    return comments_list

# Load data
df = get_data()
last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

# Show last updated time
st.sidebar.markdown(f"**Last updated:** {last_updated}")
if st.sidebar.button("Refresh now"):
    st.experimental_rerun()  # Restart the app to refresh data

# Date and candidate filters
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start date", min_value=df['date'].min().date(), value=df['date'].min().date())
end_date = st.sidebar.date_input("End date", max_value=df['date'].max().date(), value=df['date'].max().date())
selected_candidate = st.sidebar.selectbox("Select a candidate", ["All", "Kamala Harris", "Donald Trump"])

# Apply filters
df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
if selected_candidate != "All":
    df = df[df['candidates'] == selected_candidate]

# Dashboard Title
st.title("Real-Time Analysis of Candidate Posts")
st.write("This dashboard provides real-time insights into social media sentiment, interactions, and trends regarding political candidates.")

# Sentiment Analysis and Trends
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment Over Time",
    "Total Interactions",
    "Word Cloud",
    "Most Influential Posts",
    "Sentiment Distribution"
])

with tab1:
    st.subheader("Sentiment Comparison Over Time")
    if not df.empty:
        sentiment_counts = df.groupby([df['date'].dt.date, 'candidates', 'sentiment']).size().reset_index(name='counts')
        fig = px.line(
            sentiment_counts,
            x='date',
            y='counts',
            color='candidates',
            line_dash='sentiment',
            color_discrete_map=candidate_colors
        )
        fig.update_layout(transition_duration=500)
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")

with tab2:
    st.subheader("Total Interactions")
    if not df.empty:
        interactions = df.groupby('candidates').agg({'upvotes': 'sum', 'comments': 'size'}).reset_index()
        fig = px.bar(
            interactions,
            x='candidates',
            y=['upvotes', 'comments'],
            barmode='stack',
            color='candidates',
            color_discrete_map=candidate_colors
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")

with tab3:
    st.subheader("Word Cloud of Comments")
    stop_words.update({"Trump", "Donald", "Kamala", "Harris", "https", "www", "would"})
    if not df.empty:
        all_comments = " ".join(process_comments(df))
        if all_comments:
            wordcloud = WordCloud(width=800, height=400, stopwords=stop_words).generate(all_comments)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No comments available for the selected data.")
    else:
        st.write("No data available for the selected filters.")

with tab4:
    st.subheader("Most Influential Posts")
    if not df.empty:
        influential_posts = df[['title', 'upvotes', 'subreddit']].sort_values(by='upvotes', ascending=False).head(10)
        st.table(influential_posts)
    else:
        st.write("No data available for the selected filters.")

with tab5:
    st.subheader("Sentiment Distribution by Candidate")
    if not df.empty:
        sentiment_by_candidate = df.groupby(['candidates', 'sentiment']).size().reset_index(name='count')
        fig = px.bar(
            sentiment_by_candidate,
            x='candidates',
            y='count',
            color='sentiment',
            barmode='stack',
            color_discrete_map={"Positive": "green", "Negative": "red"}
        )
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected filters.")
