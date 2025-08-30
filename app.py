import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 🗂 تحميل البيانات
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    return movies

movies = load_data()

# 🧠 تجهيز TF-IDF باستخدام الـ genres فقط
tfidf = TfidfVectorizer(stop_words='english')
movies['content'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# دالة تنظيف النصوص للبحث المرن
def clean_text(text):
    if pd.isna(text):
        return ""
    return ''.join(e for e in text.lower() if e.isalnum())

# دالة الترشيح الذكية
def get_recommendations(title, cosine_sim=cosine_sim):
    title_clean = clean_text(title)
    idx_list = movies[movies['title'].apply(lambda x: clean_text(x).find(title_clean) != -1)].index
    if len(idx_list) == 0:
        return pd.DataFrame()
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres', 'movieId']]

# 🖥️ واجهة Streamlit
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.markdown('<h1 style="text-align:center; color:#E50914;">🍿 Movie Recommender System</h1>', unsafe_allow_html=True)

# 📥 إدخال الفيلم
movie_name = st.text_input("اكتب اسم فيلم:", placeholder="مثال: Toy Story")

if movie_name:
    results = get_recommendations(movie_name)
    if results.empty:
        st.warning("❌ الفيلم مش موجود في الداتا.")
    else:
        st.success("✅ أفلام مقترحة لك:")
        for row in results.itertuples():
            st.markdown(f"**{row.title}** - *{row.genres}*")
