import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 🗂 تحميل البيانات مع check
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("movies.csv")
        links = pd.read_csv("links.csv")
    except FileNotFoundError:
        st.error("❌ الملفات movies.csv أو links.csv مش موجودة في المسار.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Check الأعمدة
    expected_movies_cols = {'movieId', 'title', 'genres'}
    expected_links_cols = {'movieId', 'tmdbId', 'imdbId'}
    
    if not expected_movies_cols.issubset(set(movies.columns)):
        st.error(f"❌ الأعمدة المتوقعة في movies.csv ناقصة. الأعمدة الموجودة: {movies.columns.tolist()}")
        return pd.DataFrame(), pd.DataFrame()
    
    if not expected_links_cols.issubset(set(links.columns)):
        st.error(f"❌ الأعمدة المتوقعة في links.csv ناقصة. الأعمدة الموجودة: {links.columns.tolist()}")
        return pd.DataFrame(), pd.DataFrame()
    
    return movies, links

movies, links = load_data()

# لو الملفات فاضية وقف البرنامج
if movies.empty or links.empty:
    st.stop()

# 🧠 تجهيز TF-IDF باستخدام الـ genres فقط
tfidf = TfidfVectorizer(stop_words='english')
movies['content'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 🔑 API مفتاح TMDB
API_KEY = "a937d1d593a111c2fbed18fc773f68d"

# دالة تنظيف النصوص للبحث المرن
def clean_text(text):
    if pd.isna(text):
        return ""
    return ''.join(e for e in text.lower() if e.isalnum())

# دالة الترشيح
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

# 🌍 جلب الصورة والوصف من TMDB
def get_movie_info_by_id(movie_id):
    row = links[links['movieId'] == movie_id]
    if row.empty or pd.isna(row.iloc[0].get("tmdbId")):
        return None, "No description available."
    try:
        tmdb_id = int(row.iloc[0]["tmdbId"])
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster = f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else None
            overview = data.get("overview") or "No description available."
            return poster, overview
        else:
            return None, f"Error from TMDB: {response.status_code}"
    except Exception as e:
        return None, f"Exception: {str(e)}"

# 🖥️ واجهة Streamlit
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.markdown('<h1 style="text-align:center; color:#E50914;">🍿 Movie Recommender System</h1>', unsafe_allow_html=True)

movie_name = st.text_input("اكتب اسم فيلم:", placeholder="مثال: Toy Story")

if movie_name:
    results = get_recommendations(movie_name)
    if results.empty:
        st.warning("❌ الفيلم مش موجود في الداتا.")
    else:
        st.success("✅ أفلام مقترحة لك:")
        cols = st.columns(3)
        for i, row in enumerate(results.itertuples(), 1):
            poster, overview = get_movie_info_by_id(row.movieId)
            overview = overview or "No description available."
            overview = str(overview)[:200]
            with cols[(i-1) % 3]:
                st.markdown(f"""
                    <div style="background:#1E1E1E; border-radius:12px; padding:15px; text-align:center; margin:10px;">
                        {"<img src='" + poster + "' width='200'>" if poster else ""}
                        <div style="font-size:18px; color:white; font-weight:bold;">{row.title}</div>
                        <div style="font-size:14px; color:#aaa;">{row.genres}</div>
                        <div style="font-size:13px; color:#ccc;">{overview}...</div>
                    </div>
                """, unsafe_allow_html=True)
