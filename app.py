import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 🗂 تحميل البيانات
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    links = pd.read_csv("links.csv")
    return movies, links

movies, links = load_data()

# 🖨 التأكد من الأعمدة
st.write("Movies columns:", movies.columns)
st.write("Links columns:", links.columns)

# 🧠 تجهيز TF-IDF باستخدام الـ genres فقط
tfidf = TfidfVectorizer(stop_words='english')
movies['content'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 🔑 API مفتاح TMDB من secrets
API_KEY = st.secrets["TMDB_API_KEY"]

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

# 🌍 جلب الصورة والوصف من TMDB
def get_movie_info_by_id(movie_id):
    row = links[links['movieId'] == movie_id]
    if row.empty or pd.isna(row.iloc[0].get("tmdbId")):
        return None, "No description available."
    tmdb_id = int(row.iloc[0]["tmdbId"])
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster = f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get("poster_path") else None
        overview = data.get("overview") or "No description available."
        return poster, overview
    return None, "No description available."

# 🖥️ واجهة Streamlit
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# 💅 CSS Netflix Style
st.markdown("""
<style>
body { background-color: #141414; }
.title { 
    font-size: 50px; 
    color: #E50914; 
    text-align: center; 
    font-weight: 900; 
    text-shadow: 2px 2px 8px black;
    margin-bottom: 30px;
}
.movie-card {
    background: #1E1E1E;
    border-radius: 12px;
    padding: 15px;
    margin: 10px;
    text-align: center;
    transition: transform 0.3s;
}
.movie-card:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 15px rgba(229,9,20,0.6);
}
.movie-title { 
    font-size: 22px; 
    font-weight: bold; 
    color: white; 
    margin-bottom: 10px;
}
.movie-genre { 
    font-size: 14px; 
    color: #aaa; 
}
.movie-overview {
    font-size: 13px;
    color: #ccc;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍿 Movie Recommender System</div>', unsafe_allow_html=True)

# 📥 إدخال الفيلم
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
            overview = str(overview)[:200]  # slicing آمن
            with cols[(i-1) % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        {"<img src='" + poster + "' width='200'>" if poster else ""}
                        <div class="movie-title">{row.title}</div>
                        <div class="movie-genre">{row.genres}</div>
                        <div class="movie-overview">{overview}...</div>
                    </div>
                """, unsafe_allow_html=True)
