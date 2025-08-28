import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from imdb import IMDb
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="Recommendation System",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Advanced Dark Theme Styles (No Black) */
        .stApp {
            background: linear-gradient(rgba(30, 27, 75, 0.9), rgba(30, 27, 75, 0.9)), url('https://aisingapore.org/wp-content/uploads/2022/03/100E-2018-002.png');
            background-size: cover;
            background-attachment: fixed;
            color: #a5b4fc;
            font-family: 'Poppins', sans-serif;
        }
        .main-container {
            background: linear-gradient(135deg, rgba(55, 48, 163, 0.85), rgba(76, 29, 149, 0.85));
            border-radius: 15px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.6);
            border: 2px solid #60a5fa;
            backdrop-filter: blur(10px);
        }
        .main-title {
            font-size: 3.2em;
            font-weight: 700;
            color: #f9a8d4;
            text-align: center;
            margin-bottom: 35px;
            text-shadow: 0 0 12px rgba(249, 168, 212, 0.8);
            animation: pulseGlow 2s ease-in-out infinite;
        }
        .section-title {
            font-size: 2.2em;
            font-weight: 600;
            color: #f9a8d4;
            margin: 40px 0 20px;
            text-shadow: 0 0 10px rgba(249, 168, 212, 0.8);
            border-left: 6px solid #f9a8d4;
            padding-left: 18px;
            animation: slideInLeft 0.6s ease-in-out;
        }
        .system-content {
            font-size: 2.2em;
            font-weight: 600;
            color: #f9a8d4;
            text-align: center;
            text-shadow: 0 0 10px rgba(249, 168, 212, 0.8);
            animation: slideInLeft 0.6s ease-in-out;
        }
        .content {
            font-size: 1.15em;
            color: #a5b4fc;
            line-height: 1.9;
            text-align: justify;
        }
        .highlight {
            color: #fef08a;
            font-weight: bold;
        }
        .separator {
            height: 2px;
            background-color: #60a5fa;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background: linear-gradient(45deg, #ec4899, #7c3aed);
            color: #fef08a;
            border-radius: 12px;
            padding: 14px 30px;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            box-shadow: 0 0 15px rgba(236, 72, 153, 0.8);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #db2777, #6d28d9);
            box-shadow: 0 0 25px rgba(236, 72, 153, 1);
            transform: scale(1.1);
            color: #e0e7ff;
        }
        .stButton>button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300%;
            height: 300%;
            background: rgba(96, 165, 250, 0.2);
            transition: all 0.6s ease;
            transform: translate(-50%, -50%) scale(0);
            border-radius: 50%;
        }
        .stButton>button:hover::after {
            transform: translate(-50%, -50%) scale(1);
        }
        .stSelectbox, .stSlider {
            background: linear-gradient(135deg, rgba(55, 48, 163, 0.9), rgba(76, 29, 149, 0.9));
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #60a5fa;
            color: #a5b4fc;
            transition: all 0.3s ease;
        }
        .stSelectbox:hover, .stSlider:hover {
            border-color: #93c5fd;
            box-shadow: 0 0 8px rgba(147, 197, 253, 0.5);
        }
        .stSelectbox label, .stSlider label {
            color: #fef08a;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.3em;
            font-weight: 500;
            color: #a5b4fc;
            padding: 15px 30px;
            border-radius: 12px 12px 0 0;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, rgba(55, 48, 163, 0.9), rgba(76, 29, 149, 0.9));
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(45deg, #ec4899, #7c3aed);
            color: #fef08a;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #4c1d95, #5b21b6);
            color: #e0e7ff;
        }
        .stImage {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            animation: scaleIn 0.8s ease-in-out;
        }
        .recommendation-title {
            font-size: 1.8em;
            color: #f9a8d4;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .footer {
            font-size: 0.95em;
            color: #a5b4fc;
            margin-top: 50px;
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, rgba(55, 48, 163, 0.85), rgba(76, 29, 149, 0.85));
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 2px solid #60a5fa;
            backdrop-filter: blur(10px);
        }
        .footer a {
            color: #93c5fd;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #f9a8d4;
            text-decoration: underline;
        }
        .content ul li::marker {
            color: #60a5fa;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            background-color: rgba(42, 46, 63, 0.95);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .stDataFrame table {
            color: #a5b4fc;
        }
        /* Animations */
        @keyframes pulseGlow {
            0% { text-shadow: 0 0 10px rgba(249, 168, 212, 0.8); }
            50% { text-shadow: 0 0 20px rgba(249, 168, 212, 1); }
            100% { text-shadow: 0 0 10px rgba(249, 168, 212, 0.8); }
        }
        @keyframes slideInLeft {
            from { transform: translateX(-30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">🔄  Hybrid Recommendation System 🔄 </div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.5em; color: #f9a8d4; text-align: center; text-shadow: 0 0 8px rgba(249, 168, 212, 0.8);">Your one-stop solution for finding the best recommendation for you! 💡</div>', unsafe_allow_html=True)

# Load Data
ratings = pd.read_csv('./Data/ml-1m/ratings.csv', sep='\t', usecols=['UserID', 'MovieID', 'Ratings'])
movies = pd.read_csv('./Data/ml-1m/movies.csv', sep='\t', usecols=['MovieID', 'Title', 'Genres'])

# IMDb Metadata Fetching
def get_imdb_url(movie_title):
    ia = IMDb()
    try:
        search_results = ia.search_movie(movie_title)
        if search_results:
            movie = search_results[0]
            movie_id = movie.movieID
            image_url = movie.get('full-size cover url') or "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"
            return f"https://www.imdb.com/title/tt{movie_id}/", image_url
        return None, "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"
    except Exception:
        return None, "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"

# Tab Sections
st.text("")
st.text("")
tab1, tab2, tab3, tab4 = st.tabs(["🏠Home", "📋 Content-Based Model", "🤝 Collaborative Model", "🔀 Hybrid Model"])

with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring Data Scientist passionate about 
        <span class="highlight">🎥 Recommendation Systems</span>, 🤖 <span class="highlight">Machine Learning</span>, and <span class="highlight">NLP</span>. 
        With hands-on experience in building intelligent systems, I aim to combine my technical expertise with creativity 
        to solve real-world problems. Currently, I am pursuing my Bachelor’s in Computer Science and actively exploring innovative projects. 🚀
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        Welcome to my Hybrid Recommendation System! This project is a result of my efforts to create a robust, 
        user-friendly platform for personalized movie recommendations. Here's what it includes:
        <ul>
            <li><span class="highlight">📋 Content-Based Filtering</span>: Uses movie metadata like genres to find similar movies based on user preferences.</li>
            <li><span class="highlight">🤝 Collaborative Filtering</span>: Leverages user interactions (ratings) to recommend movies based on patterns and similarities among users.</li>
            <li><span class="highlight">🔄 Hybrid Model</span>: Combines the strengths of content-based and collaborative filtering for enhanced and diverse recommendations.</li>
            <li><span class="highlight">🌐 Deployment</span>: Built with Streamlit for a seamless and interactive user experience.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, Scikit-Learn, SciPy, TF-IDF, IMDbPY.</li>
                <li><span class="highlight">⚙️ Approaches</span>: Content-Based Filtering, Collaborative Filtering (SVD), Hybrid Methods.</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for web-based interactive systems.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">📋 Content-Based Model</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Used the 
            <a href="https://grouplens.org/datasets/movielens/1m/" target="_blank" style="color: #93c5fd;">MovieLens 1M Dataset</a>, 
            which includes movie metadata such as genres. This dataset enabled the creation of a content-based recommendation system that identifies movie similarities 
            based on genres 🎥.
            <span class="highlight"><br>🔗 Additionally,</span>
             movie metadata such as the cover images and IMDb URLs are collected using the 
            <a href="https://pypi.org/project/IMDbPY/" target="_blank" style="color: #93c5fd;">IMDbPY library</a>, which allows access to movie information, including movie posters and links to the IMDb pages. 
            If the movie image is not available, a default placeholder image is displayed.
        </div>
    """, unsafe_allow_html=True)

    # Preprocess Genres
    movies['Genres'] = movies['Genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    st.text(" ")
    selected_movie = st.selectbox("🎥 Select a Movie", ["Please Select"] + list(movies['Title']))
    n_recommendations = st.slider("🔢 Number of Recommendations", 1, 10, 5)

    if st.button("🎯 Get Recommendations"):
        if selected_movie != "Please Select":
            # Get index of the selected movie
            idx = movies[movies['Title'] == selected_movie].index[0]
            sim_scores = list(enumerate(cosine_sim_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations + 1]  # Exclude itself
            movie_indices = [i[0] for i in sim_scores]
            recommendations = movies['Title'].iloc[movie_indices]
            st.markdown("<div class='recommendation-title'>🎬 Recommended Movies:</div>", unsafe_allow_html=True)
            for i in range(0, len(recommendations), 4):
                for cols, movie in zip(st.columns(4), recommendations[i:i + 4]):
                    imdb_url, image_url = get_imdb_url(movie)
                    with cols:
                        st.image(image_url, use_column_width=True)
                        st.markdown(f"[🎬 {movie}]({imdb_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select a movie from the dropdown to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">🤝 Collaborative Model</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Used the 
            <a href="https://grouplens.org/datasets/movielens/1m/" target="_blank" style="color: #93c5fd;">MovieLens 1M Dataset</a>, 
            which includes user ratings for movies. This dataset enabled the creation of a collaborative recommendation system that identifies user-item similarities 
            based on ratings 🎥.
            <span class="highlight"><br>🔗 Additionally,</span>
             movie metadata such as the cover images and IMDb URLs are collected using the 
            <a href="https://pypi.org/project/IMDbPY/" target="_blank" style="color: #93c5fd;">IMDbPY library</a>.
        </div>
    """, unsafe_allow_html=True)

    # Collaborative Filtering
    R = ratings.pivot(index='UserID', columns='MovieID', values='Ratings').fillna(0)
    R_np = R.to_numpy()
    user_ratings_mean = np.mean(R_np, axis=1)
    R_demeaned = R_np - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds = pd.DataFrame(all_user_predicted_ratings, columns=R.columns)
    st.text(" ")
    user_ids = sorted(ratings['UserID'].unique())
    user_id_input = st.selectbox("👤 Select Recommender ID", ["Please Select"] + [int(u) for u in user_ids])
    n_recommendations = st.slider("🔢 Number of Recommendations", 1, 10, 5)

    def collaborative_recommendation(user_id, preds, ratings, movies, top_n=20):
        sorted_user_predictions = preds.loc[user_id - 1].sort_values(ascending=False).reset_index()
        sorted_user_predictions.columns = ['MovieID', 'Prediction']
        rated_movie_ids = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
        recommended_movies = sorted_user_predictions[~sorted_user_predictions['MovieID'].isin(rated_movie_ids)]
        top_recommendations = recommended_movies.head(top_n)
        top_recommendation_details = movies[movies['MovieID'].isin(top_recommendations['MovieID'])]
        return top_recommendation_details
    
    if st.button("🎯 Get Recommendations"):
        if user_id_input != "Please Select":
            recommendations = collaborative_recommendation(user_id_input, preds, ratings, movies)
            recommendations = recommendations['Title'][:n_recommendations]
            st.markdown("<div class='recommendation-title'>🎬 Recommended Movies:</div>", unsafe_allow_html=True)
            for i in range(0, len(recommendations), 4):
                for cols, movie in zip(st.columns(4), recommendations[i:i + 4]):
                    imdb_url, image_url = get_imdb_url(movie)
                    with cols:
                        st.image(image_url, use_column_width=True)
                        st.markdown(f"[🎬 {movie}]({imdb_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select a user ID to proceed.")
            
            
    if st.button("See User Details  👀"):
        if user_id_input != "Please Select":
            st.markdown("""
                <div class="content">
                    Selected User Rated Movies Details 🎬:
                </div>
            """, unsafe_allow_html=True)
            user = ratings[ratings['UserID'] == user_id_input]
            user_details = pd.merge(movies, user, on='MovieID')[['UserID', 'MovieID', 'Title', 'Ratings']]
            st.table(user_details)
        else:
            st.warning("⚠️ Please select a user ID to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">🔀 Hybrid Model</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This system recommends movies by combining collaborative filtering and content-based filtering. 
            By normalizing both SVD and content similarity matrices and blending them with adjustable weights, 
            this hybrid model offers more accurate and diverse recommendations. 💡
        </div>
    """, unsafe_allow_html=True)
    
    st.text("")
    st.text("")
    st.text("")
    
    def hybrid_recommendation(user_id, preds, cosine_matrix, ratings, movies, alpha=0.5, beta=0.5, top_n = 20):
        scaler = MinMaxScaler()

        # Normalize the prediction matrices
        collaborative_normalized = scaler.fit_transform(preds)
        content_normalized = scaler.fit_transform(cosine_matrix)

        # Convert the NumPy arrays back to DataFrames with proper column names
        collaborative_normalized = pd.DataFrame(collaborative_normalized, columns=preds.columns)
        content_normalized = pd.DataFrame(content_normalized, columns=movies['MovieID'])

        # Find common movies between both matrices
        common_movie_names = np.intersect1d(collaborative_normalized.columns, content_normalized.columns)

        # Subset both matrices to only common movies
        collaborative_normalized = collaborative_normalized[common_movie_names]
        content_normalized = content_normalized[common_movie_names]

        # Compute hybrid predictions (weighted sum of collaborative and content-based predictions)
        hybrid_predictions = (alpha * collaborative_normalized) + (beta * content_normalized)

        # Get predictions for the specific user (user_id indexing starts at 1, so subtract 1 for proper indexing)
        user_prediction = hybrid_predictions.loc[user_id - 1]  # UserID indexing starts at 1

        # Sort predictions by score in descending order
        sorted_user_predictions = user_prediction.sort_values(ascending=False).reset_index()

        # Rename columns for clarity
        sorted_user_predictions.columns = ['MovieID', 'Prediction']

        # Get the list of movies already rated by the user
        rated_movie_ids = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()

        # Filter out movies that the user has already rated
        recommended_movies = sorted_user_predictions[~sorted_user_predictions['MovieID'].isin(rated_movie_ids)]

        # Get the top 20 recommended movies
        top_recommendations = recommended_movies.head(top_n)

        # Get detailed movie information for the recommended movies
        top_recommendation_details = movies[movies['MovieID'].isin(top_recommendations['MovieID'])]

        return top_recommendation_details
    
    user_ids = sorted(ratings['UserID'].unique())
    user_id_input = st.selectbox("👤 Select Recommender ID", ["Please Select"] + [int(u) for u in user_ids])
    n_recommendations = st.slider("🔢 Number of Hybrid Based Recommendations", 1, 10, 5)
    
    if st.button("✨ Get Hybrid Recommendations"):
        if user_id_input != "Please Select":
            recommendations = hybrid_recommendation(user_id_input, preds, cosine_sim_matrix, ratings, movies, alpha=0.5, beta=0.5)
            recommendations = recommendations['Title'][:n_recommendations]
            st.markdown("<div class='recommendation-title'>🎬 Recommended Movies:</div>", unsafe_allow_html=True)
            for i in range(0, len(recommendations), 4):
                for cols, movie in zip(st.columns(4), recommendations[i:i + 4]):
                    imdb_url, image_url = get_imdb_url(movie)
                    with cols:
                        st.image(image_url, use_column_width=True)
                        st.markdown(f"[🎬 {movie}]({imdb_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select a user ID to proceed.")
            
            
    if st.button("See User Detail  👀"):
        if user_id_input != "Please Select":
            st.markdown("""
                <div class="content">
                    Selected User Rated Movies Details 🎬:
                </div>
            """, unsafe_allow_html=True)
            
            user_details = pd.merge(movies, ratings[ratings['UserID'] == user_id_input], on='MovieID')[['UserID', 'MovieID', 'Title', 'Ratings']]
            st.table(user_details)
        else:
            st.warning("⚠️ Please select a user ID to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)
    
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank">Muhammad Umer Khan</a>. Powered by Machine Learning. 🧠
    </div>""", unsafe_allow_html=True)