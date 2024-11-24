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
    layout="centered",
)

# Custom CSS for Styling
st.markdown("""
    <style>
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* System Content */
        .system-content {
            font-size: 1.8em;
            color: #3498DB;
            font-weight: bold;
            margin-top: 30px;
            text-align: center;
        }
        /* Section Content */
        .section-content {
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #2C3E50;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #34495E;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            color: #2E86C1;
            font-weight: bold;
        }
        /* Recommendation Titles and Descriptions */
        .recommendation-title {
            font-size: 22px;
            color: #2980B9;
        }
        .recommendation-desc {
            font-size: 16px;
            color: #7F8C8D;
        }
        /* Separator Line */
        .separator {
            margin-top: 10px;
            margin-bottom: 10px;
            border-top: 1px solid #BDC3C7;
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">🔄  Hybrid Recommendation System 🔄 </div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Your one-stop solution for finding the best recommendation for you! 💡</div>', unsafe_allow_html=True)

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
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, Scikit-Learn, Spacy, TF-IDF, SVD, Scipy.</li>
                <li><span class="highlight">⚙️ Approaches</span>: Content-Based Filtering, Collaborative Filtering, and Hybrid Methods</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for web-based interactive systems</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
    # Intro Section
    st.text("")
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Used the 
            <a href="https://grouplens.org/datasets/movielens/1M/" target="_blank" style="color: #2980B9;">MovieLens 1 Million Dataset</a>, 
            which includes user ratings for movies. This dataset enabled the creation of a recommendation system that identifies item-item similarities 
            based on user preferences 🎥.
            <span class="highlight"><br><br>🔗 Additionally,</span>
            movie metadata such as the cover images and IMDb URLs are collected using the 
            <a href="https://pypi.org/project/IMDbPY/" target="_blank" style="color: #2980B9;">IMDbPY library</a>, which allows access to movie information, including movie posters and links to the IMDb pages. 
            If the movie image is not available, a default placeholder image is displayed. 🖼️
        </div>
    """, unsafe_allow_html=True)
# Content-Based Model
with tab2:
    st.markdown('<div class="system-content">📋 Content-Based Model</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This system recommends movies based on user preferences for similar genres, 
            helping users discover movies they may enjoy based on their past choices 🎬.
        </div>
    """, unsafe_allow_html=True)

    # Prepare Genres for Vectorization
    movies['Genres'] = movies['Genres'].fillna('').str.split('|')
    movies['Genres_str'] = movies['Genres'].apply(" ".join)
    movies['Genres'] = movies['Genres'].apply(lambda x: ", ".join(x))
    # Calculate Cosine Similarity
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['Genres_str'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    # Recommendation Function
    movie_indices = pd.Series(movies.index, index=movies['Title']).drop_duplicates()

    def recommend_content(title, n=10):
        idx = movie_indices[title]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return movies.iloc[[i[0] for i in sim_scores]]['Title']

    # UI for Content-Based Recommendation
    selected_movie = st.selectbox("🎥 Select a Movie", ["Please Select"] + movies['Title'].tolist())
    num_recommendations = st.slider("🔢 Number of Recommendations", 1, 10, 5)


    if st.button("✨ Get Content-Based Recommendations"):
        if selected_movie != "Please Select":
            recommendations = recommend_content(selected_movie, n=num_recommendations)
            
            st.markdown("<div class='recommendation-title'>🎬 Recommended Movies:</div>", unsafe_allow_html=True)
            
            for i in range(0, len(recommendations), 4):
                for cols, movie in zip(st.columns(4), recommendations[i:i + 4]):
                    imdb_url, image_url = get_imdb_url(movie)
                    with cols:
                        st.image(image_url, use_column_width=True)
                        st.markdown(f"[🎬 {movie}]({imdb_url})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select a movie to proceed.")

with tab3:
    st.markdown('<div class="system-content">🤝 Collaborative Model</div>', unsafe_allow_html=True)
    st.markdown("""
            <div class="content">
                This system recommends movies based on user interactions (ratings). It uses collaborative filtering 
                with Singular Value Decomposition (SVD) to predict user preferences. 💬
            </div>
        """, unsafe_allow_html=True)
    
    st.text("")
    st.text("")
    st.text("")
    
    def recommend_collaborative(predictions, userID, movies, original_ratings, num_recommendations=10):
        user_row_number = userID - 1  # Adjust for 0-based indexing
        
        # Sort predictions for the user in descending order of rating
        sorted_user_predictions = predictions.iloc[user_row_number].sort_values(ascending=False)
        sorted_user_predictions = pd.DataFrame({
            'MovieID': sorted_user_predictions.index,
            'Predictions': sorted_user_predictions.values
        })  # Ensure column names are clear and explicit
        # Retrieve movies already rated by the user
        user_data = original_ratings[original_ratings["UserID"] == userID]
        user_full = (
            user_data.merge(movies, how="left", on="MovieID")
            .sort_values(["Ratings"], ascending=False)
        )
        
        # Log information about the user's rated movies
        # print(f"User {userID} has already rated {user_full.shape[0]} movies.")
        # print(f"Recommending top {num_recommendations} movies with highest predicted ratings not already rated.")
        
        # Identify movies not yet rated by the user and merge with predictions
        recommendations = (
            movies[~movies["MovieID"].isin(user_full["MovieID"])]
            .merge(sorted_user_predictions, how="left", on="MovieID")  # Explicit column names used
            .sort_values("Predictions", ascending=False)  # Sort based on the predictions
            .iloc[:num_recommendations, :]  # Select top N recommendations
        )
        
        return user_full, recommendations
    
    Ratings = ratings.pivot(index = 'UserID', columns ='MovieID', values = 'Ratings').fillna(0)
    R = np.matrix(Ratings)
    user_means = np.mean(R, axis=1)
    normalized_ratings = R - user_means.reshape(-1, 1)
    U, sigma, Vt = svds(normalized_ratings, k=50)
    sigma = np.diag(sigma)
    
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
    preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

    user_ids = sorted(ratings['UserID'].unique())
    user_id_input = st.selectbox("👤 Select User ID", ["Please Select"] + [int(u) for u in user_ids])
    n_recommendations = st.slider("🔢 Number of User Based Recommendations", 1, 10, 5)

    if st.button("Get Collaborative Recommendations ✨"):
        if user_id_input != "Please Select":
            already_rated, recommendations = recommend_collaborative(preds, user_id_input, movies, ratings, n_recommendations)
            recommendations = recommendations['Title']
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
with tab4:
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
    
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by Machine Learning. 🧠
    </div>""", unsafe_allow_html=True)