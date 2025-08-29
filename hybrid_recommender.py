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

# Load Datasets
@st.cache_data
def load_data():
    # movies = pd.read_csv('./data/movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], engine='python', encoding='latin-1')
    # ratings = pd.read_csv('./data/ratings.dat', sep='::', names=['UserID', 'MovieID', 'Ratings', 'Timestamp'], engine='python', encoding='latin-1')
    ratings = pd.read_csv('./Data/ml-1m/ratings.csv', sep='\t', usecols=['UserID', 'MovieID', 'Ratings'])
    movies = pd.read_csv('./Data/ml-1m/movies.csv', sep='\t', usecols=['MovieID', 'Title', 'Genres'])
    return movies, ratings

movies, ratings = load_data()

# Precompute SVD Predictions
@st.cache_data
def compute_svd_predictions(ratings):
    ratings_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Ratings').fillna(0)
    matrix = ratings_matrix.to_numpy()
    user_ratings_mean = np.mean(matrix, axis=1)
    ratings_demeaned = matrix - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(ratings_demeaned, k=50)
    sigma = np.diag(sigma)
    preds = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(preds, columns=ratings_matrix.columns, index=ratings_matrix.index)
    return preds_df

preds = compute_svd_predictions(ratings)

# Precompute Cosine Similarity Matrix
@st.cache_data
def compute_cosine_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['Genres'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=movies['MovieID'], columns=movies['MovieID'])
    return cosine_sim_df

cosine_sim_matrix = compute_cosine_similarity(movies)

# Function to Get IMDB URL and Image
ia = IMDb()
@st.cache_data
def get_imdb_url(movie_name):
    try:
        year = movie_name[-5:-1] if movie_name[-1] == ')' else None
        movie_name_clean = movie_name[:-7] if year else movie_name
        movies = ia.search_movie(movie_name_clean)
        if movies:
            movie = movies[0]
            ia.update(movie, info=['main'])
            imdb_id = movie.movieID
            imdb_url = f"https://www.imdb.com/title/tt{imdb_id}/"
            poster_url = movie.get('full-size cover url')
            return imdb_url, poster_url
    except Exception as e:
        st.warning(f"Error fetching IMDb data for {movie_name}: {e}")
    return "#", "https://via.placeholder.com/150"

# Tabs for Different Recommendation Types
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔄 Collaborative Filtering", "📖 Content-Based Filtering", "🔀 Hybrid Model"])

with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, a dedicated Data Scientist and Machine Learning enthusiast with a Bachelor’s in Computer Science. 
            With hands-on experience in <span class="highlight">🤖 Natural Language Processing (NLP)</span>, 🧠 Machine Learning, and MLOps, I specialize in building intelligent systems, 
            from data pipelines to deployable applications. My journey includes developing recommendation systems, optimizing ANN models, and integrating advanced LLMs, 
            all while pursuing excellence in real-world problem-solving. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project is a state-of-the-art movie recommendation system, showcasing a complete MLOps pipeline and advanced AI integration. Here's what I've achieved:
            <ul>
                <li><span class="highlight">📊 Exploratory Data Analysis (EDA)</span>: Analyzed the dataset to uncover insights, patterns, and ensure data quality.</li>
                <li><span class="highlight">🛠 Data Preprocessing</span>: Cleaned, transformed, encoded features, and balanced data with SMOTEENN for robust training.</li>
                <li><span class="highlight">🔗 Model Development</span>: Built an Artificial Neural Network (ANN) for classifying loan applications into approved or denied categories.</li>
                <li><span class="highlight">⚙️ Model Optimization</span>: Tuned hyperparameters and applied dropout layers to enhance performance metrics (accuracy, precision, recall, F1-score).</li>
                <li><span class="highlight">📈 Evaluation</span>: Achieved ~94% accuracy with comprehensive metrics, logged via MLflow for tracking.</li>
                <li><span class="highlight">📦 Model Registry</span>: Registered the model in MLflow with versioning and aliases for production readiness.</li>
                <li><span class="highlight">🌐 Deployment</span>: Developed an interactive Streamlit app with real-time predictions, batch processing, and LLM-powered analysis.</li>
                <li><span class="highlight">💬 LLM Integration</span>: Added LLM (Mixtral-8x7B via Grok API) for loan approval predictions and customer sentiment analysis.</li>
                <li><span class="highlight">🧩 MLOps Pipeline</span>: Designed a modular pipeline (ingestion to deployment) with logging and error handling.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Data Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            The
            <a href="https://www.kaggle.com/competitions/playground-series-s4e10" target="_blank" style="color: #93c5fd;">Dataset</a>
            used in this project contains key attributes for loan approval prediction. Here's a summary:
            <ul>
                <li><span class="highlight">📜 Features</span>: Includes age, income, home ownership, employment length, loan amount, interest rate, credit history, and more.</li>
                <li><span class="highlight">⚖️ Class Balance</span>: Balanced with SMOTEENN to ensure fair evaluation.</li>
                <li><span class="highlight">🔍 Feature Engineering</span>: Derived loan-to-income ratio and other features to boost prediction accuracy.</li>
                <li><span class="highlight">📊 Insights</span>: 
                    <ul>
                        <li>Higher incomes positively correlate with loan approvals.</li>
                        <li>Employment stability significantly influences decisions.</li>
                        <li>High-interest rates increase the likelihood of denial.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Imbalanced-learn, Matplotlib, Seaborn, LangChain, MLflow, Joblib.</li>
                <li><span class="highlight">⚙️ Methods</span>: Feature Engineering, Artificial Neural Networks (ANN), SMOTEENN, Hyperparameter Tuning, MLOps.</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for interactive web apps, deployable on cloud platforms.</li>
                <li><span class="highlight">📊 Visualization Tools</span>: Matplotlib and Seaborn for EDA and insights.</li>
                <li><span class="highlight">🧠 NLP & LLM</span>: Grok API (Mixtral-8x7B) for advanced predictions and sentiment analysis.</li>
                <li><span class="highlight">📦 MLOps Tools</span>: MLflow for model tracking, versioning, and registry.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌟 Why This Project?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project exemplifies my ability to design, implement, and deploy a full MLOps pipeline, integrating cutting-edge AI technologies like ANN and LLM. 
            By solving a real-world loan approval challenge, it highlights my skills in data science, software engineering, and user-focused development. 
            My goal is to empower data-driven decisions with scalable, accessible solutions. ✨
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">🔄 Collaborative Filtering</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This system recommends movies based on user ratings and patterns. Using Singular Value Decomposition (SVD), 
            it predicts ratings for unseen movies and suggests the top ones a user might like. 🎥
        </div>
    """, unsafe_allow_html=True)
    
    st.text("")
    st.text("")
    st.text("")
    
    def collaborative_recommendation(user_id, preds, ratings, movies, top_n = 20):
        user_prediction = preds.loc[user_id]
        sorted_user_predictions = user_prediction.sort_values(ascending=False).reset_index()
        sorted_user_predictions.columns = ['MovieID', 'Prediction']
        rated_movie_ids = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
        recommended_movies = sorted_user_predictions[~sorted_user_predictions['MovieID'].isin(rated_movie_ids)]
        top_recommendations = recommended_movies.head(top_n)
        top_recommendation_details = movies[movies['MovieID'].isin(top_recommendations['MovieID'])]
        return top_recommendation_details
    
    user_ids = sorted(ratings['UserID'].unique())
    user_id_input = st.selectbox("👤 Select Recommender ID", ["Please Select"] + [int(u) for u in user_ids], key="collab_user_id")
    n_recommendations = st.slider("🔢 Number of Collaborative Based Recommendations", 1, 10, 5, key="collab_n_recs")
    
    if st.button("🎯 Get Recommendations", key="collab_get_recs"):
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
            
            
    if st.button("See User Details  👀", key="collab_user_details"):
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

with tab3:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">📖 Content-Based Filtering</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This system recommends movies similar to a user's favorites based on genres and descriptions. 
            Using TF-IDF and cosine similarity, it suggests movies with matching content profiles. 🍿
        </div>
    """, unsafe_allow_html=True)
    
    st.text("")
    st.text("")
    st.text("")
    
    def content_based_recommendation(user_id, cosine_sim_matrix, ratings, movies, top_n = 20):
        user_rated_movies = ratings[ratings['UserID'] == user_id]
        user_rated_movies = user_rated_movies.sort_values(by='Ratings', ascending=False)
        top_rated_movie_id = user_rated_movies.iloc[0]['MovieID']
        sim_scores = list(enumerate(cosine_sim_matrix[top_rated_movie_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies.iloc[movie_indices]
        return recommended_movies
    
    user_ids = sorted(ratings['UserID'].unique())
    user_id_input = st.selectbox("👤 Select Recommender ID", ["Please Select"] + [int(u) for u in user_ids], key="content_user_id")
    n_recommendations = st.slider("🔢 Number of Content Based Recommendations", 1, 10, 5, key="content_n_recs")
    
    if st.button("🎯 Get Recommendations", key="content_get_recs"):
        if user_id_input != "Please Select":
            recommendations = content_based_recommendation(user_id_input, cosine_sim_matrix, ratings, movies)
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
            
            
    if st.button("See User Details  👀", key="content_user_details"):
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
    user_id_input = st.selectbox("👤 Select Recommender ID", ["Please Select"] + [int(u) for u in user_ids], key="hybrid_user_id")
    n_recommendations = st.slider("🔢 Number of Hybrid Based Recommendations", 1, 10, 5, key="hybrid_n_recs")
    
    if st.button("✨ Get Hybrid Recommendations", key="hybrid_get_recs"):
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
            
            
    if st.button("See User Details  👀", key="hybrid_user_details"):
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