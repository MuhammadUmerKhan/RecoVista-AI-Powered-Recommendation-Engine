# Libraries
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from imdb import IMDb
import spacy
import joblib as jb
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Set page configuration for Streamlit
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
            background: linear-gradient(rgba(30, 27, 75, 0.9), rgba(30, 27, 75, 0.9)), url('https://3cloudsolutions.com/wp-content/uploads/2022/11/blog-building-recommendation-system.jpg');
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

# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">💻 NLP Based Recommendation System 💻</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.5em; color: #f9a8d4; text-align: center; text-shadow: 0 0 8px rgba(249, 168, 212, 0.8);">Your one-stop solution for finding the best recommendations! 💡</div>', unsafe_allow_html=True)

st.text("")
st.text("")
# Load Data

# Tabs for each recommendation system
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📋 Content-Based Recommendation", "🤝 Collaborative Recommendation", "🔀 Hybrid Recommendation"])

# Home Tab Content
with tab1:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring Data Scientist passionate about 
            <span class="highlight">🤖 Natural Language Processing (NLP)</span> and 🧠 Machine Learning. 
            Currently pursuing my Bachelor’s in Computer Science, I bring hands-on experience in developing intelligent recommendation systems, 
            performing data analysis, and building machine learning models. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project is a culmination of my skills in NLP and recommendation systems. Here's what it encompasses:
            <ul>
                <li><span class="highlight">📋 Content-Based Filtering</span>: Leveraged course descriptions, topics, and departments to suggest similar courses.</li>
                <li><span class="highlight">🤝 Collaborative Filtering</span>: Developed a movie recommendation system using user interactions.</li>
                <li><span class="highlight">🔄Hybrid Model</span>: Planned for combining content and collaborative methods for enhanced recommendations.</li>
                <li><span class="highlight">🌐Deployment</span>: Built an interactive, user-friendly interface using Streamlit for seamless recommendations.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries</span>: Python, Pandas, Scikit-Learn, Spacy, TF-IDF, Nearest Neighbors, Scipy.</li>
                <li><span class="highlight">⚙️ Approaches</span>: Content-Based Filtering, Collaborative Filtering, and Hybrid Methods</li>
                <li><span class="highlight">🌐 Deployment</span>: Streamlit for web-based interactive systems</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌟 Why This Project?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project reflects my expertise and dedication to solving real-world problems through data science. 
            It bridges the gap between technical innovation and user-friendly application design. 
            I aim to enhance users' experiences by recommending the most relevant courses tailored to their interests. ✨
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Content-Based Recommendation Tab
with tab2:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">📋 Content-Based Recommendation System</div><br>', unsafe_allow_html=True)
    
    
    data = pd.read_csv("./Data/Cleaned_data.csv")
    data['Tags'] = data['Description'] + data['Departments'] + data['Topics']

    # Text Preprocessor Function
    def text_preprocessor(text):
        doc = nlp(text=str(text).lower())
        filtered_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "ADJ", "VERB"]
        ]
        return " ".join(filtered_tokens)

    # Applying Preprocessor Function
    data['Preprocessed_Tags'] = data['Tags'].apply(text_preprocessor)
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    tags_matrix = vectorizer.fit_transform(data['Preprocessed_Tags'])
    
    # Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tags_matrix)
    
    # Function to get recommendations
    def get_recommendations(course, cosine_sim=cosine_sim, n=5):
        # Get index of the course
        idx = data[data['Title'] == course].index[0]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort courses based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top n most similar courses
        sim_scores = sim_scores[1:n+1]  # Exclude itself
        
        # Get course indices
        course_indices = [i[0] for i in sim_scores]
        
        # Return the top n most similar courses
        return data[['Title']].iloc[course_indices]
    
    selected_course = st.selectbox("🔍 Choose a course", ["Please Select"] + list(data['Title'].values))
    
    if st.button("✨ Get Recommendations"):
        if selected_course != "Please Select":
            # Fetch recommendations
            recommendations = get_recommendations(selected_course)
            
            # Merge with original data to get links and image URLs
            result = recommendations.merge(data[['Title', 'Link', 'urls']], on='Title', how='left')
            
            # Display recommendations in rows of 4
            cols_per_row = 4
            num_recommendations = len(result)
            
            for i in range(0, num_recommendations, cols_per_row):
                # Create a row with `cols_per_row` columns
                cols = st.columns(cols_per_row)
                
                for col, (_, row) in zip(cols, result.iloc[i:i + cols_per_row].iterrows()):
                    with col:
                        # Display course image
                        if not pd.isna(row['urls']):
                            st.image(row['urls'], use_column_width=True)
                        
                        # Display clickable course title
                        st.markdown(
                            f"<a href='{row['Link']}' target='_blank' style='color: #93c5fd; font-weight: bold;'>{row['Title']}</a>",
                            unsafe_allow_html=True
                        )
        else:
            st.warning("⚠️ Please select a course from the dropdown to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# Collaborative Recommendation Tab

with tab3:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">🤝 Item-Item Collaborative Movie Recommendation System</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Used the 
            <a href="https://grouplens.org/datasets/movielens/100k/" target="_blank" style="color: #93c5fd;">MovieLens 100K Dataset</a>, 
            which includes user ratings for movies. This dataset enabled the creation of a recommendation system that identifies item-item similarities 
            based on user preferences 🎥.
            <span class="highlight"><br>🔗 Additionally,</span>
             movie metadata such as the cover images and IMDb URLs are collected using the 
            <a href="https://pypi.org/project/IMDbPY/" target="_blank" style="color: #93c5fd;">IMDbPY library</a>, which allows access to movie information, including movie posters and links to the IMDb pages. 
            If the movie image is not available, a default placeholder image is displayed.
        </div>
    """, unsafe_allow_html=True)
    # Load KNN model and movie pivot data
    knn_movie_model = jb.load("./models/item_item_knn_model.joblib")
    movie_to_user_pvt = pd.read_csv("./Data/movie_to_user_pivot.csv", index_col='Movie title')
    movie_lst = movie_to_user_pvt.index

    # Define function for recommendations
    def get_similar_movies(movie, n=5):
        
        idx = movie_to_user_pvt.index.get_loc(movie)  # Faster indexing with get_loc
        knn_input = movie_to_user_pvt.iloc[idx].values.reshape(1, -1)
        distances, indices = knn_movie_model.kneighbors(knn_input, n_neighbors=n + 1)
        return [movie_to_user_pvt.index[i] for i in indices.flatten()[1:]]  # Exclude self

    def get_imdb_url(movie_title):
        ia_collaborative = IMDb()
        try:
            movies = ia_collaborative.search_movie(movie_title)
            
            if movies:
                movie = movies[0]
                movie_id = movie.getID()
                image_url = movie.get('full-size cover url')
                
                # Return a default image if the movie image is not available
                if not image_url:
                    image_url = "https://user-images.githubusercontent.com/0/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"
                
                return f"https://www.imdb.com/title/tt{movie_id}/", image_url
            else:
                return None, "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"
        except Exception as e:
            return None, "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"

    # User inputs
    st.text(" ")
    selected_movie = st.selectbox("🎥 Select a Movie", ["Please Select"] + list(movie_lst))
    n_recommendations = st.slider("🔢 Number of Recommendations", 1, 10, 5)

    if st.button("🎯 Get Recommendations"):
        if selected_movie != "Please Select":
            similar_movies = get_similar_movies(selected_movie, n_recommendations)
            
            st.markdown("<div class='recommendation-title'>🎬 Recommended Movies:</div>", unsafe_allow_html=True)
            
            # Create rows of 4 recommendations each
            for i in range(0, len(similar_movies), 4):  # Process 4 recommendations at a time
                
                # Fill the row with up to 4 movies
                for cols, movie in zip(st.columns(4), similar_movies[i:i + 4]):  # Assign movies to columns
                    imdb_url, image_url = get_imdb_url(movie)  # Fetch IMDb data
                    
                    with cols:
                        # Display movie image
                        st.image(image_url, use_column_width=True)
                        
                        # Display clickable movie title
                        st.markdown(
                            f"[🎬 {movie}]({imdb_url})",
                            unsafe_allow_html=True
                        )
        else:
            st.warning("⚠️ Please select a movie from the dropdown to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# Hybrid Recommendation Tab
with tab4:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="system-content">🤝 Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content"><center>
            Because of complexity of app hybrid system is shifted to the  
            <span class="highlight">
            <a href="https://nlp-powered-recommendation-system-second-part.streamlit.app/" target="_blank" style="color: #93c5fd;">App</a></span>
            <center>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank">Muhammad Umer Khan</a>. Powered by Machine Learning. 🧠
    </div>""", unsafe_allow_html=True)