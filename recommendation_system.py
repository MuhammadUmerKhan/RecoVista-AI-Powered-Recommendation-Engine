# Libraries
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from imdb import IMDb
import spacy
import joblib as jb
import numpy as np
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
    layout="centered",
)

# Custom CSS for styling
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
        .section-content{
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

# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">💻 Welcome to the NLP Based Recommendation System 💻</div>', unsafe_allow_html=True)
st.markdown('<div class="intro-subtitle">Your one-stop solution for finding the best recommendation for you! 💡</div>', unsafe_allow_html=True)

st.text("")
st.text("")
# Load Data



# Tabs for each recommendation system
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📋 Content-Based Recommendation", "🤝 Collaborative Recommendation", "🔀 Hybrid Recommendation"])

# Home Tab Content
with tab1:
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


# Content-Based Recommendation Tab
with tab2:
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

    # Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.8,
        min_df=2
    )

    # Fitting to Preprocessed Column
    tfidf_matrix = vectorizer.fit_transform(data['Preprocessed_Tags'])

    # Calculating Similarities
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Define recommendation function
    def get_recommendations(title, cosine_sim=cosine_matrix, data=data, top_n=5):
        try:
            idx = data.loc[data['Title'] == title].index[0]  # Faster lookup with .loc
            sim_scores = sorted(
                enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True
            )[1 : top_n + 1]  # Skip the first (itself)
            course_indices = [i[0] for i in sim_scores]
            return data.iloc[course_indices][['Title', 'Description']]
        except IndexError:
            st.error("The selected title was not found. Please choose a valid course.")
    
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Scraped comprehensive course data from the 
            <a href="https://ocw.mit.edu/collections/environment/" target="_blank" style="color: #2980B9;">MIT OpenCourseWare Environment & Sustainability</a> sections. 
            This dataset was utilized to create a system that recommends courses based on content similarity. 💡
        </div><br>
    """, unsafe_allow_html=True)
    
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
                            f"<a href='{row['Link']}' target='_blank' style='color: #2980B9; font-weight: bold;'>{row['Title']}</a>",
                            unsafe_allow_html=True
                        )
        else:
            st.warning("⚠️ Please select a course from the dropdown to proceed.")

# Collaborative Recommendation TabCleaned_data

with tab3:
    st.markdown('<div class="system-content">🤝 Item-Item Collaborative Movie Recommendation System</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content">
            <span class="highlight">📝 Data Collection:</span> Used the 
            <a href="https://grouplens.org/datasets/movielens/100k/" target="_blank" style="color: #2980B9;">MovieLens 100K Dataset</a>, 
            which includes user ratings for movies. This dataset enabled the creation of a recommendation system that identifies item-item similarities 
            based on user preferences 🎥.
            <span class="highlight"><br>🔗 Additionally,</span>
             movie metadata such as the cover images and IMDb URLs are collected using the 
            <a href="https://pypi.org/project/IMDbPY/" target="_blank" style="color: #2980B9;">IMDbPY library</a>, which allows access to movie information, including movie posters and links to the IMDb pages. 
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
                    image_url = "https://user-images.githubusercontent.com/24848110/33519396-7e56363c-d79d-11e7-969b-09782f5ccbab.png"
                
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


# Hybrid Recommendation Tab
with tab4:
    st.markdown('<div class="system-content">🤝 Hybrid Movie Recommendation System</div>', unsafe_allow_html=True)
    st.text(" ")
    st.markdown("""
        <div class="content"><center>
            Because of complexity of app hybrid system is shifted to the  
            <span class="highlight">
            <a href="https://nlp-powered-recommendation-system-second-part.streamlit.app/" target="_blank" style="color: #2980B9;">App</a></span>
            <center>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by Machine Learning. 🧠
    </div>""", unsafe_allow_html=True)