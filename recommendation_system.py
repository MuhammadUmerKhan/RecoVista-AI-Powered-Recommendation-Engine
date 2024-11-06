# Libraries
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Set page configuration for Streamlit
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="📚",
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
st.markdown('<div class="main-title">📚 Welcome to the NLP Based Course Recommendation System 📚</div>', unsafe_allow_html=True)

# Load Data
data = pd.read_csv("./Data/Cleaned_data.csv").drop(columns='Unnamed: 0')
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
    idx = data[data['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    course_indices = [i[0] for i in sim_scores]
    
    return data.iloc[course_indices][['Title', 'Description']]

# Tabs for each recommendation system
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📋 Content-Based Recommendation", "🤝 Collaborative Recommendation", "🔄 Hybrid Recommendation"])

# Home Tab Content
with tab1:
    # st.markdown('<div class="intro-title">📚 Welcome to the NLP Based Course Recommendation System 📚</div>', unsafe_allow_html=True)
    st.markdown('<div class="intro-subtitle">Your one-stop solution for finding the best courses tailored for you! 💡</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">Hi! I\'m Muhammad Umer Khan, an aspiring Data Scientist with a passion for Natural Language Processing (NLP) and recommendation systems. Currently pursuing my Bachelor’s in Computer Science, I have hands-on experience with projects in data science, data scraping, and building intelligent recommendation systems.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚀 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            This project focuses on creating a comprehensive <span class="highlight">Course Recommendation System</span> using advanced NLP techniques. Here’s what we achieved:
            <ul>
                <li><span class="highlight">Data Collection 🗂️</span>: Scraped relevant data from multiple sections on courses in Environment & Sustainability.</li>
                <li><span class="highlight">Content-based Filtering 🔍</span>: Utilized course descriptions and topics to recommend similar courses.</li>
                <li><span class="highlight">Collaborative Filtering & Hybrid Models 🤝🔄</span>: Planned for further development to enhance recommendation accuracy.</li>
                <li><span class="highlight">Deployment 🌐</span>: Built a user-friendly app with an intuitive interface for course recommendations.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔧 Technologies Used</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            - <span class="highlight">Languages & Libraries</span>: Python, Pandas, Scikit-Learn, Spacy, TF-IDF, and Streamlit<br>
            - <span class="highlight">Recommendation Approaches</span>: Content-based filtering, Collaborative filtering, and Hybrid models<br>
            - <span class="highlight">Deployment</span>: Streamlit for interactive interface and easy deployment.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# Content-Based Recommendation Tab
with tab2:
    st.markdown('<div class="section-title">📋 Content-Based Recommendation System</div>', unsafe_allow_html=True)
    
    selected_course = st.selectbox("Choose a course", data['Title'].values)
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_course)
        
        for no, (idx, row) in enumerate(recommendations.iterrows(), start=1):
            st.markdown(f"<div class='recommendation-title'>{no}. {row['Title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='recommendation-desc'>{row['Description']}</div>", unsafe_allow_html=True)
            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Collaborative Recommendation Tab
with tab3:
    st.markdown('<div class="section-title">🤝 Collaborative Recommendation System</div>', unsafe_allow_html=True)
    st.write('<div class="section-content">🚧 This feature is under development. Please check back soon for updates!', unsafe_allow_html=True)

# Hybrid Recommendation Tab
with tab4:
    st.markdown('<div class="section-title">🔄 Hybrid Recommendation System</div>', unsafe_allow_html=True)
    st.write('<div class="section-content">🚧 This feature is under development. Stay tuned for enhanced recommendations!', unsafe_allow_html=True)
    # st.write('<div class="section-content">🚧 This feature is under development. Please check back soon for updates!', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app/" target="_blank" style="color: #2980B9;">Muhammad Umer Khan</a>. Powered by Spacy and TF-IDF. 🌐
    </div>
""", unsafe_allow_html=True)
