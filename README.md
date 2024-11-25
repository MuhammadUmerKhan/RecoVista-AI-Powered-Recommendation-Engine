# 🌟 Recommendation System 🌟  

Welcome to the **[Recommendation System](https://nlp-powered-recommendation-system.streamlit.app/)** project! This tool leverages **Content-Based**, **Collaborative Filtering**, and **Hybrid** approaches to provide personalized course and movie recommendations tailored to user preferences. 🚀  

---

## 📚 Table of Contents  
- [🔍 Overview](#-overview)  
- [🛠️ Project Structure](#-project-structure)  
- [💻 Technologies Used](#-technologies-used)  
- [✔️ Current Work](#-current-work)  
- [🎯 Planned Future Enhancements](#-planned-future-enhancements)  
- [🚀 Getting Started](#-getting-started)  
- [📄 Acknowledgments](#-acknowledgments)  

---  

## 🔍 Overview  

This project is a comprehensive recommendation system that employs **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach**. It recommends courses or movies by analyzing content metadata, user interaction data, or a combination of both, ensuring personalized suggestions.  

---

## 🛠️ Project Structure  

### 📌 **Content-Based Filtering**  
- Analyzes metadata (e.g., course title, tags, and descriptions) to generate recommendations based on content similarity.  

### 📌 **Collaborative Filtering**  
- Utilizes user interaction data to recommend items based on similar users' preferences.  

### 📌 **Hybrid Approach**  
- Combines both content-based and collaborative filtering techniques to deliver more accurate and personalized recommendations.  

---  

## 💻 Technologies Used  
- **🐍 Python**: Primary language for implementing recommendation algorithms.  
- **📊 Streamlit**: For deploying the interactive web application.  
- **🔠 spaCy**: Used for natural language processing tasks in content-based systems.  
- **🛠️ NLTK**: For text preprocessing, including stopword removal and tokenization.  
- **🧮 Pandas & NumPy**: Data manipulation and numerical operations.  
- **📈 Scikit-learn**: Machine learning models, evaluation metrics, and clustering algorithms.  
- **📚 IMDbPY**: For fetching movie metadata like covers and IMDb URLs in collaborative filtering.  

---  

## ✔️ Current Work  

### 📘 **Content-Based Recommendation System**  

1. **Data Collection and Preprocessing**:  
   - Scraped comprehensive course data from the [MIT OpenCourseWare Environment & Sustainability](https://ocw.mit.edu/collections/environment/) sections.  
   - Merged datasets for "Earth Systems and Climate Science" and "Engineering" into a unified structure.  

2. **Feature Engineering**:  
   - Created a `Tags` column combining `Description`, `Departments`, and `Topics`.  
   - Vectorized the `Tags` column using **TF-IDF** to compute cosine similarity.  

3. **Recommendations**:  
   - Developed a content-based recommendation model to suggest courses similar to a given course.  

---  

### 🎥 **Collaborative Recommendation System**  

1. **Data Collection and Preprocessing**:  
   - Used the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) for user-item rating data.  

2. **Model Implementation**:  
   - Employed matrix factorization to predict user-item ratings.  
   - Used **cosine similarity** with the **NearestNeighbors algorithm** to identify similar users or items.  
   - Recommended movies to users by identifying top-rated items among closely matched similar users.  

3. **Enhancements**:  
   - Integrated **IMDbPY** to fetch movie metadata such as cover images and IMDb URLs.  
   - Added a fallback mechanism to display placeholder images for unavailable movie posters.  

---  

### 🌐 **Hybrid Recommendation System**  

1. **Data Collection and Preprocessing**:  
   - Used the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) for building the hybrid system.  

2. **Model Implementation**:  
   - Normalized predictions from the content-based and collaborative filtering models.  
   - Combined predictions using a **weighted averaging approach** (50% content-based, 50% collaborative).  
   - Built a hybrid recommendation function to provide movie/course suggestions based on this combined model.  

3. **Evaluation**:  
   - Implemented RMSE with 5-fold cross-validation, achieving a score of **0.8** for hybrid predictions.  

4. **Pending Work**:  
   - Fine-tuning weight proportions for optimal recommendations.  
   - Implementing real-time user feedback integration for dynamic updates.  

---  

## 🎯 Planned Future Enhancements  

1. **⚙️ Advanced Hybrid Systems**:  
   - Experimenting with ensemble techniques or deep learning models for hybrid recommendations.  

2. **📈 Improved Metrics**:  
   - Introducing metrics like **precision@k** and **recall@k** for better evaluation.  

3. **📱 [Streamlit UI](https://nlp-powered-recommendation-system.streamlit.app/) Enhancements**:  
   - Adding advanced filtering options for recommendations.  
   - Visualizing preferences, trends, and popularity metrics using interactive charts.  

---  

## 🚀 Getting Started  

To set up this project locally:  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/MuhammadUmerKhan/NLP-Powered-Recommendation-System.git  


2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the recommendation system:
    ```bash
    streamlit run recommendation_system.py


## 🛠️ Prerequisites
- Python 3.x
- Required packages are listed in requirements.txt.

## 📄 Acknowledgments
- **Datasets:**
   - (MovieLens)[https://grouplens.org/datasets/movielens/] for collaborative and hybrid systems.
   - (MIT OpenCourseWare)[https://ocw.mit.edu/collections/environment/] for content-based recommendations.
