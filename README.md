# 🌟 Recommendation System 🌟

Welcome to the **Recommendation System** project! This tool leverages **Content-Based**, **Collaborative Filtering**, and **Hybrid** approaches to provide personalized course recommendations tailored to user preferences. 🚀

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
This project is a recommendation system that employs **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach** to recommend courses or movies based on metadata and user interaction data. 🎓

---

## 🛠️ Project Structure

### 📌 **Content-Based Filtering**
- Analyzes metadata (e.g., course title, tags, and descriptions) to generate recommendations based on content similarity.

### 📌 **Collaborative Filtering**
- Utilizes user interaction data to recommend items based on similar users' preferences.

### 📌 **Hybrid Approach**
- Combines both content-based and collaborative filtering to deliver more accurate and personalized recommendations.

---

## 💻 Technologies Used
- **🐍 Python**: Primary language for implementing recommendation algorithms.
- **📊 Streamlit**: For deploying the interactive web application.
- **🔠 spaCy**: Natural language processing tasks.
- **🛠️ NLTK**: Text preprocessing.
- **🧮 Pandas & NumPy**: Data manipulation and numerical operations.
- **📈 Scikit-learn**: Machine learning models and evaluation metrics.

---

## ✔️ Current Work

### 📘 **Content-Based Recommendation System**
1. **Data Collection and Preprocessing**:
   - Scraped comprehensive course data from the [MIT OpenCourseWare Environment & Sustainability](https://ocw.mit.edu/collections/environment/) sections.
   - Merged datasets for "Earth Systems and Climate Science" and "Engineering" into a unified structure for recommendation purposes.

2. **Feature Engineering**:
   - Constructed a `Tags` column combining course `Description`, `Departments`, and `Topics`.
   - Vectorized the `Tags` column using TF-IDF to compute cosine similarity.

3. **Recommendations**:
   - Built a model that recommends courses similar to a given course based on content similarity.

---

### 🎥 **Collaborative Recommendation System**
1. **Data Collection and Preprocessing**:
   - Used the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/) to build a collaborative filtering system.

2. **Model Implementation**:
   - Employed **SVD (Singular Value Decomposition)** for matrix factorization and user-item rating prediction.
   - Evaluated performance using RMSE and cross-validation.

---

### 🌐 **Hybrid Recommendation System**
1. **Work Completed**:
   - Normalized predictions from both content-based and collaborative filtering models.
   - Combined the two predictions using weighted averaging (50% content-based, 50% collaborative).
   - Built a function to recommend movies/courses based on this hybrid model.

2. **Work Remaining**:
   - Fine-tune the weights for content-based and collaborative filtering.
   - Implement user feedback to adapt recommendations dynamically.
   - Integrate the hybrid system into the Streamlit app as a new interactive tab.

---

## 🎯 Planned Future Enhancements
1. **⚙️ Advanced Hybrid Systems**:
   - Experiment with ensemble techniques or deep learning models for hybrid recommendations.

2. **📈 Improved Metrics**:
   - Introduce precision@k and recall@k for better evaluation.

3. **📱 Streamlit UI Enhancements**:
   - Include advanced filtering options for recommendations.
   - Add visualizations (e.g., bar charts for preferences, popularity trends).

---

## 🚀 Getting Started

To set up this project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MuhammadUmerKhan/NLP-Powered-Recommendation-System.git
