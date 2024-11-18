# 🌟 Recommendation System 🌟

Welcome to the **Recommendation System** project! This tool leverages **Content-Based** and **Collaborative Filtering** approaches to provide personalized course recommendations tailored to user preferences. 🚀

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
This project is a recommendation system that employs **Content-Based Filtering** and **Collaborative Filtering** techniques. It provides personalized course recommendations by analyzing course metadata and user interaction data. 🎓

---

## 🛠️ Project Structure

### 📌 **Content-Based Filtering**
1. **Data Collection and Preprocessing**:
   - Scraped comprehensive course data from the [MIT OpenCourseWare Environment & Sustainability](https://ocw.mit.edu/collections/environment/) sections.          This dataset was utilized to create a system that recommends courses based on content similarity. 💡
   - Scraped and combined course data for "Environment & Sustainability: Earth Systems and Climate Science" and "Environment & Sustainability: Engineering".

3. **Feature Engineering**:
   - Constructed a `Tags` column by merging `Description`, `Departments`, and `Topics` columns.
   - Cleaned strings and lists in the dataset to prepare for vectorization and similarity calculations.

### 📌 **Collaborative Filtering**
1. **Data Collection and Preprocessing**:
   - Data Collection: Used the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/), which includes user ratings for movies. This dataset enabled the creation of a recommendation system that identifies item-item similarities based on user preferences 🎥.
   - Developed and deployed a collaborative filtering model for personalized recommendations using user preferences and similar user interactions.

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
   - Combined course data from *Environment & Sustainability: Earth Systems and Climate Science* and *Environment & Sustainability: Engineering*.
   - Created a unified dataset with key columns: `['Title', 'Difficulty', 'Tags']`.

2. **Feature Engineering**:
   - Constructed a `Tags` column by merging `Description`, `Departments`, and `Topics` columns.
   - Preprocessed data for vectorization and similarity calculations.

### 🎥 **Collaborative Recommendation System**
1. **Data Collection**:
   - Used the **MovieLens 100K Dataset**, which includes user ratings for movies. This dataset helped build a model that identifies item-item similarities based on user preferences.

2. **Model Implementation**:
   - Developed and deployed a collaborative filtering model for personalized recommendations using user preferences and similar user interactions.

### 🌐 **Streamlit App**
- **Homepage**: Introduces the user and project.  
- **Interactive Tabs**: Allow users to test both **Content-Based** and **Collaborative Filtering** models.

[🚀 **Explore the App**](https://nlp-powered-recommendation-system.streamlit.app/)

---

## 🎯 Planned Future Enhancements
1. **🔗 Hybrid Recommendation System**:
   - Combine **Content-Based** and **Collaborative Filtering** for a hybrid model.

2. **📱 Streamlit App Enhancements**:
   - Add a tab for the hybrid recommendation system.
   - Improve user interface and interactivity.

3. **⚙️ Optimization and Fine-Tuning**:
   - Enhance model performance through hyperparameter tuning and feedback integration.

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
