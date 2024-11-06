# Content-Based, Collaborative, and Hybrid Recommendation System

Welcome to the Recommendation System project! This tool leverages content-based filtering, collaborative filtering, and hybrid approaches to provide personalized course recommendations based on various factors.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Current Work](#current-work)
- [Planned Future Enhancements](#planned-future-enhancements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Overview
This project is a multi-faceted recommendation system designed to provide tailored course recommendations. Leveraging both content-based filtering and collaborative filtering techniques, the recommendation system caters to users' unique interests by analyzing the metadata of courses, user interactions, and hybrid methodologies.

## Project Structure

- **Content-Based Filtering**: The model analyzes course metadata (e.g., course title, tags, and descriptions) to generate recommendations.
- **Collaborative Filtering**: Upcoming feature; will use user interaction data to recommend courses based on similar users' interests.
- **Hybrid Filtering**: Upcoming feature; combines both content and collaborative filtering to deliver robust and diverse recommendations.

## Technologies Used
- **Python**: Primary language for implementing recommendation algorithms.
- **Streamlit**: Used to deploy the interactive web application.
- **spaCy**: For natural language processing tasks.
- **NLTK**: For text preprocessing.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-learn**: Machine learning models and evaluation metrics.

## Current Work
The project is progressing well with the following completed tasks:

1. **Data Collection and Preprocessing**:
   - Scraped and combined course data for "Environment & Sustainability: Earth Systems and Climate Science" and "Environment & Sustainability: Engineering".
   - Created a unified dataset with key columns: `['Title', 'Difficulty', 'Tags']`.
   
2. **Feature Engineering**:
   - Constructed a `Tags` column by merging `Description`, `Departments`, and `Topics` columns.
   - Cleaned strings and lists in the dataset to prepare for vectorization and similarity calculations.
   
3. **Content-Based Recommendation System**:
   - Built a preliminary content-based model using `Tags` for recommending courses based on textual similarity.

4. **Streamlit App**:
   - Designed a homepage to introduce the user and project.
   - Will soon integrate the content-based model for users to interact with the recommendation engine.

## Planned Future Enhancements

1. **Collaborative Filtering**:
   - Implement user-based and item-based collaborative filtering to analyze user behaviors and preferences.
   - Use matrix factorization techniques to enhance scalability and recommendation accuracy.

2. **Hybrid Recommendation System**:
   - Combine content-based and collaborative filtering for a hybrid approach, capturing both content relevance and user preference patterns.

3. **Advanced Streamlit Deployment**:
   - Add tabs for each recommendation type (content-based, collaborative, and hybrid) within the Streamlit app.
   - Improve the app’s user interface and make results more intuitive.

4. **Optimization and Fine-Tuning**:
   - Test and improve model performance with hyperparameter tuning and additional NLP techniques.
   - Incorporate user feedback for real-time recommendation adjustments.

## Getting Started

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
    ```

## Results

The recommendation system provides relevant courses based on similarity in descriptions, topics, and difficulty. Sample recommendations are included in the results.

## Future Workrecommend

- **Collaborative Filtering:** Extend the recommendation system to include collaborative filtering.
- **Hybrid Model:** Combine content-based and collaborative filtering approaches.
- **Deployment:** Deploy as a web application using Streamlit.

## Requirements

- Python 3.x
- Required packages are listed in `requirements.txt`.

## Acknowledgments

Thanks to MIT OpenCourseWare for providing free access to high-quality educational content.


This README file should serve as a comprehensive guide to your project, making it easy for viewers to understand the scope, progress, and future plans for your recommendation system. Let me know if you'd like further customizations!
