# Content-Based and Collaborative Recommendation System

Welcome to the Recommendation System project! This tool leverages content-based and collaborative filtering approaches to provide personalized course recommendations based on various factors.

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
This project is a recommendation system designed to provide tailored course recommendations. It uses content-based filtering and collaborative filtering techniques to cater to users' unique interests by analyzing metadata and user interaction data.

## Project Structure

- **Content-Based Filtering**: The model analyzes course metadata (e.g., course title, tags, and descriptions) to generate recommendations.
- **Collaborative Filtering**: The system uses user interaction data to recommend courses based on similar users' interests.

## Technologies Used
- **Python**: Primary language for implementing recommendation algorithms.
- **Streamlit**: Used to deploy the interactive web application.
- **spaCy**: For natural language processing tasks.
- **NLTK**: For text preprocessing.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-learn**: Machine learning models and evaluation metrics.

## Current Work
The project includes the following completed tasks:

## Content Based Recommendation:

1. **Data Collection and Preprocessing**:
   - Scraped and combined course data for "Environment & Sustainability: Earth Systems and Climate Science" and "Environment & Sustainability: Engineering".
   - Created a unified dataset with key columns: `['Title', 'Difficulty', 'Tags']`.

2. **Feature Engineering**:
   - Constructed a `Tags` column by merging `Description`, `Departments`, and `Topics` columns.
   - Cleaned strings and lists in the dataset to prepare for vectorization and similarity calculations.

## Content-Based Recommendation System:
   1. **Data Collection and Preprocessing**:
   - Data Collection: Used the MovieLens 100K Dataset, which includes user ratings for movies. This dataset enabled the creation of a recommendation system that identifies item-item similarities based on user preferences 🎥.
   - Developed and deployed a collaborative filtering model for personalized recommendations using user preferences and similar user interactions.

[**Streamlit App**](https://nlp-powered-recommendation-system.streamlit.app/):
   - Designed a homepage to introduce the user and project.
   - Integrated both content-based and collaborative filtering models, providing interactive tabs for users to test each system.

## Planned Future Enhancements

1. **Hybrid Recommendation System**:
   - Combine content-based and collaborative filtering for a hybrid model.

2. **Streamlit App Enhancements**:
   - Add a tab for the hybrid recommendation system.
   - Improve the app’s user interface.

3. **Optimization and Fine-Tuning**:
   - Test and improve model performance with hyperparameter tuning.

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
### Prerequisites

Ensure you have Python 3.x installed. Required packages are listed in `requirements.txt`.

## Acknowledgments

Thanks to MIT OpenCourseWare for providing free access to high-quality educational content.

---

This README file should serve as a comprehensive guide to your project, making it easy for viewers to understand the scope, progress, and future plans for your recommendation system. Let me know if you'd like further customizations!
