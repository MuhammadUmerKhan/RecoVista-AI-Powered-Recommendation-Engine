# Content-Based Recommendation System for MIT OpenCourseWare

This project is a content-based recommendation system that suggests MIT OpenCourseWare courses in the domains of "Environment & Sustainability: Earth Systems and Climate Science" and "Environment & Sustainability: Engineering." This project was built to showcase a content-based recommendation model that uses course attributes to suggest relevant courses, with the goal of demonstrating a recommendation system for educational data.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Methods](#methods)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Requirements](#requirements)
9. [Acknowledgments](#acknowledgments)

---

## Introduction

This recommendation system focuses on suggesting courses based on course attributes such as title, difficulty, description, department, and topics. The purpose of this project is to showcase my ability to build a recommendation system using a content-based approach, which can be applied to other domains as well.

## Features

- **Content-based Filtering:** Recommends courses based on similarity between course attributes.
- **Combined Dataset:** Includes course data from two domains, "Earth Systems and Climate Science" and "Engineering."
- **Scalable Design:** The system is designed to be deployed on a web application.

## Dataset

The dataset used for this project was scraped from MIT OpenCourseWare. It includes the following features for each course:

- **Title:** The name of the course.
- **Difficulty:** The course difficulty level.
- **Description:** A brief course description.
- **Departments:** Departments offering the course.
- **Topics:** Topics covered in the course.

## Methods

1. **Data Collection and Preprocessing (Compeleted Tasks)**
   - Data was collected through web scraping, and features like titles, descriptions, and topics were extracted.
   - Text preprocessing was applied to normalize the text fields.

2. **Feature Engineering (Ongoing Task)**
   - Using NLP techniques to create vectors from text data (course title, description, etc.).
   - Applying TF-IDF to the description to capture relevant keywords.

3. **Modeling (Ongoing Task)**
   - Using cosine similarity as the similarity metric.
   - Generating similarity scores to recommend courses based on the most similar attributes.

4. **Evaluation (Ongoing Task)**
   - Evaluating recommendations based on relevance to selected courses.

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/MuhammadUmerKhan/NLP-Powered-Recommendation-System.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the recommendation system (soon):
    ```bash
    python recommend.py
    ```

## Results

The recommendation system provides relevant courses based on similarity in descriptions, topics, and difficulty. Sample recommendations are included in the results.

## Future Work

- **Collaborative Filtering:** Extend the recommendation system to include collaborative filtering.
- **Hybrid Model:** Combine content-based and collaborative filtering approaches.
- **Deployment:** Deploy as a web application using Streamlit.

## Requirements

- Python 3.x
- Required packages are listed in `requirements.txt`.

## Acknowledgments

Thanks to MIT OpenCourseWare for providing free access to high-quality educational content.

