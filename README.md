# Aries_Hackathon

This submission outlines a Multi-Target Machine Learning Pipeline designed to automate customer complaint triage using Natural Language Processing (NLP) and Ensemble Learning.
Core Methodology
The system transforms raw text into actionable business intelligence through a three-stage technical architecture:
1. Text Vectorization (TF-IDF): It uses the Term Frequency-Inverse Document Frequency algorithm to convert text into numerical data. This process highlights unique, high-value keywords while filtering out common "stop words" that carry little meaning.
2. Categorization (Random Forest): Two Random Forest Classifiers predict the primary_category and secondary_category. By aggregating the "votes" of 100 individual decision trees, the model remains robust against "noisy" text and avoids overfitting.
3. Severity Estimation (Random Forest Regressor): The system predicts urgency on a continuous scale. To align with business logic, these scores are rounded and clipped to a strict 1–5 integer scale.
System Strengths
* Reliability: Includes automated handling for missing text to prevent runtime errors.
* Efficiency: Restricts the model to the top 5,000 text features, balancing high performance with fast processing speeds.
* Consistency: Implements a fixed random_state to ensure reproducible results across different environments.
Expected Output
The pipeline generates a submission.csv containing the unique complaint_id, the predicted department and sub-issue, and a finalized severity rating for management review.
