# Email Spam Classifier

## Project Description

This project is a machine learning-based Email Spam Classifier
that classifies messages as Spam or Ham.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- NLP
- TF-IDF

## Machine Learning Algorithms

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Workflow

1. Load the dataset
2. Preprocess email text
3. Convert text into numerical features using TF-IDF
4. Train machine learning models
5. Compare model performance
6. Generate classification metrics
7. Display confusion matrix
8. Predict new email messages

## Results

The models are compared using accuracy, precision,
recall, F1-score and confusion matrix.

SVM achieved the best overall performance in this project.

## How to Run

Install the required libraries:

pip install -r requirements.txt

Then run:

python spam_classifier.py
