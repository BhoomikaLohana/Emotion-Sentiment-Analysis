# Sentiment Analysis Using Machine Learning and LLM
## Objective
This project aims to build a system that combines Machine Learning (ML) and Large Language Model (LLM) to classify the emotions in tweets and analyze their sentiment. Using Logistic Regression, the system classifies tweets into six emotions: Anger, Fear, Joy, Love, Sadness, and Surprise. It also uses the OpenAi API to determine whether the sentiment of each tweet is Positive, Negative, or Neutral. The goal is to compare the predicted emotions with the sentiment to see how they match. The model’s performance is evaluated using accuracy and other metrics, and the results are visualized to show the sentiment distribution across different emotions.

## Project Overview
This project aims to classify tweets into six emotional categories (Fear, Anger, Joy, Love, Sadness, and Surprise) and analyze their sentiment (Positive, Negative, Neutral) using Logistic Regression for emotion classification and OpenAi API Model Gpt-4 for sentiment analysis.

The model leverages TF-IDF vectorization to process the text data and Logistic Regression to predict the emotions. Additionally, OpenAi API is used to extract sentiment labels from the tweets.

## Technologies Used
Python
scikit-learn (for Logistic Regression and TF-IDF vectorization)
OpenAi API (for sentiment analysis)
pandas (for data manipulation)
matplotlib & seaborn (for visualizations)

## Features
Emotion Classification: Classifies tweets into one of the six emotions (Fear, Anger, Joy, Love, Sadness, Surprise).
Sentiment Analysis: Uses the OpenAi API to assign a sentiment label to each tweet (Positive, Negative, Neutral).
Data Visualizations: Visualizes the distribution of sentiment across different emotions using pie charts and bar plots.
Model Evaluation: Evaluates the Logistic Regression model with accuracy, precision, recall, and F1-score metrics.

## Dataset
The dataset used for this project contains tweets with labeled emotions.
Link to Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/emotions

## How to run the code
preprocess.py:
Preprocess the data (cleaning, text preparation, etc.) and get it ready for further analysis.

LLM_configuration.py:
Configure and set up OpenAi model Gpt-4 for sentiment analysis on the tweets.

main.py:
Execute core tasks such as training the Logistic Regression model, extracting sentiment labels, and running the overall workflow.
text_analysis.ipynb:

Visualize the results, train the ML model, and analyze the sentiment data obtained from OpenAi API.
Generate necessary visualizations (such as sentiment distribution across emotions) and evaluate the model's performance.
Make sure to update file paths and insert your OpenAi API key in the appropriate places in the scripts.

## Results
The Logistic Regression model performed well in classifying tweets into their respective emotions, achieving 90% accuracy. It showed strong performance for emotions like Anger, Sadness, and Fear, which are more clearly associated with specific sentiments.

Sentiment Analysis using OpenAI API was successful in categorizing the tweets as Positive, Negative, or Neutral. The sentiment labels generally aligned well with the predicted emotions, especially for Negative emotions like Fear and Anger, which were mostly predicted as Negative sentiment.

The visualizations revealed interesting insights, such as:

Fear, Sadness, and Anger having mostly Negative sentiments.
Joy and Love showing a mix of Neutral and Positive sentiments, indicating the complex nature of these emotions.
Surprise showed a balanced split between Negative and Positive sentiments, suggesting that surprise can be perceived in both positive and negative contexts.
The analysis of sentiment distribution across emotions showed that the Logistic Regression model was effective in capturing the key emotional signals, while the OpenAi API enhanced the understanding of the overall tone (sentiment) in each tweet.








