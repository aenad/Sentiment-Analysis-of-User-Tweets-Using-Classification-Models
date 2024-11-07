# Sentiment Analysis of User Tweets Using Classification Models

Overview
This project aims to perform sentiment analysis on Twitter data using different text processing and machine learning approaches. The dataset contains tweets labeled with various sentiment classes, and the goal is to classify the sentiment accurately using different models. The project involves multiple steps: data preprocessing, feature extraction, model training, and evaluation.

## Project Stages
### 1. Data Preprocessing:

#### Cleaning Data:
 Removed unnecessary columns, filtered the dataset to retain only important features (Tweet ID, Sentiment, and Tweet Content), and handled class imbalance issues by applying class weights.
#### Text Cleaning: 
Removed URLs, special characters, and stopwords. Performed stemming and lemmatization to standardize the text data.

### 2. Feature Extraction:

Used different feature extraction techniques to convert text data into numerical vectors:
#### TF-IDF (Term Frequency-Inverse Document Frequency):
#### Word Embedding:
 Used pre-trained GloVe and Word2Vec models to convert text to word vectors, capturing contextual relationships between words.

### 3. Model Training:

Trained multiple models to classify the sentiments, including:
#### Logistic Regression
#### SVM (Support Vector Machine)
#### Random Forest
#### AdaBoost
Used different evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix to assess model performance.

### 4. Evaluation and Results:

#### Random Forest model performed the best, with an accuracy of 88.4%.
Detailed metrics such as precision and recall were used for each sentiment class to understand model effectiveness.
Addressed data imbalance by tuning class weights and using robust sampling techniques.

### Techniques Used
#### Text Processing: 
Cleaned and normalized the tweet data using natural language processing techniques.
#### Feature Extraction: 
TF-IDF, GloVe, and Word2Vec were used to transform textual data into feature vectors.
#### Modeling:
 Compared several models, choosing the best one based on evaluation metrics.
### Results Summary
The Random Forest classifier provided the highest accuracy (88.4%) and balanced performance across all sentiment classes.
The feature extraction methods, especially Word2Vec and GloVe, significantly contributed to model accuracy.
Class imbalance was effectively mitigated using class weights, enhancing the model's capability to identify minority classes.
### How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
2. **Clone the Repository**:
   ```bash
   git clone <repository_url>
3. **Preprocess Data**:   
Run the preprocessing script to clean and transform the data.
3. **Train the Model**: Execute the training script to train models using different feature extraction techniques.
4.  **Evaluate**: Use the evaluation script to test models on the validation dataset.

### Dependencies
Python 3.x

Libraries: NumPy, pandas, scikit-learn, NLTK, Gensim

### Project Structure
```bash notebooks/```: Contains Jupyter notebooks used for data exploration and model building.

```scripts/:``` Python scripts for data preprocessing, training, and evaluation.

```data/:``` Directory for storing the dataset.





