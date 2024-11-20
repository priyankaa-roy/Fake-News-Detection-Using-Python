# Detecting Fake News 📰🔍
This repository contains a Python-based solution to detect fake news using a dataset from Kaggle. Leveraging natural language processing (NLP) techniques and machine learning algorithms, the project identifies fake news articles with high accuracy.
# Project Overview
Fake news has become a widespread issue in the digital age, impacting public opinion and decision-making. This project aims to classify news articles as real or fake by training a predictive model on labeled datasets. The pipeline includes data preprocessing, feature extraction, model training, and evaluation.

# Features
- Data cleaning and preprocessing (handling missing values, text normalization)
- Feature extraction using TF-IDF vectorization
- Machine learning classification (e.g., Logistic Regression, Naive Bayes)
- Model evaluation (accuracy, precision, recall, and F1 score)
- Python notebook for reproducibility

# Dataset
The dataset used for this project is sourced from Kaggle and contains labeled news articles categorized as real or fake.

Dataset: Use Kaggle to download Fake News Dataset

# Installation

1. Clone the repository:

git clone [https://github.com/your-username/fake-news-detection.git](https://github.com/priyankaa-roy/Fake-News-Detection-Using-Python)
cd fake-news-detection

2. Set up a virtual environment:

python -m venv env
source env/bin/activate      # On Windows: env\Scripts\activate

3. Install required dependencies:

pip install -r requirements.txt

5. Download the Kaggle dataset and place it in the data folder.


# Methodology

1. Data Preprocessing:
- Removing null values and duplicates
- Text cleaning: lowercasing, removing punctuation, and stopword removal

2. Feature Extraction:
- Using TF-IDF vectorization to convert text data into numerical form for modeling

3. Model Training:
- Training classifiers such as Logistic Regression, Naive Bayes, or Support Vector Machines

4. Model Evaluation:
- Evaluating the model's performance using metrics like accuracy, precision, recall, and F1 score

# Results
Metric	Value

Accuracy	93%

Precision	93%

Recall	94%

F1 Score	93%


# Technologies Used
- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
- Modeling Techniques: Logistic Regression, Naive Bayes, TF-IDF
- Visualization: Matplotlib, Seaborn

# Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.
1. Fork the repository
2. Create a new branch: git checkout -b feature-branch-name
3. Commit your changes: git commit -m "Add feature"
4. Push to the branch: git push origin feature-branch-name
5. Open a pull request

# License
This project is licensed under the MIT License. See the LICENSE file for details.
