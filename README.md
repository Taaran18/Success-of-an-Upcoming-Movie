# Project Name

## Description
This project is focused on text classification using a Logistic Regression model. The goal is to predict whether a given text is related to a disaster or not. The project utilizes the scikit-learn library for machine learning and pandas for data manipulation.

## Installation
To run this project, you need to have the following dependencies installed:
- pandas
- numpy
- scikit-learn

You can install the dependencies by running the following command:
```
pip install pandas numpy scikit-learn
```

## Usage
1. Clone the repository:
```
git clone https://github.com/your-username/your-repo.git
```

2. Change into the project directory:
```
cd your-repo
```

3. Import necessary libraries:
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

4. Load the datasets:
```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')
```

5. Explore the data:
```python
train_df.head()
```

6. Split the data into features and target:
```python
X = train_df['text']
y = train_df['target']
```

7. Split the training data into train and validation sets:
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

8. Create a pipeline for preprocessing and model training:
```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])
```

9. Fit the pipeline on the training data:
```python
pipeline.fit(X_train, y_train)
```

10. Display pipeline and transformers/estimators:
```python
pipeline_steps = [step[1] for step in pipeline.steps]
pipeline_names = [step[0] for step in pipeline.steps]
pipeline_df = pd.DataFrame({'Pipeline': pipeline_names, 'Transformer/Estimator': pipeline_steps})
print("Pipeline:")
print(pipeline_df)
```

11. Evaluate the model on the validation set:
```python
y_val_pred = pipeline.predict(X_val)
```

12. Calculate classification report:
```python
report = classification_report(y_val, y_val_pred, target_names=['Not Disaster', 'Disaster'])
print(report)
```

13. Calculate confusion matrix:
```python
cm = confusion_matrix(y_val, y_val_pred)
cm_df = pd.DataFrame(cm, index=['Not Disaster', 'Disaster'], columns=['Not Disaster', 'Disaster'])
print("Confusion Matrix:")
print(cm_df)
```

14. Calculate precision, recall, F1-score, and support:
```python
precision = cm.diagonal() / cm.sum(axis=0)
recall = cm.diagonal() / cm.sum(axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)
support = cm.sum(axis=1)
```

15. Create a dataframe for results:
```python
results_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-score': f1_score, 'Support': support},
                          index=['Not Disaster', 'Disaster'])
results_df.index.name = 'Class'
print("Results:")
print(results_df)
```

16. Apply the pipeline on the test data:
```python
X_test = test_df['text
