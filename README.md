# NEET Rank Prediction 🎯

## Table of Contents 📚

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Data Description](#data-description)
7. [Model and Prediction Pipeline](#model-and-prediction-pipeline)
8. [Visualization](#visualization)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [License](#license)
12. [Script Explanation](#script-explanation)

---

## Project Overview 🌟

The **NEET Rank Prediction** project is designed to predict the ranks of candidates appearing for the NEET (National Eligibility cum Entrance Test) based on their performance metrics. Using historical performance data, this system provides insights into rank predictions, enabling candidates to analyze their potential outcomes and strategize accordingly.

This project leverages advanced machine learning techniques to ensure accuracy and reliability. It also provides visualizations for better understanding and analysis.

---

## Features ✨

- **Rank Prediction**: Predict NEET ranks based on accuracy, weighted scores, and other metrics.
- **Model Training**: Uses advanced machine learning models with hyperparameter tuning for optimal performance.
- **Data Visualization**: Generates insightful plots to visualize trends and accuracy.
- **Customizable Inputs**: Accepts custom inputs for predictions.

---

## Installation 🛠️

### Prerequisites

Ensure you have Python 3.13.2 or later installed.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neet-rank-prediction.git
   cd neet-rank-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the directory structure:
   ```bash
   mkdir -p visualizations
   ```
4. Run the script:
   ```bash
   python scripts/main_pipeline.py
   ```

---

## Usage 🚀

1. **Run the Main Pipeline**:
   Execute the following command:
   ```bash
   python scripts/main_pipeline.py
   ```
   The script will process the data, train the model, and output predictions and visualizations.

2. **Input Custom Data**:
   Modify the input data in `data/input_data.json` and rerun the script.

3. **Check Outputs**:
   - Predictions will be printed to the console.
   - Visualizations will be saved in the `visualizations` directory.

---

## Project Structure 🗂️

```
neet-rank-prediction/
|-- data/
|   |-- input_data.json          # Input data for predictions
|-- scripts/
|   |-- main_pipeline.py         # Main script for processing and predictions
|-- visualizations/              # Directory for saving plots
|-- requirements.txt             # Dependencies
|-- README.md                    # Project documentation
```

---

## Data Description 📊

### Input Metrics
- **Accuracy**: Overall percentage of correct answers.
- **Weighted Score**: Score calculated based on the NEET marking scheme.
- **Total Questions**: Total number of questions attempted.
- **Correct-to-Incorrect Ratio**: Ratio of correct answers to incorrect answers.
- **Time Per Question**: Average time spent on each question.

### Example Input
```json
{
  "accuracy": 95,
  "weighted_score": 600,
  "total_questions": 200,
  "correct_to_incorrect_ratio": 4,
  "time_per_question": 0.5
}
```

---

## Model and Prediction Pipeline 🔍

1. **Data Preprocessing**:
   - Combine and clean datasets.
   - Handle missing values and outliers.

2. **Model Training**:
   - Use XGBoost for regression tasks.
   - Perform hyperparameter tuning with grid search.

3. **Prediction**:
   - Generate predictions for NEET ranks.

4. **Evaluation**:
   - Calculate Mean Squared Error (MSE) to evaluate model performance.

---

## Visualization 📈

The script generates plots to provide better insights into the data. Example plots include:

- **Accuracy Trends**: Visual representation of candidate accuracy.
- **Score Distributions**: Insights into the weighted scores.

Visualizations are saved in the `visualizations/` directory, e.g., `visualizations/topic_accuracy.png`.

---

## Dependencies 📦

The project relies on the following Python libraries:

- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**

Install these dependencies via:
```bash
pip install -r requirements.txt
```

---

## Contributing 🤝

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with detailed explanations of your changes.

---

## License 📜

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Script Explanation 📝

### Overview

The `scripts/main_pipeline.py` script is the core of the NEET Rank Prediction project. It handles data loading, preprocessing, feature engineering, model training, evaluation, and setting up a Flask API for predictions.

### Sections

1. **Imports**: The script starts by importing necessary libraries such as `pandas`, `numpy`, `json`, `sklearn`, `flask`, `matplotlib`, and `seaborn`.

2. **Loading JSON Data**: The script defines a function `load_json` to load JSON files. It then loads data from `Quiz_Endpoint.json`, `Quiz_Submission_Data.json`, and `API_Endpoint.json`.

3. **Data Preprocessing**: The script converts JSON data to DataFrames, extracts nested fields, and merges the DataFrames to create a combined dataset. It also handles cases where the combined dataset has insufficient data by using mock data.

4. **Feature Engineering**: The script creates new features such as `weighted_score`, `correct_to_incorrect_ratio`, and `time_per_question`. It then groups the data by `user_id` and calculates aggregate features.

5. **Model Training**: The script splits the data into training and testing sets, scales the features, and trains a `GradientBoostingRegressor` model with hyperparameter tuning using `GridSearchCV`.

6. **Model Evaluation**: The script evaluates the model using Mean Squared Error (MSE) and prints sample predictions.

7. **Visualization**: The script generates visualizations using `matplotlib` and `seaborn`, and saves the plots in the `visualizations` directory.

8. **Flask API**: The script sets up a Flask API with an endpoint `/predict` to accept POST requests with input data and return predicted NEET ranks and colleges.

### Detailed Explanation

- **Imports**: 
  ```python
  import pandas as pd
  import numpy as np
  import json
  from sklearn.model_selection import train_test_split, GridSearchCV
  from sklearn.ensemble import GradientBoostingRegressor
  from sklearn.metrics import mean_squared_error
  from sklearn.preprocessing import StandardScaler
  from flask import Flask, request, jsonify
  import matplotlib.pyplot as plt
  import seaborn as sns
  import os
  ```

- **Loading JSON Data**:
  ```python
  def load_json(file_path):
      with open(file_path, 'r') as f:
          return json.load(f)

  quiz_endpoint = load_json('data/Quiz_Endpoint.json')
  quiz_submission = load_json('data/Quiz_Submission_Data.json')
  api_endpoint = load_json('data/API_Endpoint.json')
  ```

- **Data Preprocessing**:
  ```python
  df_submission = pd.json_normalize(quiz_submission)
  df_api = pd.json_normalize(api_endpoint)
  df_quiz = pd.json_normalize(quiz_endpoint['quiz'])

  df_submission['quiz_id'] = df_submission['quiz.id']
  df_submission['score'] = df_submission['score'].astype(float)
  df_quiz['quiz_id'] = df_quiz['id']
  df_api['quiz_id'] = df_api['quiz_id']
  df_api['user_id'] = df_api['user_id']

  df_combined = pd.merge(df_submission, df_quiz, on='quiz_id', how='left')
  df_combined = pd.merge(df_combined, df_api, on=['quiz_id', 'user_id'], how='left')

  if len(df_combined) <= 1:
      mock_data = {
          'user_id': [f'user_{i}' for i in range(100)],
          'quiz_id': np.random.randint(1, 10, 100),
          'score': np.random.randint(0, 100, 100),
          'accuracy': np.random.randint(50, 100, 100),
          'topic': np.random.choice(['Biology', 'Chemistry', 'Physics'], 100),
          'difficulty_level': np.random.randint(1, 5, 100),
          'neet_rank': np.random.randint(1, 100000, 100),
          'time_taken': np.random.randint(10, 60, 100),
          'correct_answers': np.random.randint(0, 50, 100),
          'incorrect_answers': np.random.randint(0, 10, 100)
      }
      df_combined = pd.DataFrame(mock_data)
  ```

- **Feature Engineering**:
  ```python
  df_combined['weighted_score'] = df_combined['score'] * df_combined['difficulty_level']
  df_combined['correct_to_incorrect_ratio'] = df_combined['correct_answers'] / (df_combined['incorrect_answers'] + 1)
  df_combined['time_per_question'] = df_combined['time_taken'] / df_combined['total_questions'] if 'total_questions' in df_combined.columns else df_combined['time_taken']

  features = df_combined.groupby('user_id').agg({
      'accuracy': 'mean',
      'weighted_score': 'mean',
      'score': 'sum',
      'correct_to_incorrect_ratio': 'mean',
      'time_per_question': 'mean'
  }).reset_index()

  features = pd.merge(features, df_combined[['user_id', 'neet_rank']], on='user_id', how='left')
  ```

- **Model Training**:
  ```python
  if len(features) > 1 and features['neet_rank'].notna().sum() > 1:
      X = features.drop(['user_id', 'neet_rank'], axis=1)
      y = features['neet_rank']

      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)

      X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

      model = GradientBoostingRegressor(random_state=42)
      param_grid = {
          'n_estimators': [100, 200],
          'learning_rate': [0.01, 0.1],
          'max_depth': [3, 5]
      }
      grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
      grid_search.fit(X_train, y_train)

      best_model = grid_search.best_estimator_
      predictions = best_model.predict(X_test)
      mse = mean_squared_error(y_test, predictions)
  ```

- **Visualization**:
  ```python
  plt.figure(figsize=(10, 6))
  sns.barplot(x='topic', y='accuracy', data=df_combined)
  plt.title('Accuracy by Topic')
  os.makedirs('visualizations', exist_ok=True)
  plt.savefig('visualizations/topic_accuracy.png')
  plt.close()
  ```

- **Flask API**:
  ```python
  app = Flask(__name__)

  @app.route('/predict', methods=['POST'])
  def predict_rank():
      data = request.json
      accuracy = data.get('accuracy', 0)
      weighted_score = data.get('weighted_score', 0)
      total_questions = data.get('total_questions', 0)
      correct_to_incorrect_ratio = data.get('correct_to_incorrect_ratio', 0)
      time_per_question = data.get('time_per_question', 0)

      if not (0 <= accuracy <= 100):
          return jsonify({"error": "Accuracy must be between 0 and 100"}), 400

      features_input = [accuracy, weighted_score, total_questions, correct_to_incorrect_ratio, time_per_question]
      features_scaled = scaler.transform([features_input])
      predicted_rank = best_model.predict(features_scaled)[0]

      def rank_to_college(rank):
          if rank < 5000:
              return "AIIMS Delhi"
          elif rank < 10000:
              return "Maulana Azad Medical College"
          elif rank < 20000:
              return "Grant Medical College"
          else:
              return "Other Colleges"

      predicted_college = rank_to_college(predicted_rank)
      return jsonify({"predicted_rank": predicted_rank, "predicted_college": predicted_college})

  if __name__ == "__main__":
      app.run(debug=True, host="127.0.0.1", port=5000)
  ```

This detailed explanation should help users understand the functionality and purpose of each section of the `scripts/main_pipeline.py` script.
