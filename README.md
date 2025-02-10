# 📊 NEET Rank Predictor 🎓

Welcome to the **NEET Rank Predictor**! This project is designed to predict a student's NEET rank based on their quiz performance and suggest the most likely college they could get into. Whether you're a student preparing for the exam or an educator analyzing performance trends, this tool provides accurate and insightful predictions. 🏥

---

## 🌟 Features

- **Rank Prediction**: Predicts NEET rank using quiz performance data. 📈
- **College Suggestion**: Recommends colleges based on predicted rank. 🏫
- **Visual Insights**: Generates visualizations for topic-wise accuracy and performance trends. 📊
- **Flask API**: Exposes predictions via a REST API for easy integration. 🌐
- **Mock Data Support**: Automatically generates mock data for testing when real data is insufficient. 🛠️

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neet-rank-predictor.git
cd neet-rank-predictor
```

### 2. Install Dependencies

Ensure Python 3.13.2 or later is installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the Flask server by running the main script:

```bash
python scripts/main_pipeline.py
```

### 4. Access the API

The Flask API will start at `http://127.0.0.1:5000`. You can send a POST request to `http://127.0.0.1:5000/predict` with the following JSON body:

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

## 🧠 How It Works

### Data Pipeline 🚀

1. **Data Loading**:
   - Loads data from three JSON files:
     - `Quiz_Endpoint.json`: Contains quiz details.
     - `Quiz_Submission_Data.json`: Contains user quiz submissions.
     - `API_Endpoint.json`: Contains historical quiz data and NEET ranks.

2. **Data Merging**:
   - Merges datasets using `quiz_id` and `user_id` to create a combined dataset.

3. **Feature Engineering**:
   - Creates new features, such as:
     - `weighted_score`: Combines score and difficulty level.
     - `correct_to_incorrect_ratio`: Measures the ratio of correct to incorrect answers.
     - `time_per_question`: Calculates the average time taken per question.

4. **Model Training**:
   - Trains a Gradient Boosting Regressor on the dataset to predict NEET ranks.
   - Optimizes model performance using GridSearchCV.

5. **Flask API**:
   - Exposes the trained model via a REST API for easy access and predictions.

---

## 📊 Visualizations

The script generates insightful visualizations to analyze performance trends. Example visualizations include:

- **Topic Accuracy**: Bar chart showing accuracy by topic (`topic_accuracy.png`).
- **Score Trends**: Line charts visualizing weighted score distributions.

All visualizations are saved in the `visualizations/` folder.

---

## 🚀 Example API Response

Here’s an example of what the API might return:

```json
{
  "predicted_rank": 23718.22,
  "predicted_college": "Other Colleges"
}
```

---

## 🛠️ Troubleshooting

### Common Issues

- **Insufficient Data**:
  - If the merged dataset has insufficient data, the script will automatically generate mock data for testing.

- **Missing Columns**:
  - If required columns (e.g., `topic`, `difficulty_level`) are missing, the script will skip related analyses and use default values.

- **Visualization Errors**:
  - If the `visualizations/` directory does not exist, the script will create it automatically.

---

## 🎁 Bonus Features

- **Input Validation**: Ensures inputs are within valid ranges (e.g., `accuracy` must be between 0 and 100).
- **Mock Data**: Automatically generates data for testing when real data is insufficient.
- **Hyperparameter Tuning**: Optimizes model performance using GridSearchCV.

---

## 🤝 Contributing

Contributions are always welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with detailed explanations of your changes.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

Special thanks to the NEET Testline app for inspiring this project and to the open-source community for tools like Pandas, Scikit-learn, and Flask.

Enjoy using the **NEET Rank Predictor**! Let us know if you have any questions or suggestions. 🚀



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
