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
import os  # Import os module for directory creation

# Function to load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load JSON data
quiz_endpoint = load_json('data/Quiz_Endpoint.json')
quiz_submission = load_json('data/Quiz_Submission_Data.json')
api_endpoint = load_json('data/API_Endpoint.json')

# Convert JSON data to DataFrames and Extract Nested Fields
df_submission = pd.json_normalize(quiz_submission)
df_api = pd.json_normalize(api_endpoint)
df_quiz = pd.json_normalize(quiz_endpoint['quiz'])  # Extract 'quiz' dictionary

# Ensure necessary columns exist before merging
df_submission['quiz_id'] = df_submission['quiz.id']
df_submission['score'] = df_submission['score'].astype(float)
df_quiz['quiz_id'] = df_quiz['id']
df_api['quiz_id'] = df_api['quiz_id']
df_api['user_id'] = df_api['user_id']

# Merge DataFrames
df_combined = pd.merge(df_submission, df_quiz, on='quiz_id', how='left')
df_combined = pd.merge(df_combined, df_api, on=['quiz_id', 'user_id'], how='left')

# Debugging: Check the size of the merged DataFrame
print("Shape of df_combined:", df_combined.shape)
print("Final DataFrame Columns:", df_combined.columns)

# If df_combined has insufficient data, use mock data
if len(df_combined) <= 1:
    print("Warning: Insufficient data in df_combined. Using mock data for testing.")
    mock_data = {
        'user_id': [f'user_{i}' for i in range(100)],
        'quiz_id': np.random.randint(1, 10, 100),
        'score': np.random.randint(0, 100, 100),
        'accuracy': np.random.randint(50, 100, 100),
        'topic': np.random.choice(['Biology', 'Chemistry', 'Physics'], 100),
        'difficulty_level': np.random.randint(1, 5, 100),
        'neet_rank': np.random.randint(1, 100000, 100),
        'time_taken': np.random.randint(10, 60, 100),  # Mock time taken per quiz
        'correct_answers': np.random.randint(0, 50, 100),  # Added missing column
        'incorrect_answers': np.random.randint(0, 10, 100)  # Added missing column
    }
    df_combined = pd.DataFrame(mock_data)

# Feature Engineering
df_combined['weighted_score'] = df_combined['score'] * df_combined['difficulty_level']
df_combined['correct_to_incorrect_ratio'] = df_combined['correct_answers'] / (df_combined['incorrect_answers'] + 1)  # Avoid division by zero
df_combined['time_per_question'] = df_combined['time_taken'] / df_combined['total_questions'] if 'total_questions' in df_combined.columns else df_combined['time_taken']

# Use the correct column names
features = df_combined.groupby('user_id').agg({
    'accuracy': 'mean',
    'weighted_score': 'mean',
    'score': 'sum',
    'correct_to_incorrect_ratio': 'mean',
    'time_per_question': 'mean'
}).reset_index()

# Merge with API data for NEET rank
features = pd.merge(features, df_combined[['user_id', 'neet_rank']], on='user_id', how='left')

# Train-Test Split
if len(features) > 1 and features['neet_rank'].notna().sum() > 1:
    X = features.drop(['user_id', 'neet_rank'], axis=1)
    y = features['neet_rank']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model Training with Hyperparameter Tuning using GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best Model Parameters:", grid_search.best_params_)

    # Model Evaluation
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)
    print("Sample Predictions:", predictions)

    # Generate Visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='topic', y='accuracy', data=df_combined)
    plt.title('Accuracy by Topic')

    # Create the visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)

    # Save the plot
    plt.savefig('visualizations/topic_accuracy.png')
    plt.close()  # Close the plot to free up memory

    # Flask API
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict_rank():
        data = request.json
        # Handle missing keys with default values
        accuracy = data.get('accuracy', 0)
        weighted_score = data.get('weighted_score', 0)
        total_questions = data.get('total_questions', 0)
        correct_to_incorrect_ratio = data.get('correct_to_incorrect_ratio', 0)
        time_per_question = data.get('time_per_question', 0)

        # Validate inputs
        if not (0 <= accuracy <= 100):
            return jsonify({"error": "Accuracy must be between 0 and 100"}), 400

        features_input = [accuracy, weighted_score, total_questions, correct_to_incorrect_ratio, time_per_question]
        features_scaled = scaler.transform([features_input])
        predicted_rank = best_model.predict(features_scaled)[0]
        
        # Predict college based on rank
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
        print("Starting Flask server...")
        app.run(debug=True, host="127.0.0.1", port=5000)
else:
    print("Error: Not enough samples for train-test split. Check the data merging and grouping steps.")