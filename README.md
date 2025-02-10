# NEET Rank Prediction

## Table of Contents

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

---

## Project Overview

The **NEET Rank Prediction** project is designed to predict the ranks of candidates appearing for the NEET (National Eligibility cum Entrance Test) based on their performance metrics. Using historical performance data, this system provides insights into rank predictions, enabling candidates to analyze their potential outcomes and strategize accordingly.

This project leverages advanced machine learning techniques to ensure accuracy and reliability. It also provides visualizations for better understanding and analysis.

---

## Features

- **Rank Prediction**: Predict NEET ranks based on accuracy, weighted scores, and other metrics.
- **Model Training**: Uses advanced machine learning models with hyperparameter tuning for optimal performance.
- **Data Visualization**: Generates insightful plots to visualize trends and accuracy.
- **Customizable Inputs**: Accepts custom inputs for predictions.

---

## Installation

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

## Usage

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

## Project Structure

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

## Data Description

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

## Model and Prediction Pipeline

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

## Visualization

The script generates plots to provide better insights into the data. Example plots include:

- **Accuracy Trends**: Visual representation of candidate accuracy.
- **Score Distributions**: Insights into the weighted scores.

Visualizations are saved in the `visualizations/` directory, e.g., `visualizations/topic_accuracy.png`.

---

## Dependencies

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

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with detailed explanations of your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

