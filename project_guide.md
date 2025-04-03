# StrokeShield: Stroke Prediction System

## Project Overview

StrokeShield is a machine learning-based system that predicts stroke risk with 90% accuracy using Random Forest and Logistic Regression models trained on a dataset of 5,000+ patients. This guide will walk you through setting up and using the system.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Web Interface](#web-interface)
6. [Using StrokeShield](#using-strokeshield)
7. [Technical Details](#technical-details)
8. [Future Enhancements](#future-enhancements)

## Project Structure

```
strokeshield/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── models/
│   ├── random_forest_model.pkl
│   └── logistic_regression_model.pkl
├── src/
│   ├── strokeshield.py
│   ├── download_dataset.py
│   └── web_app.py
├── notebooks/
│   └── model_development.ipynb
├── requirements.txt
└── README.md
```

## Installation

To set up StrokeShield, follow these steps:

1. Clone the repository (if applicable) or create a new project folder
2. Create a virtual environment and activate it:

```bash
python -m venv strokeshield-env
source strokeshield-env/bin/activate  # On Windows: strokeshield-env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

## Data Preparation

Run the data preparation script to download and prepare the dataset:

```bash
python src/download_dataset.py
```

This script will:
- Download the stroke dataset (or create a synthetic one if download fails)
- Perform basic preprocessing
- Save the dataset to the `data/` directory

The dataset includes features such as:
- age: Age of the patient
- gender: Gender of the patient
- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- heart_disease: 0 if the patient doesn't have heart disease, 1 if the patient has heart disease
- ever_married: Has the patient ever been married?
- work_type: Type of work the patient does
- Residence_type: Type of residence (Rural or Urban)
- avg_glucose_level: Average glucose level in blood
- bmi: Body Mass Index
- smoking_status: Smoking status
- stroke: 1 if the patient had a stroke, 0 if not (target variable)

## Model Training

To train the stroke prediction models, use the main StrokeShield class:

```python
from strokeshield import StrokeShield

# Initialize StrokeShield
shield = StrokeShield()

# Load and explore data
shield.load_data("data/healthcare-dataset-stroke-data.csv")
shield.explore_data()

# Preprocess data
shield.preprocess_data()

# Train models
shield.train_random_forest()
shield.train_logistic_regression()

# Evaluate models
shield.evaluate_models()

# Analyze feature importance
shield.feature_importance()

# Save models
shield.save_models("models/random_forest_model.pkl", "models/logistic_regression_model.pkl")
```

This will create two trained models:
1. A Random Forest classifier
2. A Logistic Regression classifier

Both models are optimized using grid search with cross-validation to find the best hyperparameters.

## Web Interface

To launch the web interface:

```bash
streamlit run src/web_interface.py
```

This will start a local Streamlit server (typically at http://localhost:8501) where you can interact with the StrokeShield system.

## Using StrokeShield

### From the Web Interface

The web interface provides four main sections:

1. **Home**: Overview of the StrokeShield system
2. **Data Exploration**: Visualizations and insights from the dataset
3. **Risk Assessment**: Input patient data to predict stroke risk
4. **Model Performance**: View metrics and comparisons of the models

### Programmatically

You can also use StrokeShield programmatically:

```python
from strokeshield import StrokeShield
import joblib

# Load pre-trained models
shield = StrokeShield()
shield.load_models("models/random_forest_model.pkl", "models/logistic_regression_model.pkl")

# Patient data
patient = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked'
}

# Predict stroke risk
result = shield.predict_stroke_risk(patient)

# Output results
print(f"Random Forest prediction: {result['random_forest']['prediction']} (Probability: {result['random_forest']['probability']:.2f})")
print(f"Logistic Regression prediction: {result['logistic_regression']['prediction']} (Probability: {result['logistic_regression']['probability']:.2f})")
print(f"Ensemble prediction: {result['ensemble']['prediction']} (Probability: {result['ensemble']['probability']:.2f})")
```

## Technical Details

### Data Preprocessing

- **Missing Values**: Handled using median imputation for numeric features and mode imputation for categorical features
- **Categorical Features**: Encoded using OneHotEncoder
- **Numeric Features**: Scaled using StandardScaler

### Random Forest Model

- Trained with grid search to optimize hyperparameters
- Optimized for parameters such as:
  - Number of estimators
  - Maximum depth
  - Minimum samples to split
  - Minimum samples at leaf

### Logistic Regression Model

- Trained with grid search to optimize hyperparameters
- Optimized for parameters such as:
  - Regularization strength (C)
  - Penalty type (L1/L2)
  - Solver algorithm

### Ensemble Method

- Combines predictions from both models by averaging probabilities
- More robust than either model alone

## Future Enhancements

Potential improvements to consider:

1. **Additional Models**: Incorporate more algorithms like XGBoost or Neural Networks
2. **Feature Engineering**: Create new features from existing ones to improve prediction
3. **Time-Series Analysis**: Extend to longitudinal data to predict risk over time
4. **External Validation**: Validate on external datasets to ensure generalizability
5. **Explainability**: Add SHAP values or other explainability techniques
6. **Mobile App**: Develop a mobile application for healthcare professionals
7. **API Integration**: Create an API for integration with hospital systems
