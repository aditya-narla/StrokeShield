import pandas as pd
import numpy as np
import requests
import io
import os


def download_stroke_dataset():
    """
    Downloads the healthcare stroke dataset from Kaggle
    """
    print("Downloading stroke dataset...")

    # URL for the dataset (this is a common public dataset)
    url = "https://raw.githubusercontent.com/plotly/datasets/master/stroke-data.csv"

    try:
        # Try downloading from the direct GitHub source
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.text))
            print(f"Successfully downloaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        else:
            print(f"Failed to download dataset from URL. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

    print("Using fallback method to create synthetic dataset...")
    return create_synthetic_dataset()


def create_synthetic_dataset(n_samples=5000):
    """
    Creates a synthetic stroke dataset when download fails
    """
    np.random.seed(42)

    # Generate synthetic data based on realistic distributions
    data = pd.DataFrame()

    # Generate age (higher risk for older people)
    data['age'] = np.random.normal(55, 15, n_samples).clip(18, 100).astype(int)

    # Generate gender (slightly higher risk for males)
    data['gender'] = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.51, 0.01], size=n_samples)

    # Generate hypertension (about 25% prevalence)
    data['hypertension'] = np.random.binomial(1, 0.25, n_samples)

    # Generate heart disease (about 10% prevalence)
    data['heart_disease'] = np.random.binomial(1, 0.1, n_samples)

    # Generate marriage status (about 70% married)
    data['ever_married'] = np.random.choice(['Yes', 'No'], p=[0.7, 0.3], size=n_samples)

    # Generate work type
    work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    work_probs = [0.6, 0.15, 0.15, 0.09, 0.01]
    data['work_type'] = np.random.choice(work_types, p=work_probs, size=n_samples)

    # Generate residence type (equal urban/rural)
    data['Residence_type'] = np.random.choice(['Urban', 'Rural'], size=n_samples)

    # Generate glucose level (higher for diabetics)
    # Normal: 70-140, Prediabetic: 140-200, Diabetic: 200+
    diabetic_prob = 0.1 + 0.01 * (data['age'] / 10)  # Probability increases with age
    is_diabetic = np.random.binomial(1, diabetic_prob)

    # FIX: Use numpy arrays instead of lists for the choice function
    glucose_normal = np.random.normal(100, 15, n_samples)
    glucose_high = np.random.normal(170, 30, n_samples)
    glucose_very_high = np.random.normal(250, 50, n_samples)

    # FIX: Use numpy's where function with array selections instead of random.choice
    data['avg_glucose_level'] = np.where(is_diabetic == 1,
                                         np.where(np.random.random(n_samples) > 0.5, glucose_very_high, glucose_high),
                                         glucose_normal)
    data['avg_glucose_level'] = data['avg_glucose_level'].clip(55, 300).round(2)

    # Generate BMI
    data['bmi'] = np.random.normal(28, 5, n_samples).clip(15, 50).round(1)

    # Generate smoking status
    smoking_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
    smoking_probs = [0.2, 0.5, 0.2, 0.1]
    data['smoking_status'] = np.random.choice(smoking_statuses, p=smoking_probs, size=n_samples)

    # Generate stroke status based on risk factors
    # Base probability is low (around 1%)
    base_prob = 0.01

    # Age factor (higher risk with age)
    age_factor = (data['age'] - 50) / 100
    age_factor = age_factor.clip(0, 0.5)

    # Hypertension factor
    hypertension_factor = data['hypertension'] * 0.1

    # Heart disease factor
    heart_factor = data['heart_disease'] * 0.15

    # Glucose level factor
    glucose_factor = (data['avg_glucose_level'] - 100) / 400
    glucose_factor = glucose_factor.clip(0, 0.2)

    # BMI factor
    bmi_factor = (data['bmi'] - 25) / 100
    bmi_factor = bmi_factor.clip(0, 0.1)

    # Smoking factor
    smoking_factor = np.zeros(n_samples)
    smoking_factor[data['smoking_status'] == 'smokes'] = 0.05
    smoking_factor[data['smoking_status'] == 'formerly smoked'] = 0.02

    # Calculate total probability
    stroke_prob = base_prob + age_factor + hypertension_factor + heart_factor + glucose_factor + bmi_factor + smoking_factor
    stroke_prob = stroke_prob.clip(0, 0.9)  # Cap at 90% probability

    # Generate stroke status
    data['stroke'] = np.random.binomial(1, stroke_prob)

    # Add id column
    data['id'] = np.arange(1, n_samples + 1)

    # Reorder columns to match original dataset
    cols = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
            'smoking_status', 'stroke']
    data = data[cols]

    # Introduce some missing values in BMI (about 5%)
    missing_bmi_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    data.loc[missing_bmi_indices, 'bmi'] = np.nan

    print(f"Successfully created synthetic dataset with {n_samples} samples")
    print(f"Stroke prevalence: {data['stroke'].mean():.2%}")

    return data


def prepare_dataset():
    """
    Downloads or creates the dataset and saves it to disk
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Check if dataset already exists
    if os.path.exists('data/healthcare-dataset-stroke-data.csv'):
        print("Dataset already exists, loading from disk...")
        data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
    else:
        # Download or create dataset
        data = download_stroke_dataset()

        # Save to disk
        data.to_csv('data/healthcare-dataset-stroke-data.csv', index=False)
        print("Dataset saved to disk")

    # Display basic statistics
    print("\nBasic Dataset Statistics:")
    print(f"Number of samples: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")
    print(f"Stroke prevalence: {data['stroke'].mean():.2%}")
    print(f"Age range: {data['age'].min()} to {data['age'].max()} years")
    print(f"Glucose level range: {data['avg_glucose_level'].min():.1f} to {data['avg_glucose_level'].max():.1f}")
    print(f"BMI range: {data['bmi'].min():.1f} to {data['bmi'].max():.1f}")
    print(f"Missing values:\n{data.isnull().sum()}")

    return data


if __name__ == "__main__":
    data = prepare_dataset()