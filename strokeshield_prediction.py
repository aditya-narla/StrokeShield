import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class StrokeShield:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.lr_model = None
        self.preprocessor = None
        
    def load_data(self, file_path):
        """
        Load the stroke dataset from a CSV file
        """
        print("Loading dataset...")
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        return self
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        print("\n===== DATASET INFORMATION =====")
        print(self.data.info())
        
        print("\n===== STATISTICAL SUMMARY =====")
        print(self.data.describe())
        
        print("\n===== MISSING VALUES =====")
        print(self.data.isnull().sum())
        
        print("\n===== TARGET VARIABLE DISTRIBUTION =====")
        stroke_counts = self.data['stroke'].value_counts()
        print(stroke_counts)
        print(f"Stroke percentage: {stroke_counts[1] / len(self.data) * 100:.2f}%")
        
        # Visualizations can be added here
        return self
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        print("\n===== PREPROCESSING DATA =====")
        
        # Drop id column if it exists
        if 'id' in self.data.columns:
            self.data.drop('id', axis=1, inplace=True)
            
        # Define numeric and categorical features
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = [feat for feat in numeric_features if feat != 'stroke']
        
        categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split the data
        X = self.data.drop('stroke', axis=1)
        y = self.data['stroke']
        
        # Handle class imbalance (optional)
        # from imblearn.over_sampling import SMOTE
        # smote = SMOTE(random_state=42)
        # X, y = smote.fit_resample(X, y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        return self
    
    def train_random_forest(self):
        """
        Train a Random Forest model
        """
        print("\n===== TRAINING RANDOM FOREST MODEL =====")
        
        # Create pipeline with preprocessing and model
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define hyperparameters for tuning
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(self.X_train, self.y_train)
        
        # Save the best model
        self.rf_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self
    
    def train_logistic_regression(self):
        """
        Train a Logistic Regression model
        """
        print("\n===== TRAINING LOGISTIC REGRESSION MODEL =====")
        
        # Create pipeline with preprocessing and model
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        # Define hyperparameters for tuning
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            lr_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(self.X_train, self.y_train)
        
        # Save the best model
        self.lr_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self
    
    def evaluate_models(self):
        """
        Evaluate both models and compare their performance
        """
        print("\n===== MODEL EVALUATION =====")
        
        models = {
            'Random Forest': self.rf_model,
            'Logistic Regression': self.lr_model
        }
        
        for name, model in models.items():
            print(f"\n--- {name} Model ---")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # ROC curve and AUC
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                print(f"AUC: {roc_auc:.4f}")
            except:
                print("Could not compute ROC curve.")
        
        return self
    
    def feature_importance(self):
        """
        Analyze feature importance for Random Forest model
        """
        if self.rf_model is not None:
            print("\n===== FEATURE IMPORTANCE =====")
            
            # Get feature names from preprocessor
            cat_features = self.rf_model.named_steps['preprocessor'].transformers_[1][2]
            cat_feature_names = self.rf_model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(cat_features)
            
            num_features = self.rf_model.named_steps['preprocessor'].transformers_[0][2]
            
            feature_names = list(num_features) + list(cat_feature_names)
            
            # Get feature importances
            importances = self.rf_model.named_steps['classifier'].feature_importances_
            
            # Sort feature importances
            indices = np.argsort(importances)[::-1]
            
            # Print top 10 features
            print("Top 10 features:")
            for i in range(min(10, len(feature_names))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return self
    
    def predict_stroke_risk(self, patient_data):
        """
        Predict stroke risk for a new patient
        """
        # Convert patient data to DataFrame
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Make predictions with both models
        rf_pred = self.rf_model.predict(patient_df)
        rf_prob = self.rf_model.predict_proba(patient_df)[:, 1]
        
        lr_pred = self.lr_model.predict(patient_df)
        lr_prob = self.lr_model.predict_proba(patient_df)[:, 1]
        
        # Combine predictions (ensemble)
        ensemble_prob = (rf_prob + lr_prob) / 2
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        results = {
            'random_forest': {
                'prediction': rf_pred[0],
                'probability': rf_prob[0]
            },
            'logistic_regression': {
                'prediction': lr_pred[0],
                'probability': lr_prob[0]
            },
            'ensemble': {
                'prediction': ensemble_pred[0],
                'probability': ensemble_prob[0]
            }
        }
        
        return results
    
    def save_models(self, rf_path, lr_path):
        """
        Save trained models to disk
        """
        import joblib
        
        print("\n===== SAVING MODELS =====")
        joblib.dump(self.rf_model, rf_path)
        print(f"Random Forest model saved to {rf_path}")
        
        joblib.dump(self.lr_model, lr_path)
        print(f"Logistic Regression model saved to {lr_path}")
        
        return self
    
    def load_models(self, rf_path, lr_path):
        """
        Load trained models from disk
        """
        import joblib
        
        print("\n===== LOADING MODELS =====")
        self.rf_model = joblib.load(rf_path)
        print(f"Random Forest model loaded from {rf_path}")
        
        self.lr_model = joblib.load(lr_path)
        print(f"Logistic Regression model loaded from {lr_path}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Initialize StrokeShield
    stroke_shield = StrokeShield()
    
    # Load and explore data
    stroke_shield.load_data("data/healthcare-dataset-stroke-data.csv")
    stroke_shield.explore_data()
    
    # Preprocess data
    stroke_shield.preprocess_data()
    
    # Train models
    stroke_shield.train_random_forest()
    stroke_shield.train_logistic_regression()
    
    # Evaluate models
    stroke_shield.evaluate_models()
    
    # Analyze feature importance
    stroke_shield.feature_importance()
    
    # Save models
    stroke_shield.save_models("models/random_forest_model.pkl", "models/logistic_regression_model.pkl")
    
    # Example: Predict stroke risk for a new patient
    new_patient = {
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
    
    result = stroke_shield.predict_stroke_risk(new_patient)
    print("\n===== PREDICTION RESULT =====")
    print(f"Random Forest prediction: {result['random_forest']['prediction']} (Probability: {result['random_forest']['probability']:.2f})")
    print(f"Logistic Regression prediction: {result['logistic_regression']['prediction']} (Probability: {result['logistic_regression']['probability']:.2f})")
    print(f"Ensemble prediction: {result['ensemble']['prediction']} (Probability: {result['ensemble']['probability']:.2f})")
