import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score
)
import joblib

class EventStatusPredictor:
    def __init__(self):
        # Initialize key components
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        
    def load_data(self, filepath):
        """
        Load and prepare data from CSV
        
        Args:
            filepath (str): Path to CSV file
        
        Returns:
            tuple: Features (X) and target variable (y)
        """
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Print basic information about the dataset
        print("Dataset Overview:")
        print(f"Total samples: {len(df)}")
        print("\nColumn Types:")
        print(df.dtypes)
        
        # Print unique status values
        print("\nUnique Status Values:")
        print(df['status'].value_counts())
        
        # Separate features and target
        y = df['status']
        X = df.drop('status', axis=1)
        
        return X, y
    
    def preprocess_data(self, X):
        """
        Create preprocessing steps for features
        
        Args:
            X (pd.DataFrame): Input features
        
        Returns:
            ColumnTransformer: Preprocessor for features
        """
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return preprocessor
    
    def create_pipeline(self, preprocessor):
        """
        Create machine learning pipeline
        
        Args:
            preprocessor (ColumnTransformer): Feature preprocessor
        
        Returns:
            Pipeline: Complete ML pipeline
        """
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            ))
        ])
        
        return pipeline
    
    def train(self, filepath):
        """
        Train the machine learning model
        
        Args:
            filepath (str): Path to training CSV
        """
        # Load data
        X, y = self.load_data(filepath)
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create preprocessor
        self.preprocessor = self.preprocess_data(X)
        
        # Create pipeline
        self.model = self.create_pipeline(self.preprocessor)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5)
        print("\nCross-Validation Scores:")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluation
        print("\nModel Performance:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self
    
    def predict(self, input_data):
        """
        Make predictions on new data
        
        Args:
            input_data (pd.DataFrame): New data for prediction
        
        Returns:
            array: Predicted statuses
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Predictions
        predictions_encoded = self.model.predict(input_data)
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def save_model(self, filepath='event_status_model.pkl'):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save model
        """
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='event_status_model.pkl'):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to load model from
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.label_encoder = saved_data['label_encoder']
        print("Model loaded successfully")
        return self

def main():
    # File paths
    training_csv = 'history2.csv'
    
    # Initialize predictor
    predictor = EventStatusPredictor()
    
    # Train the model
    predictor.train(training_csv)
    
    # Save the model
    predictor.save_model('event_status_model.pkl')
    
    # Optional: Demonstrate prediction
    # Load the first few rows of the training data for prediction
    sample_data = pd.read_csv(training_csv).drop('status', axis=1).head(220)
    predictions = predictor.predict(sample_data)
    print("\nSample Predictions:", predictions)

if __name__ == "__main__":
    main()
