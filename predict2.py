import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

class EventStatusPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.warning_threshold = 0.3
        self.feature_columns = None  # Store feature columns for prediction
        
    def prepare_features(self, df, is_training=True):
        """
        Prepare features including temporal and rolling features
        
        Args:
            df (pd.DataFrame): Input dataframe
            is_training (bool): Whether this is training data
        
        Returns:
            tuple: Features (X) and target (y) if training, otherwise just features (X)
        """
        # Convert creation_date to datetime
        df = df.copy()
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        
        # Add temporal features
        df['hour'] = df['creation_date'].dt.hour
        df['minute'] = df['creation_date'].dt.minute
        
        # Sort by creation_date for rolling features
        df = df.sort_values('creation_date')
        
        # Create rolling window features
        df['severity_rolling_mean'] = df.groupby('node_id')['severity_id'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['occurance_rolling_sum'] = df.groupby('node_id')['occurance'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        )
        
        if is_training:
            y = df['status']
            X = df.drop(['status', 'creation_date'], axis=1)
            self.feature_columns = X.columns  # Store feature columns
            return X, y
        else:
            X = df.drop(['creation_date'], axis=1)
            # Ensure columns match training data
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]  # Reorder columns to match training data
            return X

    def load_data(self, filepath):
        """Load and prepare data from CSV"""
        df = pd.read_csv(filepath)
        
        print("Dataset Overview:")
        print(f"Total samples: {len(df)}")
        print("\nColumn Types:")
        print(df.dtypes)
        
        print("\nUnique Status Values:")
        print(df['status'].value_counts())
        
        return self.prepare_features(df, is_training=True)
    
    def preprocess_data(self, X):
        """Create preprocessing steps for features"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return preprocessor
    
    def create_pipeline(self, preprocessor):
        """Create machine learning pipeline"""
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
    
    def predict_with_warning(self, input_data):
        """
        Make predictions with early warning system
        
        Args:
            input_data (pd.DataFrame): New data for prediction
        
        Returns:
            tuple: (predictions, warnings)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Prepare features for prediction
        X = self.prepare_features(input_data, is_training=False)
        
        # Get probability estimates
        probabilities = self.model.predict_proba(X)
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Print available classes for debugging
        print("Available classes:", self.label_encoder.classes_)
        
        # Find the index of the most severe status (assuming it's the last class)
        problem_class_idx = -1  # Use the last class as the "problem" class
        
        warnings = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred != self.label_encoder.classes_[problem_class_idx] and prob[problem_class_idx] >= self.warning_threshold:
                warnings.append({
                    'index': i,
                    'current_status': pred,
                    'problem_probability': prob[problem_class_idx],
                    'warning_message': f"Warning: {prob[problem_class_idx]:.2%} chance of {self.label_encoder.classes_[problem_class_idx]} occurring"
                })
        
        return predictions, warnings
    
    
    def train(self, filepath):
        """Train the machine learning model"""
        X, y = self.load_data(filepath)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.preprocessor = self.preprocess_data(X)
        self.model = self.create_pipeline(self.preprocessor)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5)
        print("\nCross-Validation Scores:")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
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
    
    def save_model(self, filepath='event_status_model.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'warning_threshold': self.warning_threshold,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='event_status_model.pkl'):
        """Load a pre-trained model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.label_encoder = saved_data['label_encoder']
        self.warning_threshold = saved_data.get('warning_threshold', 0.3)
        self.feature_columns = saved_data.get('feature_columns')
        print("Model loaded successfully")
        return self

def main():
    training_csv = 'history2.csv'
    
    predictor = EventStatusPredictor()
    predictor.train(training_csv)
    predictor.save_model('event_status_model.pkl')
    
    # Demonstrate prediction with warning system
    sample_data = pd.read_csv(training_csv)
    predictions, warnings = predictor.predict_with_warning(sample_data)
    
    print("\nSample Predictions:", predictions[:5])
    if warnings:
        print("\nWarnings Detected:")
        for warning in warnings[:200]:
            print(warning['warning_message'])

if __name__ == "__main__":
    main()
