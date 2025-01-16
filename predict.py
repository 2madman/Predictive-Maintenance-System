import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime

def prepare_predictive_features(df):
    """
    Prepares features for the predictive model by creating time-based and severity-based
    features from the raw event data.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing event data with columns:
            - node_id: Identifier for the node
            - creation_date: Timestamp of the event
            - severity: Severity level of the event
            - status: Current status (e.g., 'PROBLEM', 'OK')
    
    Returns:
        pd.DataFrame: Processed DataFrame with additional predictive features
    """
    df = df.copy()
    
    # Convert creation_date to datetime and ensure events are ordered chronologically
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    df = df.sort_values(['node_id', 'creation_date']).reset_index(drop=True)
    
    # Calculate time difference between consecutive events for each node
    df['time_since_last_event'] = df.groupby('node_id')['creation_date'].diff().dt.total_seconds()
    
    # Create one-hot encoded columns for each severity level
    severity_dummies = pd.get_dummies(df['severity'], prefix='severity')
    df = pd.concat([df, severity_dummies], axis=1)
    
    # Generate node-specific rolling window features
    for node_id, group in df.groupby('node_id'):
        idx = group.index
        
        # Calculate rolling counts of each severity level in last 3 events
        for col in severity_dummies.columns:
            rolling = severity_dummies.loc[idx, col].rolling(window=3, min_periods=1).sum()
            df.loc[idx, f'{col}_last_3'] = rolling
        
        # Track how many times severity changed in last 3 events
        severity_changes = (group['severity'] != group['severity'].shift(1)).astype(int)
        df.loc[idx, 'severity_changes_last_3'] = severity_changes.rolling(window=3, min_periods=1).sum()
        
        # Calculate time-based statistics over last 3 events
        df.loc[idx, 'avg_time_between_events'] = group['time_since_last_event'].rolling(window=3, min_periods=1).mean()
        df.loc[idx, 'max_time_between_events'] = group['time_since_last_event'].rolling(window=3, min_periods=1).max()
    
    # Create binary indicator for current problems
    df['is_problem'] = (df['status'] == 'PROBLEM').astype(int)

    # Create target variable: will there be a problem in the next 5 events?
    df['will_be_problem'] = (
        df.groupby('node_id')['is_problem']
        .transform(lambda x: x.shift(-1).rolling(window=5, min_periods=1).sum() > 0)
        .astype(int)
    )

    # Handle missing values in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

def train_problem_predictor(df):
    """
    Trains a Random Forest model to predict future problems based on current and historical event data.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with predictive features
    
    Returns:
        tuple: (trained RandomForestClassifier, list of feature column names)
    """
    # Select relevant feature columns
    feature_columns = [col for col in df.columns if col.endswith('_last_3') 
                      or col in ['severity_changes_last_3', 'avg_time_between_events', 
                                'max_time_between_events', 'severity_id', 'occurance']]
    
    X = df[feature_columns]
    y = df['will_be_problem']
    
    # Create stratified train-test split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train Random Forest with balanced class weights to handle imbalanced data
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Print model evaluation metrics
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Display feature importance ranking
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    return model, feature_columns

def predict_problems(model, new_data, feature_columns):
    """
    Generate probability predictions for new events.
    
    Args:
        model: Trained RandomForestClassifier
        new_data (pd.DataFrame): New event data to predict on
        feature_columns (list): List of feature columns used by the model
    
    Returns:
        np.array: Probability of problem occurring for each event
    """
    return model.predict_proba(new_data[feature_columns])[:, 1]

def main():
    """
    Main execution function that orchestrates the model training and evaluation process.
    """
    print("Reading data...")
    df = pd.read_csv('preprocessed33.csv')
    
    print("Preparing features...")
    df_prepared = prepare_predictive_features(df)
    
    print("Training model...")
    model, feature_columns = train_problem_predictor(df_prepared)
    
    # Evaluate predictions on recent events
    print("\nSample Predictions:")
    recent_events = df_prepared.tail(10000)
    probabilities = predict_problems(model, recent_events, feature_columns)
    
    # Count high-risk events (probability > 0.5)
    count = 0
    for i, (prob, status, severity) in enumerate(zip(
        probabilities, 
        recent_events['status'],
        recent_events['severity']
    )):
        if prob > 0.5: 
            count += 1
    print(f"Number of high-risk events: {count}")        
    return model, feature_columns, df_prepared

def save_model_components(model, feature_columns, file_prefix='problem_predictor'):
    """
    Serializes and saves the trained model and its associated components.
    
    Args:
        model: Trained RandomForestClassifier
        feature_columns (list): List of feature columns used by the model
        file_prefix (str): Prefix for the saved files
    
    Returns:
        tuple: (model filename, features filename)
    """
    # Save the model
    model_filename = f'{file_prefix}_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the feature columns
    features_filename = f'{file_prefix}_features.pkl'
    with open(features_filename, 'wb') as f:
        pickle.dump(feature_columns, f)
    
    return model_filename, features_filename

def load_model_components(model_filename, features_filename):
    """
    Loads a previously saved model and its components.
    
    Args:
        model_filename (str): Path to the saved model file
        features_filename (str): Path to the saved features file
    
    Returns:
        tuple: (loaded model, feature columns list)
    """
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    with open(features_filename, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, feature_columns

def predict_with_saved_model(new_data, model_filename, features_filename):
    """
    Makes predictions using a previously saved model.
    
    Args:
        new_data (dict or pd.DataFrame): New event data to predict on
        model_filename (str): Path to the saved model file
        features_filename (str): Path to the saved features file
    
    Returns:
        np.array: Probability predictions for each event
    """
    model, feature_columns = load_model_components(model_filename, features_filename)
    prepared_data = prepare_predictive_features(pd.DataFrame(new_data))
    return predict_problems(model, prepared_data, feature_columns)

if __name__ == "__main__":
    # Train and save the model
    model, feature_columns, df_prepared = main()
    model_filename, features_filename = save_model_components(model, feature_columns)