import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime

def prepare_predictive_features(df):
    df = df.copy()
    
    # Convert creation_date to datetime and sort
    df['creation_date'] = pd.to_datetime(df['creation_date'])
    df = df.sort_values(['node_id', 'creation_date']).reset_index(drop=True)
    
    # Basic features
    df['time_since_last_event'] = df.groupby('node_id')['creation_date'].diff().dt.total_seconds()
    
    # Create severity state features
    severity_dummies = pd.get_dummies(df['severity'], prefix='severity')
    df = pd.concat([df, severity_dummies], axis=1)
    
    # Rolling features per node
    for node_id, group in df.groupby('node_id'):
        idx = group.index
        
        # Count of each severity in last 3 events
        for col in severity_dummies.columns:
            rolling = severity_dummies.loc[idx, col].rolling(window=3, min_periods=1).sum()
            df.loc[idx, f'{col}_last_3'] = rolling
        
        # Severity changes in last 3 events
        severity_changes = (group['severity'] != group['severity'].shift(1)).astype(int)
        df.loc[idx, 'severity_changes_last_3'] = severity_changes.rolling(window=3, min_periods=1).sum()
        
        # Time-based features
        df.loc[idx, 'avg_time_between_events'] = group['time_since_last_event'].rolling(window=3, min_periods=1).mean()
        df.loc[idx, 'max_time_between_events'] = group['time_since_last_event'].rolling(window=3, min_periods=1).max()
    
    '''
    # Target variable: will become PROBLEM in next 1-2 events
    df['next_status'] = df.groupby('node_id')['status'].shift(-1)
    df['next_next_status'] = df.groupby('node_id')['status'].shift(-2)
    df['will_be_problem'] = ((df['next_status'] == 'PROBLEM') | (df['next_next_status'] == 'PROBLEM')).astype(int)
    '''
    
    df['is_problem'] = (df['status'] == 'PROBLEM').astype(int)

    df['will_be_problem'] = (
        df.groupby('node_id')['is_problem']
        .transform(lambda x: x.shift(-1).rolling(window=5, min_periods=1).sum() > 0)
        .astype(int)
    )

    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

def train_problem_predictor(df):
    # Prepare features
    feature_columns = [col for col in df.columns if col.endswith('_last_3') 
                      or col in ['severity_changes_last_3', 'avg_time_between_events', 
                                'max_time_between_events', 'severity_id', 'occurance']]
    
    X = df[feature_columns]
    y = df['will_be_problem']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    return model, feature_columns

def predict_problems(model, new_data, feature_columns):
    return model.predict_proba(new_data[feature_columns])[:, 1]

def main():
    print("Reading data...")
    df = pd.read_csv('preprocessed33.csv')
    
    print("Preparing features...")
    df_prepared = prepare_predictive_features(df)
    
    print("Training model...")
    model, feature_columns = train_problem_predictor(df_prepared)
    
    # Get predictions for last 10 events
    print("\nSample Predictions:")
    recent_events = df_prepared.tail(10000)
    probabilities = predict_problems(model, recent_events, feature_columns)
    
    count = 0
    for i, (prob, status, severity) in enumerate(zip(
        probabilities, 
        recent_events['status'],
        recent_events['severity']
    )):
        if prob > 0.5: 
            count +=1
            '''
            print(f"\nEvent {i+1}:")
            print(f"  Current Status: {status}")
            print(f"  Current Severity: {severity}")
            print(f"  Risk Score: {prob:.3f}")
            '''
    print(count)        
    return model, feature_columns, df_prepared

def save_model_components(model, feature_columns, file_prefix='problem_predictor'):
    """
    Save the model and associated components
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
    Load the saved model and features
    """
    # Load the model
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    # Load the feature columns
    with open(features_filename, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, feature_columns

# Example usage of saved model
def predict_with_saved_model(new_data, model_filename, features_filename):
    """
    Make predictions using the saved model
    """
    # Load the model and features
    model, feature_columns = load_model_components(model_filename, features_filename)
    
    # Prepare the data
    prepared_data = prepare_predictive_features(pd.DataFrame(new_data))
    
    # Make predictions
    probabilities = predict_problems(model, prepared_data, feature_columns)
    
    return probabilities

if __name__ == "__main__":
    # First run the main training
    model, feature_columns, df_prepared = main()
    
    # Save the model and features
    model_filename, features_filename = save_model_components(model, feature_columns)
    