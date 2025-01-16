import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Union
import sys
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pattern_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PatternAnalyzer:
    def __init__(self, csv_file: str):
        """
        Initialize the PatternAnalyzer with the CSV file path.
        
        Args:
            csv_file (str): Path to the CSV file
        """
        self.csv_file = csv_file
        self.columns = ['event_id', 'status', 'severity', 'category', 'source',
                       'creation_date', 'node_id', 'subcategory', 'eti_type',
                       'eti_value', 'monitor_uuid', 'severity_id', 'occurance']
        self.df = None
        self.patterns = {}

    def load_data(self) -> bool:
        """
        Load the CSV file into a pandas DataFrame.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.df = pd.read_csv(self.csv_file)
            
            # Validate columns
            missing_cols = set(self.columns) - set(self.df.columns)
            if missing_cols:
                logger.error(f"Missing columns in CSV: {missing_cols}")
                return False
                
            logger.info(f"Successfully loaded data with {len(self.df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return False

    def analyze_severity_patterns(self) -> Dict[str, int]:
        """
        Analyze severity patterns across all data using a sliding window of 3 records.
        
        Returns:
            Dict[str, int]: Dictionary containing count of each pattern occurrence
        """
        total_rows = len(self.df)
        warning_count = 0
        minor_count = 0
        warning_minor_unknown_count = 0
        
        # Analyze patterns using sliding window of 3
        for i in range(total_rows - 2):  # -2 to ensure we always have 3 records to check
            window = self.df.iloc[i:i+3]
            
            # Count patterns where all 3 consecutive records are Warning
            if all(window['severity'] == 'Warning'):
                warning_count += 1
                
            # Count patterns where all 3 consecutive records are Minor
            if all(window['severity'] == 'Minor'):
                minor_count += 1
                
            # Count patterns where all 3 consecutive records are Warning, Minor, or Unknown
            if all(window['severity'].isin(['Warning', 'Minor', 'Unknown'])):
                warning_minor_unknown_count += 1
        
        severity_patterns = {
            'consecutive_3_warning_count': warning_count,
            'consecutive_3_minor_count': minor_count,
            'consecutive_3_warning_minor_unknown_count': warning_minor_unknown_count,
        }
        
        return severity_patterns

    def analyze_category_patterns(self) -> Dict[str, int]:
        """
        Analyze category and ETI patterns across all data using sliding window of 5 records.
        """
        total_rows = len(self.df)
        same_category_count = 0
        same_eti_type_count = 0
        
        # Analyze patterns using sliding window of 5
        for i in range(total_rows - 4):  # -4 to ensure we always have 5 records to check
            window = self.df.iloc[i:i+5]
            
            # Count patterns where all 5 consecutive records have same category
            if len(window['category'].unique()) == 1:
                same_category_count += 1
                
            # Count patterns where all 5 consecutive records have same eti_type
            if len(window['eti_type'].unique()) == 1:
                same_eti_type_count += 1
        
        category_patterns = {
            'consecutive_5_same_category_count': same_category_count,
            'consecutive_5_same_eti_type_count': same_eti_type_count
        }
        
        return category_patterns

    def analyze_event_repetition(self) -> Dict[str, Union[int, List[str]]]:
        """
        Analyze event repetition patterns using sliding window of 10 records.
        """
        total_rows = len(self.df)
        repeated_events_count = 0
        
        # Analyze patterns using sliding window of 10
        for i in range(total_rows - 9):  # -9 to ensure we always have 10 records to check
            window = self.df.iloc[i:i+10]
            event_counts = window['event_id'].value_counts()
            
            # If any event appears more than once in the window of 10
            if any(count > 1 for count in event_counts):
                repeated_events_count += 1
        
        repetition_patterns = {
            'windows_with_repeated_events_in_10': repeated_events_count
        }
        
        return repetition_patterns

    def analyze_eti_patterns(self) -> Dict[str, int]:
        """
        Analyze ETI patterns across all data using a sliding window of 5 records.
        
        Returns:
            Dict[str, int]: Dictionary containing count of ETI pattern occurrences
        """
        total_rows = len(self.df)
        same_eti_type_count = 0
        same_eti_value_count = 0
        
        # Analyze patterns using sliding window of 5
        for i in range(total_rows - 4):  # -4 to ensure we always have 5 records to check
            window = self.df.iloc[i:i+5]
            
            # Check if all 5 records have the same ETI type
            if len(window['eti_type'].unique()) == 1:
                same_eti_type_count += 1
                
            # Check if all 5 records have the same ETI value
            if len(window['eti_value'].unique()) == 1:
                same_eti_value_count += 1
        
        # Additional statistics about ETI patterns
        eti_patterns = {
            'consecutive_5_same_eti_type_count': same_eti_type_count,
            'consecutive_5_same_eti_value_count': same_eti_value_count,
            'total_windows_checked': total_rows - 4  # Number of windows analyzed
        }
        
        return eti_patterns

    def analyze_status_patterns(self) -> Dict[str, Union[bool, Dict]]:
        """
        Analyze patterns related to status in the last 5 entries.
        
        Returns:
            Dict[str, Union[bool, Dict]]: Dictionary containing status pattern results
        """
        last_5 = self.df.head(5)
        
        status_patterns = {
            'same_status_last_5': len(last_5['status'].unique()) == 1,
            'status_value_if_same': last_5['status'].iloc[0] if len(last_5['status'].unique()) == 1 else None,
            'status_frequency': dict(last_5['status'].value_counts()),
            'status_progression': list(last_5['status'])
        }
        
        return status_patterns

    def run_analysis(self) -> Dict:
        """
        Run all pattern analyses and compile results.
        
        Returns:
            Dict: Complete analysis results with pattern counts
        """
        if not self.load_data():
            return {}

        self.patterns = {
            'severity_patterns': self.analyze_severity_patterns(),
            'category_patterns': self.analyze_category_patterns(),
            'eti_patterns': self.analyze_eti_patterns(),
            'repetition_patterns': self.analyze_event_repetition()
        }

        # Add total number of records analyzed
        self.patterns['metadata'] = {
            'total_records': len(self.df),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return self.patterns
def print_analysis_results(patterns: Dict):
    """
    Print analysis results in a formatted way.
    
    Args:
        patterns (Dict): Dictionary containing analysis results
    """
    print("\n=== Pattern Analysis Results ===\n")
    
    # Severity Patterns
    print("SEVERITY PATTERNS:")
    print("-" * 50)
    for key, value in patterns['severity_patterns'].items():
        print(f"{key}: {value}")
    
    # Category Patterns
    print("\nCATEGORY PATTERNS:")
    print("-" * 50)
    for key, value in patterns['category_patterns'].items():
        print(f"{key}: {value}")
    
    # ETI Patterns
    print("\nETI PATTERNS:")
    print("-" * 50)
    for key, value in patterns['eti_patterns'].items():
        print(f"{key}: {value}")

    
    

def main():
    """
    Main function to run the pattern analysis.
    """
    try:
        # Initialize and run analysis
        analyzer = PatternAnalyzer('preprocessed33.csv')
        patterns = analyzer.run_analysis()
        
        if patterns:
            # Print results
            print_analysis_results(patterns)
            
        else:
            logger.error("Analysis failed to produce results")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()