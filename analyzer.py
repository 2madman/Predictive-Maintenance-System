import pandas as pd
import numpy as np
from datetime import timedelta

class LogPatternAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the log analyzer with the CSV file
        
        :param csv_path: Path to the history.csv file
        """
        self.df = pd.read_csv(csv_path, parse_dates=['creation_date'])
        self.df.sort_values('creation_date', inplace=True)
        
    def detect_patterns(self):
        """
        Detect various log patterns in the event data
        
        :return: Dictionary of detected patterns
        """
        patterns = {
            'warning_minor_sequences': self._detect_warning_minor_sequences(),
            'repeated_events': self._detect_repeated_events(),
            'event_frequency_changes': self._detect_frequency_changes(),
            'unknown_after_warning': self._detect_unknown_after_warning(),
            'sudden_severity_increase': self._detect_sudden_severity_increase(),
            'problem_root_causes': self._detect_problem_root_causes()
        }
        
        return patterns
    
    def _detect_warning_minor_sequences(self, consecutive_threshold=3):
        """
        Detect sequences of warning, minor, or unknown severity events
        
        :param consecutive_threshold: Number of consecutive events to trigger detection
        :return: List of detected sequences
        """
        warning_levels = ['Warning', 'Minor', 'Unknown']
        sequences = []
        
        # Group consecutive warning/minor events
        current_sequence = []
        for _, row in self.df.iterrows():
            if row['severity'] in warning_levels:
                current_sequence.append(row)
            else:
                if len(current_sequence) >= consecutive_threshold:
                    sequences.append(current_sequence)
                current_sequence = []
        
        # Check final sequence
        if len(current_sequence) >= consecutive_threshold:
            sequences.append(current_sequence)
        
        return sequences
    
    def _detect_repeated_events(self, time_window=timedelta(minutes=30), repetition_threshold=3):
        """
        Detect repeated events within a short time frame
        
        :param time_window: Time window to consider for event repetition
        :param repetition_threshold: Number of times an event must repeat to be flagged
        :return: List of repeated events
        """
        repeated_events = []
        
        # Group events by event_id and check repetitions within time window
        grouped = self.df.groupby('event_id')
        for event_id, event_group in grouped:
            if len(event_group) >= repetition_threshold:
                # Check if events are within the specified time window
                time_diffs = event_group['creation_date'].diff()
                if all(diff <= time_window for diff in time_diffs[1:]):
                    repeated_events.append(event_group)
        
        return repeated_events
    
    def _detect_frequency_changes(self, window_size=10):
        """
        Detect decrease in normal events
        
        :param window_size: Number of events to use for moving window analysis
        :return: Detected frequency changes
        """
        normal_event_counts = []
        frequency_changes = []
        
        # Slide a window and track normal event frequency
        for i in range(len(self.df) - window_size):
            window = self.df.iloc[i:i+window_size]
            normal_count = (window['severity'] == 'Normal').sum()
            normal_event_counts.append(normal_count)
        
        # Detect significant drops in normal events
        normal_event_series = pd.Series(normal_event_counts)
        significant_drops = normal_event_series[normal_event_series < normal_event_series.mean() * 0.5]
        
        if not significant_drops.empty:
            frequency_changes = [
                {
                    'start_index': idx, 
                    'normal_event_count': significant_drops[idx],
                    'window': self.df.iloc[idx:idx+window_size]
                } 
                for idx in significant_drops.index
            ]
        
        return frequency_changes
    
    def _detect_unknown_after_warning(self):
        """
        Detect unknown events following warning or minor events
        
        :return: List of sequences where unknown events follow warnings
        """
        warning_levels = ['Warning', 'Minor']
        unknown_sequences = []
        
        for i in range(len(self.df) - 1):
            if (self.df.iloc[i]['severity'] in warning_levels and 
                self.df.iloc[i+1]['severity'] == 'Unknown'):
                unknown_sequences.append([
                    self.df.iloc[i],
                    self.df.iloc[i+1]
                ])
        
        return unknown_sequences
    
    def _detect_sudden_severity_increase(self, increase_threshold=2):
        """
        Detect sudden increases in warning or minor events
        
        :param increase_threshold: Multiplier to determine significant increase
        :return: Detected severity increases
        """
        severity_mapping = {
            'Normal': 1,
            'Warning': 2,
            'Minor': 3,
            'Unknown': 4
        }
        
        severity_increases = []
        
        # Compare severity levels between consecutive windows
        for i in range(len(self.df) - 10):
            window1 = self.df.iloc[i:i+5]
            window2 = self.df.iloc[i+5:i+10]
            
            avg_severity1 = window1['severity'].map(severity_mapping).mean()
            avg_severity2 = window2['severity'].map(severity_mapping).mean()
            
            if avg_severity2 > avg_severity1 * increase_threshold:
                severity_increases.append({
                    'first_window': window1,
                    'second_window': window2,
                    'severity_increase': avg_severity2 / avg_severity1
                })
        
        return severity_increases
    
    def _detect_problem_root_causes(self):
        """
        Detect root causes in PROBLEM status events
        
        :return: List of events with potential root causes
        """
        problem_events = self.df[self.df['status'] == 'PROBLEM']
        root_cause_events = []
        
        for _, event in problem_events.iterrows():
            if 'root' in str(event['information']).lower():
                root_cause_events.append(event)
        
        return root_cause_events

def main():
    analyzer = LogPatternAnalyzer('history.csv')
    patterns = analyzer.detect_patterns()
    
    # Print detected patterns
    for pattern_name, pattern_data in patterns.items():
        print(f"\n{pattern_name.replace('_', ' ').title()}:")
        if pattern_data:
            print(f"  Detected {len(pattern_data)} instances")
        else:
            print("  No instances detected")

if __name__ == "__main__":
    main()