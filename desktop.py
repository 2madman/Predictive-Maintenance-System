import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from EventStatusPredictor import EventStatusPredictor
import time
from datetime import datetime
from collections import deque

class MonitoringPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Event Status Predictor & Pattern Monitor")
        self.root.geometry("1500x600")
        
        # Initialize predictor
        self.predictor = EventStatusPredictor()
        
        # Initialize rolling window for pattern detection
        self.recent_logs = deque(maxlen=3)  # Keep last 3 records for pattern detection
        
        # Load model
        try:
            self.predictor.load_model('event_status_model.pkl')
            print("Model loaded successfully!")
        except:
            print("No model found. Please ensure 'event_status_model.pkl' exists.")
        
        self.create_gui()
        self.load_and_predict()
        
    def create_gui(self):
        # Main container
        container = ttk.Frame(self.root, padding="10")
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left side - Event Prediction
        left_frame = ttk.Frame(container)
        left_frame.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Label(left_frame, text="Event Monitoring:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        # Create text widget with scrollbar for predictions
        self.result_text = tk.Text(left_frame, height=30, width=100)
        left_scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=left_scrollbar.set)
        
        self.result_text.grid(row=1, column=0, pady=5, padx=(0, 5))
        left_scrollbar.grid(row=1, column=1, sticky='ns')
        
        # Right side - Pattern Detection
        right_frame = ttk.Frame(container)
        right_frame.grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(right_frame, text="Pattern Detection:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        # Create text widget with scrollbar for patterns
        self.pattern_text = tk.Text(right_frame, height=30, width=70)
        right_scrollbar = ttk.Scrollbar(right_frame, orient='vertical', command=self.pattern_text.yview)
        self.pattern_text.configure(yscrollcommand=right_scrollbar.set)
        
        self.pattern_text.grid(row=1, column=0, pady=5, padx=(0, 5))
        right_scrollbar.grid(row=1, column=1, sticky='ns')
        
        # Configure tags for highlighting
        self.result_text.tag_configure('prediction', foreground='blue', font=('Arial', 10, 'bold'))
        self.result_text.tag_configure('problem', foreground='red', font=('Arial', 10, 'bold'))
        self.result_text.tag_configure('ok', foreground='green', font=('Arial', 10, 'bold'))
        self.result_text.tag_configure('timestamp', foreground='gray')
        
        self.pattern_text.tag_configure('warning', foreground='red', font=('Arial', 10, 'bold'))
        self.pattern_text.tag_configure('normal', foreground='green', font=('Arial', 10))
        self.pattern_text.tag_configure('timestamp', foreground='gray')
        
    def check_pattern(self, new_record):
        """
        Check for patterns as each new record arrives
        Returns: (pattern_found: bool, pattern_type: str or None)
        """
        # Add new record to rolling window
        self.recent_logs.append(new_record)
        
        # Only check pattern if we have at least 2 records
        if len(self.recent_logs) >= 2:
            # Check for Unknown after Warning/Minor pattern
            last_record = self.recent_logs[-1]
            previous_record = self.recent_logs[-2]
            
            if (last_record['severity'] == 'Unknown' and 
                previous_record['severity'] in {'Warning', 'Minor'}):
                return True, 'unknown_after_alert'
            
            # Original pattern check (if we have 3 records)
            if len(self.recent_logs) == 3:
                monitored_severities = {'Warning', 'Minor', 'Unknown'}
                if all(log['severity'] in monitored_severities for log in self.recent_logs):
                    return True, 'three_consecutive'
                    
        return False, None

    def show_unknown_after_alert_popup(self, current_record, previous_record):
        """
        Show a popup message when Unknown severity follows Warning/Minor
        """
        message = f"Alert: Unknown Status After {previous_record['severity']}!\n\n"
        message += f"Previous Event: {previous_record['severity']}\n"
        message += f"Current Event: Unknown\n\n"
        message += f"Information: {current_record.get('information', 'No additional information available')}"
        
        messagebox.showwarning("Pattern Alert", message, icon='warning')

    def update_pattern_display(self, new_record):
        """
        Update pattern detection display with new record
        """
        # Check for pattern with new record
        pattern_detected, pattern_type = self.check_pattern(new_record)
        
        if pattern_detected:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.pattern_text.insert(tk.END, f"\n[{timestamp}] ", 'timestamp')
            
            if pattern_type == 'unknown_after_alert':
                self.pattern_text.insert(tk.END, "Unknown Status After Alert Pattern Detected!\n", 'warning')
                self.pattern_text.insert(tk.END, f"Previous: {self.recent_logs[-2]['severity']}\n")
                self.pattern_text.insert(tk.END, f"Current: Unknown\n")
                # Show popup for this specific pattern
                self.show_unknown_after_alert_popup(self.recent_logs[-1], self.recent_logs[-2])
            else:  # three_consecutive pattern
                self.pattern_text.insert(tk.END, "Three Consecutive Alert Pattern Detected!\n", 'warning')
                self.pattern_text.insert(tk.END, "Last 3 logs showing concerning pattern:\n")
                
                for log in self.recent_logs:
                    self.pattern_text.insert(tk.END, f"- Severity: {log['severity']}\n")
                    self.pattern_text.insert(tk.END, f"  Category: {log['category']}\n")
                    self.pattern_text.insert(tk.END, f"  Time: {log['creation_date']}\n")
            
            self.pattern_text.insert(tk.END, "\n")
            self.pattern_text.see(tk.END)

    def show_problem_popup(self, record):
        """
        Show a popup message when a problem is detected and wait for user acknowledgment
        """
        information = record.get('information', 'No additional information available')
        message = f"Problem Occurred!\n\nInformation: {information}"
        response = messagebox.showwarning("Alert", message, icon='warning')
        # Continue monitoring after user closes the popup
        self.root.after(1, self.predict_next)

    def load_and_predict(self):
        try:
            data = pd.read_csv('history.csv', low_memory=False, nrows=10000)
            if 'status' in data.columns:
                data = data.drop('status', axis=1)
            
            self.current_index = 0
            self.total_records = len(data)
            self.data = data
            self.predict_next()
            
        except Exception as e:
            self.result_text.insert(tk.END, f"Error loading data: {str(e)}\n")
    
    def format_record_details(self, record, is_problem=False):
        details = [
            f"Severity: {record.get('severity', 'N/A')}",
            f"Subcategory: {record.get('subcategory', 'N/A')}"
        ]
        
        # Only include information field if it's a problem
        if is_problem:
            if 'information' in record and not pd.isna(record['information']):
                details.append(f"Information: {record['information']}")
        
        if 'source' in record and not pd.isna(record['source']):
            details.append(f"Source: {record['source']}")
        
        return " | ".join(details)
    
    def predict_next(self):
        if self.current_index < self.total_records:
            # Get current record
            current_record = self.data.iloc[self.current_index]
            record_dict = current_record.to_dict()
            
            # Make prediction
            prediction = self.predictor.predict(current_record.to_frame().T)[0]
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update prediction display
            self.result_text.insert(tk.END, f"[{timestamp}] ", 'timestamp')
            if prediction == "PROBLEM":
                details = self.format_record_details(current_record, is_problem=True)
            else:
                details = self.format_record_details(current_record, is_problem=False)
            self.result_text.insert(tk.END, f"{details}\n")
            self.result_text.insert(tk.END, "Status: ", 'prediction')
            
            if prediction == "PROBLEM":
                self.result_text.insert(tk.END, f"{prediction}\n\n", 'problem')
                self.current_index += 1  # Increment index before showing popup
                # Show popup with record information and wait for user acknowledgment
                self.root.after(1, lambda: self.show_problem_popup(record_dict))
            else:
                self.result_text.insert(tk.END, f"{prediction}\n\n", 'ok')
                # Schedule next prediction immediately for non-problem cases
                self.current_index += 1
                self.root.after(300, self.predict_next)
            
            self.result_text.see(tk.END)
            
            # Update pattern detection with new record
            self.update_pattern_display(record_dict)

def main():
    root = tk.Tk()
    app = MonitoringPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()