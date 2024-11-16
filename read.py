import csv

'''
0-Normal
1-Warning
2-Unknown
3-Minor
'''
def read_rows_readable_with_headers(file_name='history.csv',linenumber = 10):
    headers = [
        'event_id', 'status', 'severity', 'category', 'source', 'node', 'creation_date', 'node_id',
        'subcategory', 'eti_type', 'eti_value', 'monitor_uuid', 'severity_id', 'information',
        'occurance', 'update_date', 'alert_name', 'ip', 'dns', 'mac_address', 'correlation'
    ]
    
    try:
        with open(file_name, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row from the CSV
            
            for i, row in enumerate(csv_reader, start=1):
                if i > linenumber:  # Stop after reading 10 rows
                    break
                
                print(f"Row {i}:")
                for header, value in zip(headers, row):
                    print(f"  {header:<15}: {value}")
                print("-" * 50)
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

read_rows_readable_with_headers()


def extract_unique_severity(input_file='history.csv', limit=100):
    severities = []
    informations = []
    
    try:
        with open(input_file, mode='r', encoding='utf-8') as infile:
            csv_reader = csv.reader(infile)
            headers = next(csv_reader)  # Read and skip the header row
            
            if 'severity' not in headers:
                raise ValueError("The column 'severity' does not exist in the CSV file.")
            
            severity_index = headers.index('status')  # Burayla unique alırsın
            information_index = headers.index('information')

            # Extract the 'severity' column, limiting to 'limit' unique values
            for i, row in enumerate(csv_reader):
                if len(row) > severity_index:  # Ensure the row has the expected column
                    severity = row[severity_index]
                    information = row[information_index]
                    
                    #if  severity == "Unknown":
                    severities.add(severity)
                    informations.append(information)

                if len(severities) >= limit:
                    break
            
            return severities,informations
    
    except FileNotFoundError:
        print(f"The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

'''
severities,informations = extract_unique_severity(limit=20)
if severities is not None:
    print("Extracted Unique Severities (First 20):")
    for index, (severity, information) in enumerate(zip(severities, informations)):
        print(f"Severity: {severity}")
        print(f"Information: {information}")
'''

def extract_unknown_severity(input_file='history.csv'):
    unknown_rows = []
    
    try:
        with open(input_file, mode='r', encoding='utf-8') as infile:
            csv_reader = csv.reader(infile)
            headers = next(csv_reader)  # Read and skip the header row
            
            if 'severity' not in headers:
                raise ValueError("The column 'severity' does not exist in the CSV file.")
            
            severity_index = headers.index('severity')  # Get the index of the 'severity' column
            
            # Iterate over the rows to find severity = 'unknown'
            for row in csv_reader:
                if len(row) > severity_index:  # Ensure the row has the expected column
                    severity = row[severity_index]
                    if severity == "Unknown":
                        unknown_rows.append(row)  # Add the entire row if severity is "unknown"
                        break  # Stop after the first occurrence
            
            return unknown_rows
    
    except FileNotFoundError:
        print(f"The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
