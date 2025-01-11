import os
import csv
from tabulate import tabulate

# Specify the working directory and file path
working_dir = r"C:\Users\admin\Desktop\article-analyzer"  # Replace with your working directory
file_path = os.path.join(working_dir, "test.nbib")

# Function to parse .nbib file and extract DOI, Title, Authors, and Abstract
def parse_nbib(file_path):
    """
    Parse a .nbib file and extract relevant information.
    
    Args:
        file_path (str): Path to the .nbib file
        
    Returns:
        list: List of dictionaries containing extracted records
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    records = []
    record = {}
    authors = []
    current_field = None  # Track which field is currently being processed

    for line in lines:
        if line.startswith('TI  - '):
            record['Title'] = line.replace('TI  - ', '').strip()
            current_field = 'Title'
        elif line.startswith('AB  - '):
            record['Abstract'] = line.replace('AB  - ', '').strip()
            current_field = 'Abstract'
        elif line.startswith('AU  - '):
            authors.append(line.replace('AU  - ', '').strip())
            current_field = None
        elif line.startswith('LID - '):
            if '[doi]' in line:
                doi_part = line.replace('LID - ', '').strip()
                record['DOI'] = doi_part.replace(' [doi]', '').strip()
            current_field = None
        elif line.startswith('PMID- '):
            if record:  # Save the previous record before starting a new one
                record['Authors'] = '; '.join(authors)
                records.append(record)
                record = {}
                authors = []
            current_field = None
        elif line.startswith('      ') and current_field in ['Abstract', 'Title']:  # Handle continuation lines for Abstract and Title
            record[current_field] += ' ' + line.strip()

    # Add the last record
    if record:
        record['Authors'] = '; '.join(authors)
        records.append(record)

    return records

# Save the extracted data to a CSV file
def save_to_csv(records, output_path):
    """
    Save extracted records to a CSV file.
    
    Args:
        records (list): List of dictionaries containing the records
        output_path (str): Path where the CSV file will be saved
    """
    headers = ['DOI', 'Title', 'Authors', 'Abstract']
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

# Main execution
if __name__ == "__main__":
    output_csv = os.path.join(working_dir, "extracted_data.csv")

    # Parse the .nbib file
    parsed_records = parse_nbib(file_path)

    # Prepare table data for preview
    headers = ['DOI', 'Title', 'Authors', 'Abstract']
    preview_data = []
    for record in parsed_records[:3]:  # Display only first 3 records
        # Truncate long fields for better display
        truncated_record = {
            'DOI': record.get('DOI', '')[:50],
            'Title': record.get('Title', '')[:50] + '...' if len(record.get('Title', '')) > 50 else record.get('Title', ''),
            'Authors': record.get('Authors', '')[:50] + '...' if len(record.get('Authors', '')) > 50 else record.get('Authors', ''),
            'Abstract': record.get('Abstract', '')[:100] + '...' if len(record.get('Abstract', '')) > 100 else record.get('Abstract', '')
        }
        preview_data.append([truncated_record['DOI'], truncated_record['Title'], 
                           truncated_record['Authors'], truncated_record['Abstract']])

    # Print preview table
    print("\nPreview of extracted data:")
    print(tabulate(preview_data, headers=headers, tablefmt='grid'))
    print(f"\nTotal records extracted: {len(parsed_records)}")

    # Save the extracted data to a CSV file
    save_to_csv(parsed_records, output_csv)

    print(f"\nData extraction complete. Output saved to {output_csv}")
    