import os
import pandas as pd
import logging
from typing import Tuple, Optional

class FileProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def parse_nbib(self, file_path: str) -> Tuple[Optional[str], str]:
        """Parse NBIB file and return results"""
        if not file_path or not os.path.exists(file_path):
            return None, "Invalid file"
            
        try:
            records = []
            record = {}
            authors = []
            current_field = None

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return None, "Empty file"

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
                    if record:  # Save the previous record
                        record['Authors'] = '; '.join(authors)
                        records.append(record)
                        record = {}
                        authors = []
                    current_field = None
                elif line.startswith('      ') and current_field in ['Abstract', 'Title']:
                    record[current_field] += ' ' + line.strip()

            # Add the last record
            if record:
                record['Authors'] = '; '.join(authors)
                records.append(record)

            # Create DataFrame and add index
            df = pd.DataFrame(records)
            df.index.name = 'Index'
            
            # Save to CSV
            output_path = os.path.join(self.data_dir, "extracted_data.csv")
            df.to_csv(output_path)

            # Prepare preview data
            preview = ""
            for i, record in enumerate(records[:3], 0):
                preview += f"\nRecord {i}:\n"
                preview += f"DOI: {record.get('DOI', '')[:50]}\n"
                preview += f"Title: {record.get('Title', '')[:100]}...\n"
                preview += f"Authors: {record.get('Authors', '')[:100]}...\n"
                preview += f"Abstract: {record.get('Abstract', '')[:200]}...\n"
                preview += "-" * 80 + "\n"
            
            preview += f"\nTotal records extracted: {len(records)}"
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing NBIB file: {str(e)}"
            
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file and ensure correct index"""
        try:
            # First try to read the file, keeping the index column
            df = pd.read_csv(file_path, index_col=0)
            
            # Check if Index column exists (might be in data columns)
            if "Index" in df.columns:
                # If Index is in columns, set it as index
                df.set_index("Index", inplace=True)
            
            # Ensure index name is "Index"
            df.index.name = "Index"
            
            # Ensure index is string type
            df.index = df.index.astype(str)
            
            logging.debug(f"Loaded DataFrame from {file_path}")
            logging.debug(f"Columns: {df.columns.tolist()}")
            logging.debug(f"Index name: {df.index.name}")
            logging.debug(f"First few indices: {df.index.tolist()[:5]}")
            
            return df
        except Exception as e:
            logging.error(f"Error loading CSV file: {str(e)}")
            return None
            
    def save_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to CSV file"""
        try:
            output_path = os.path.join(self.data_dir, filename)
            df.to_csv(output_path, index=True)
            return output_path
        except Exception as e:
            logging.error(f"Error saving CSV file: {str(e)}")
            return "" 