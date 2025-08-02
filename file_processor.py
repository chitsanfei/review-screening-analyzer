import os
import pandas as pd
import logging
import re
from typing import Tuple, Optional

class FileProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize FileProcessor
        
        Args:
            data_dir: Directory path for storing processed data
        """
        self.data_dir = data_dir
        
    def parse_nbib(self, file_path: str) -> Tuple[Optional[str], str]:
        """
        Parse NBIB file and return Excel output path and preview text
        
        Args:
            file_path: Path to the NBIB file to parse
            
        Returns:
            tuple: (output_path, preview_text) where:
                - output_path: Path to the generated Excel file (None if parsing fails)
                - preview_text: Preview of the parsed data or error message
        """
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

            # Process each line in the NBIB file
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

            # Save the last record if exists
            if record:
                record['Authors'] = '; '.join(authors)
                records.append(record)

            # Create DataFrame and save to Excel
            df = pd.DataFrame(records)
            df.index.name = 'Index'
            output_path = os.path.join(self.data_dir, "extracted_data.xlsx")
            df.to_excel(output_path, index=True)
            preview = self._generate_preview(records)
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing NBIB file: {str(e)}"

    def parse_wos_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """
        Parse Web of Science RIS file and return Excel output path and preview text
        
        Args:
            file_path: Path to the WOS RIS file to parse
            
        Returns:
            tuple: (output_path, preview_text) where:
                - output_path: Path to the generated Excel file (None if parsing fails)
                - preview_text: Preview of the parsed data or error message
        """
        if not file_path or not os.path.exists(file_path):
            return None, "Invalid file"
            
        try:
            records = []
            record = {}
            authors = []
            current_field = None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                return None, "Empty file"

            # Split content into individual articles
            articles = content.split("\nER  -")
            
            for article in articles:
                if not article.strip():
                    continue
                    
                record = {}
                authors = []
                
                # Process each line in the article
                lines = article.strip().split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith('TI  - '):
                        record['Title'] = line.replace('TI  - ', '').strip()
                    elif line.startswith('AB  - '):
                        record['Abstract'] = line.replace('AB  - ', '').strip()
                    elif line.startswith('AU  - '):
                        authors.append(line.replace('AU  - ', '').strip())
                    elif line.startswith('DO  - '):
                        record['DOI'] = line.replace('DO  - ', '').strip()
                    elif line.startswith('   '):
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()

                if record:
                    record['Authors'] = '; '.join(authors)
                    records.append(record)

            # Create DataFrame with required columns
            df = pd.DataFrame(records)
            required_columns = ['Title', 'Abstract', 'Authors', 'DOI']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            df.index.name = 'Index'
            output_path = os.path.join(self.data_dir, "extracted_data.xlsx")
            df.to_excel(output_path, index=True)
            preview = self._generate_preview(records)
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing WOS RIS file: {str(e)}"

    def parse_embase_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """
        Parse Embase RIS file and return Excel output path and preview text
        
        Args:
            file_path: Path to the Embase RIS file to parse
            
        Returns:
            tuple: (output_path, preview_text) where:
                - output_path: Path to the generated Excel file (None if parsing fails)
                - preview_text: Preview of the parsed data or error message
        """
        if not file_path or not os.path.exists(file_path):
            return None, "Invalid file"
            
        try:
            records = []
            record = {}
            authors = []
            current_field = None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                return None, "Empty file"

            # Split content into individual articles
            articles = content.split("\n\n")
            
            for article in articles:
                if not article.strip():
                    continue
                    
                record = {}
                authors = []
                
                # Process each line in the article
                lines = article.strip().split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith('T1  - '):  # Title field
                        record['Title'] = line.replace('T1  - ', '').strip()
                    elif line.startswith('N2  - '):  # Abstract field
                        record['Abstract'] = line.replace('N2  - ', '').strip()
                    elif line.startswith('A1  - '):  # Authors field
                        authors.append(line.replace('A1  - ', '').strip())
                    elif line.startswith('DO  - '):  # DOI field
                        record['DOI'] = line.replace('DO  - ', '').strip()
                    elif line.startswith('   '):  # Handle multi-line fields
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()

                if record:
                    record['Authors'] = '; '.join(authors) if authors else ''
                    records.append(record)

            # Create DataFrame with required columns
            df = pd.DataFrame(records)
            required_columns = ['Title', 'Abstract', 'Authors', 'DOI']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            df.index.name = 'Index'
            output_path = os.path.join(self.data_dir, "extracted_data.xlsx")
            df.to_excel(output_path, index=True)
            preview = self._generate_preview(records)
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing Embase RIS file: {str(e)}"

    def parse_scopus_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """
        Parse Scopus RIS file and return Excel output path and preview text
        
        Args:
            file_path: Path to the Scopus RIS file to parse
            
        Returns:
            tuple: (output_path, preview_text) where:
                - output_path: Path to the generated Excel file (None if parsing fails)
                - preview_text: Preview of the parsed data or error message
        """
        if not file_path or not os.path.exists(file_path):
            return None, "Invalid file"
            
        try:
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content:
                return None, "Empty file"
            
            # Use regex to split records by "ER  -" (note the double space)
            articles = re.split(r'\nER\s*-\s*', content)
            
            for article in articles:
                if not article.strip():
                    continue
                record = {}
                authors = []
                lines = article.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('TI  - '):
                        record['Title'] = line.replace('TI  - ', '').strip()
                    elif line.startswith('AB  - '):
                        record['Abstract'] = line.replace('AB  - ', '').strip()
                    elif line.startswith('AU  - '):
                        authors.append(line.replace('AU  - ', '').strip())
                    elif line.startswith('DO  - '):
                        record['DOI'] = line.replace('DO  - ', '').strip()
                    elif line.startswith('   '):
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()
                record['Authors'] = '; '.join(authors)
                records.append(record)
            
            # Create DataFrame with required columns
            df = pd.DataFrame(records)
            required_columns = ['Title', 'Abstract', 'Authors', 'DOI']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''
            df.index.name = 'Index'
            output_path = os.path.join(self.data_dir, "extracted_data.xlsx")
            df.to_excel(output_path, index=True)
            preview = self._generate_preview(records)
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing Scopus RIS file: {str(e)}"

    def _generate_preview(self, records: list) -> str:
        """
        Generate a preview text for the first few parsed records
        
        Args:
            records: List of parsed records
            
        Returns:
            str: Formatted preview text showing sample records
        """
        preview = ""
        for i, record in enumerate(records[:3], 0):
            preview += f"\nRecord {i}:\n"
            preview += f"DOI: {record.get('DOI', '')[:50]}\n"
            preview += f"Title: {record.get('Title', '')[:100]}...\n"
            preview += f"Authors: {record.get('Authors', '')[:100]}...\n"
            preview += f"Abstract: {record.get('Abstract', '')[:200]}...\n"
            preview += "-" * 80 + "\n"
        
        preview += f"\nTotal records extracted: {len(records)}"
        return preview
            
    def load_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load Excel file and ensure the index is set correctly
        
        Args:
            file_path: Path to the Excel file to load
            
        Returns:
            DataFrame or None if loading fails
        """
        try:
            # First try to read with index_col=0
            df = pd.read_excel(file_path, index_col=0)
            
            # If Index is still in columns, it means it wasn't properly set as index
            if "Index" in df.columns:
                df.set_index("Index", inplace=True)
            elif df.index.name != "Index":
                df.index.name = "Index"
            
            # Ensure index is string type and handle any potential NaN values
            df.index = df.index.astype(str)
            df.index = df.index.str.strip()
            
            # Remove any duplicate indices by keeping the first occurrence
            if df.index.duplicated().any():
                logging.warning(f"Found duplicate indices in {file_path}")
                df = df[~df.index.duplicated(keep='first')]
            
            logging.debug(f"Loaded DataFrame from {file_path}")
            logging.debug(f"Shape: {df.shape}")
            logging.debug(f"Columns: {df.columns.tolist()}")
            logging.debug(f"Index name: {df.index.name}")
            logging.debug(f"First few indices: {df.index.tolist()[:5]}")
            
            return df
        except Exception as e:
            logging.error(f"Error loading Excel file: {str(e)}")
            return None
            
    def save_excel(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save a DataFrame to an Excel file
        
        Args:
            df: DataFrame to save
            filename: Target filename
            
        Returns:
            str: Path to the saved file or empty string if saving fails
        """
        try:
            # Ensure we have a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure index is properly named
            if df.index.name != "Index":
                df.index.name = "Index"
            
            # Ensure index is string type
            df.index = df.index.astype(str)
            
            # Remove any duplicate indices
            if df.index.duplicated().any():
                logging.warning(f"Found duplicate indices when saving {filename}")
                df = df[~df.index.duplicated(keep='first')]
            
            output_path = os.path.join(self.data_dir, filename)
            
            # Save with index
            df.to_excel(output_path, index=True)
            
            logging.debug(f"Saved DataFrame to {output_path}")
            logging.debug(f"Shape: {df.shape}")
            logging.debug(f"Columns: {df.columns.tolist()}")
            
            return output_path
        except Exception as e:
            logging.error(f"Error saving Excel file: {str(e)}")
            return ""
