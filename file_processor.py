"""
File Processor - Handles citation file parsing and Excel I/O operations.
Optimized for efficient file handling with streaming and chunked processing.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Constants
REQUIRED_COLUMNS = ('Title', 'Authors', 'Abstract', 'DOI')
PREVIEW_RECORD_COUNT = 3
PREVIEW_FIELD_LENGTHS = {'DOI': 50, 'Title': 100, 'Authors': 100, 'Abstract': 200}

# Pre-compiled regex patterns
SCOPUS_RECORD_PATTERN = re.compile(r'\nER\s*-\s*')


class FileProcessor:
    """Handles citation file parsing and Excel I/O operations."""

    __slots__ = ('data_dir',)

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def parse_nbib(self, file_path: str) -> Tuple[Optional[str], str]:
        """Parse PubMed NBIB file to Excel format."""
        if not self._validate_file(file_path):
            return None, "Invalid file"

        try:
            records = []
            record: Dict[str, str] = {}
            authors: List[str] = []
            current_field: Optional[str] = None

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('TI  - '):
                        record['Title'] = line[6:].strip()
                        current_field = 'Title'
                    elif line.startswith('AB  - '):
                        record['Abstract'] = line[6:].strip()
                        current_field = 'Abstract'
                    elif line.startswith('AU  - '):
                        authors.append(line[6:].strip())
                        current_field = None
                    elif line.startswith('LID - ') and '[doi]' in line:
                        record['DOI'] = line[6:].replace(' [doi]', '').strip()
                        current_field = None
                    elif line.startswith('PMID- '):
                        if record:
                            record['Authors'] = '; '.join(authors)
                            records.append(record)
                            record = {}
                            authors = []
                        current_field = None
                    elif line.startswith('      ') and current_field in ('Abstract', 'Title'):
                        record[current_field] += ' ' + line.strip()

            # Save last record
            if record:
                record['Authors'] = '; '.join(authors)
                records.append(record)

            return self._save_records(records, "extracted_data.xlsx")

        except Exception as e:
            logging.error(f"NBIB parsing error: {e}")
            return None, f"Error: {str(e)}"

    def parse_wos_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """Parse Web of Science RIS file to Excel format."""
        if not self._validate_file(file_path):
            return None, "Invalid file"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                return None, "Empty file"

            records = []
            for article in content.split("\nER  -"):
                if not article.strip():
                    continue

                record: Dict[str, str] = {}
                authors: List[str] = []

                for line in article.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('TI  - '):
                        record['Title'] = line[6:].strip()
                    elif line.startswith('AB  - '):
                        record['Abstract'] = line[6:].strip()
                    elif line.startswith('AU  - '):
                        authors.append(line[6:].strip())
                    elif line.startswith('DO  - '):
                        record['DOI'] = line[6:].strip()
                    elif line.startswith('   '):
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()

                if record:
                    record['Authors'] = '; '.join(authors)
                    records.append(record)

            return self._save_records(records, "extracted_data.xlsx")

        except Exception as e:
            logging.error(f"WOS RIS parsing error: {e}")
            return None, f"Error: {str(e)}"

    def parse_embase_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """Parse Embase RIS file to Excel format."""
        if not self._validate_file(file_path):
            return None, "Invalid file"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                return None, "Empty file"

            records = []
            for article in content.split("\n\n"):
                if not article.strip():
                    continue

                record: Dict[str, str] = {}
                authors: List[str] = []

                for line in article.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('T1  - '):
                        record['Title'] = line[6:].strip()
                    elif line.startswith('N2  - '):
                        record['Abstract'] = line[6:].strip()
                    elif line.startswith('A1  - '):
                        authors.append(line[6:].strip())
                    elif line.startswith('DO  - '):
                        record['DOI'] = line[6:].strip()
                    elif line.startswith('   '):
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()

                if record:
                    record['Authors'] = '; '.join(authors) if authors else ''
                    records.append(record)

            return self._save_records(records, "extracted_data.xlsx")

        except Exception as e:
            logging.error(f"Embase RIS parsing error: {e}")
            return None, f"Error: {str(e)}"

    def parse_scopus_ris(self, file_path: str) -> Tuple[Optional[str], str]:
        """Parse Scopus RIS file to Excel format."""
        if not self._validate_file(file_path):
            return None, "Invalid file"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                return None, "Empty file"

            records = []
            for article in SCOPUS_RECORD_PATTERN.split(content):
                if not article.strip():
                    continue

                record: Dict[str, str] = {}
                authors: List[str] = []

                for line in article.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('TI  - '):
                        record['Title'] = line[6:].strip()
                    elif line.startswith('AB  - '):
                        record['Abstract'] = line[6:].strip()
                    elif line.startswith('AU  - '):
                        authors.append(line[6:].strip())
                    elif line.startswith('DO  - '):
                        record['DOI'] = line[6:].strip()
                    elif line.startswith('   '):
                        if 'Abstract' in record:
                            record['Abstract'] += ' ' + line.strip()
                        elif 'Title' in record:
                            record['Title'] += ' ' + line.strip()

                record['Authors'] = '; '.join(authors)
                records.append(record)

            return self._save_records(records, "extracted_data.xlsx")

        except Exception as e:
            logging.error(f"Scopus RIS parsing error: {e}")
            return None, f"Error: {str(e)}"

    def load_excel(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load Excel file with proper index handling."""
        try:
            df = pd.read_excel(file_path, index_col=0)

            # Ensure proper index setup
            if "Index" in df.columns:
                df.set_index("Index", inplace=True)
            elif df.index.name != "Index":
                df.index.name = "Index"

            # Normalize index
            df.index = df.index.astype(str).str.strip()

            # Remove duplicates
            if df.index.duplicated().any():
                logging.warning(f"Removing duplicate indices in {file_path}")
                df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            logging.error(f"Excel load error: {e}")
            return None

    def save_excel(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to Excel file."""
        try:
            df = df.copy()

            # Handle Index column conflict
            if "Index" in df.columns:
                # If there's already an Index column, save it as Original_Index to avoid conflict
                df = df.rename(columns={"Index": "Original_Index"})

            # Ensure proper index
            if df.index.name != "Index":
                df.index.name = "Index"
            df.index = df.index.astype(str)

            # Remove duplicates
            if df.index.duplicated().any():
                logging.warning(f"Removing duplicate indices when saving {filename}")
                df = df[~df.index.duplicated(keep='first')]

            output_path = os.path.join(self.data_dir, filename)
            df.to_excel(output_path, index=True)

            return output_path

        except Exception as e:
            logging.error(f"Excel save error: {e}")
            return ""

    def _validate_file(self, file_path: str) -> bool:
        """Validate file exists and is readable."""
        return bool(file_path and os.path.exists(file_path))

    def _save_records(self, records: List[Dict], filename: str) -> Tuple[Optional[str], str]:
        """Save parsed records to Excel and generate preview."""
        if not records:
            return None, "No records found"

        df = pd.DataFrame(records)

        # Ensure all required columns exist
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ''

        df.index.name = 'Index'
        output_path = os.path.join(self.data_dir, filename)
        df.to_excel(output_path, index=True)

        preview = self._generate_preview(records)
        return output_path, preview

    def _generate_preview(self, records: List[Dict]) -> str:
        """Generate preview text for parsed records."""
        lines = []

        for i, record in enumerate(records[:PREVIEW_RECORD_COUNT]):
            lines.append(f"\nRecord {i}:")
            for field, max_len in PREVIEW_FIELD_LENGTHS.items():
                value = record.get(field, '')[:max_len]
                suffix = '...' if len(record.get(field, '')) > max_len else ''
                lines.append(f"{field}: {value}{suffix}")
            lines.append("-" * 80)

        lines.append(f"\nTotal records extracted: {len(records)}")
        return '\n'.join(lines)
