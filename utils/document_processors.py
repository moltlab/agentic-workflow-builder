import os
import fitz  # PyMuPDF
import pandas as pd
import csv
from pathlib import Path

class DocumentProcessor:
    """Base class for document processors"""
    @staticmethod
    def get_processor(file_path):
        """Factory method to get the appropriate processor based on file extension"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return PDFProcessor()
        elif ext == '.csv':
            return CSVProcessor()
        elif ext == '.txt':
            return TextProcessor()
        elif ext in ['.docx', '.doc']:
            return DocxProcessor()
        elif ext in ['.json', '.jsonl']:
            return JSONProcessor()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

class PDFProcessor:
    """Process PDF documents"""
    def extract_text(self, file_path):
        """Extract text from a PDF file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print("11111")
        doc = fitz.open(file_path)
        print("22222")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class CSVProcessor:
    """Process CSV documents"""
    def extract_text(self, file_path):
        """Extract text from a CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        # Convert each row to a string representation
        rows = []
        for _, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
            rows.append(row_str)
        
        return "\n".join(rows)

class TextProcessor:
    """Process plain text documents"""
    def extract_text(self, file_path):
        """Extract text from a plain text file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

class DocxProcessor:
    """Process DOCX documents"""
    def extract_text(self, file_path):
        """Extract text from a DOCX file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx package required for DOCX processing. Install with 'pip install python-docx'")

class JSONProcessor:
    """Process JSON documents"""
    def extract_text(self, file_path):
        """Extract text from a JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                # Handle JSONL (JSON Lines) format
                lines = []
                for line in f:
                    json_obj = json.loads(line)
                    lines.append(json.dumps(json_obj, ensure_ascii=False))
                return "\n".join(lines)
            else:
                # Handle regular JSON
                data = json.load(f)
                return json.dumps(data, ensure_ascii=False, indent=2) 