import os
import re
import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

class RAGDatabase:
    def __init__(self, config):
        """Initialize the RAG database with a PDF document"""
        self.config = config
        self.client = OpenAI(api_key=config.get('llms', {}).get('openai', {}).get('api_key'))
        self.document_chunks = []
        self.document_embeddings = []
        self.chunk_size = 1000
        self.overlap = 200
        
    def load_pdf(self, pdf_path):
        """Load and chunk a PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        # Extract text from PDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Create overlapping chunks
        self.document_chunks = self._create_chunks(text)
        
        # Generate embeddings for each chunk
        self._generate_embeddings()
        
        return len(self.document_chunks)
    
    def _create_chunks(self, text):
        """Split the text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good breaking point (end of sentence)
            if end < len(text):
                # Look for the last period, question mark, or exclamation point within 100 characters
                last_period = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end)
                )
                
                if last_period != -1 and last_period > start + 100:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - self.overlap
        
        return chunks
    
    def _generate_embeddings(self):
        """Generate embeddings for all chunks using OpenAI's embedding model"""
        self.document_embeddings = []
        
        for chunk in self.document_chunks:
            response = self.client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            self.document_embeddings.append(response.data[0].embedding)
    
    def search(self, query, top_k=3):
        """Search for relevant chunks based on the query"""
        # Generate embedding for the query
        query_response = self.client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate similarity with all document chunks
        similarities = []
        for doc_embedding in self.document_embeddings:
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities.append(similarity)
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.document_chunks[idx],
                "similarity": similarities[idx]
            })
        
        return results 