import os
import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import argparse
from datetime import datetime
from email.utils import parsedate_to_datetime
import time
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Conditionally import semantic chunker
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
    print("Warning: langchain_experimental not installed or semantic chunker not available.")
    print("Semantic chunking option will be disabled.")


def extract_email_address(sender_string: str) -> Optional[str]:
    """
    Extract email address from various sender string formats.

    Examples:
    - "Jesse Chen (via eClass)" <noreply@ualberta.ca> -> noreply@ualberta.ca
    - user@example.com -> user@example.com
    - Display Name <email@example.com> -> email@example.com

    Args:
        sender_string: Raw sender string

    Returns:
        Extracted email address or None if no email found
    """
    if not sender_string:
        return None

    # Pattern to match email in angle brackets: anything like <email@example.com>
    angle_bracket_pattern = r'<([^<>]+@[^<>]+\.[^<>]+)>'
    match = re.search(angle_bracket_pattern, sender_string)
    if match:
        return match.group(1).lower().strip()

    # Pattern to match a standard email address without brackets
    email_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    match = re.search(email_pattern, sender_string)
    if match:
        return match.group(0).lower().strip()

    return None


# Configuration class
class Config:
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_CHROMA_DIR = "./chromaDB"
    DEFAULT_INPUT_DIR = "gmail_emails_batch"
    DEFAULT_METADATA_FIELDS = ["subject", "sender", "date"]
    EMBEDDING_MODEL = "text-embedding-3-small"


# Pydantic models for email data
class MetaData(BaseModel):
    id: str
    thread_id: str
    snippet: Optional[str] = None
    subject: Optional[str] = None
    sender: Optional[str] = None
    sender_email: Optional[str] = None  # Add this new field
    date: Optional[str] = None
    date_timestamp: Optional[float] = None


class EmailData(BaseModel):
    metadata: MetaData
    body: Optional[str] = None
    body_chunks: Optional[List[str]] = None


# Text processing utilities
class TextProcessor:
    @staticmethod
    def clean_chunk(chunk: str) -> str:
        """
        Clean a single chunk, removing unwanted content that might have been split across chunk boundaries.
        """
        # Remove any partial URLs at chunk boundaries
        chunk = chunk.replace("[link]", "")

        # Remove any isolated CSS properties that might have been split
        chunk = chunk.strip()

        # Remove any incomplete sentences at the beginning that start with lowercase
        # This likely means it's a continuation of a previous chunk
        if chunk and not chunk[0].isupper() and not chunk[0].isdigit():
            first_sentence_end = chunk.find(". ")
            if first_sentence_end > 0 and first_sentence_end < len(chunk) // 3:
                chunk = chunk[first_sentence_end + 2:]

        return chunk.strip()

    @classmethod
    def chunk_text(cls,
                   text: str,
                   chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
                   chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP,
                   use_semantic_chunking: bool = False,
                   openai_api_key: Optional[str] = None) -> List[str]:
        """
        Split text into chunks using recursive character splitting by default,
        with optional semantic chunking if available.

        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            use_semantic_chunking: Whether to use semantic chunking
            openai_api_key: OpenAI API key (required for semantic chunking)

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Track if semantic chunking is actually used
        semantic_chunking_used = False
        semantic_chunks = [text]

        # Try semantic chunking if requested and available
        if use_semantic_chunking and SEMANTIC_CHUNKER_AVAILABLE and openai_api_key:
            try:
                embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model=Config.EMBEDDING_MODEL
                )
                semantic_splitter = SemanticChunker(
                    embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=80
                )
                temp_chunks = semantic_splitter.split_text(text)

                # Only use semantic chunks if we got results
                if temp_chunks:
                    semantic_chunks = temp_chunks
                    semantic_chunking_used = True
                    print(f"Created {len(semantic_chunks)} semantic chunks")
                else:
                    print(
                        "Semantic chunking returned no chunks, falling back to character splitting only")
            except Exception as e:
                print(f"Error during semantic chunking: {e}")
                print("Falling back to character splitting only")

        # Character-based chunking (applied to each semantic chunk if semantic chunking was used)
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Apply character splitting to each chunk
        final_chunks = []
        for chunk in semantic_chunks:
            char_chunks = char_splitter.split_text(chunk)
            final_chunks.extend(char_chunks)

        # Clean each chunk and filter out empty ones
        cleaned_chunks = [cls.clean_chunk(chunk) for chunk in final_chunks]
        cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk.strip()]

        return cleaned_chunks


# Document creation utility
class DocumentCreator:
    @staticmethod
    def create_documents(email: EmailData,
                         inject_metadata_fields: List[str],
                         store_metadata_fields: List[str]) -> List[Document]:
        """
        Create Document objects for each chunk, with metadata.

        Args:
            email: The EmailData object
            inject_metadata_fields: Fields to inject into the chunk text
            store_metadata_fields: Fields to store in the document metadata

        Returns:
            List of Document objects
        """
        metadata_dict = email.metadata.model_dump()

        # Prepare injected metadata text
        injected_info = " | ".join(
            f"{key}: {metadata_dict.get(key)}"
            for key in inject_metadata_fields
            if key in metadata_dict and metadata_dict.get(key) is not None
            and key != "date_timestamp"  # Don't inject the timestamp
        )

        # Filter metadata to only include requested fields
        filtered_metadata = {
            key: metadata_dict.get(key)
            for key in store_metadata_fields
            if key in metadata_dict
        }

        # Create documents
        documents = []
        for chunk in (email.body_chunks or []):
            content = f"{injected_info}\n{chunk}" if injected_info else chunk
            doc = Document(page_content=content, metadata=filtered_metadata)
            documents.append(doc)

        return documents


# Main email processor class
class EmailProcessor:
    def __init__(self,
                 input_dir: str = Config.DEFAULT_INPUT_DIR,
                 chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP,
                 use_semantic_chunking: bool = False,
                 openai_api_key: Optional[str] = None,
                 inject_metadata_fields: Optional[List[str]] = None,
                 chroma_persist_dir: str = Config.DEFAULT_CHROMA_DIR):

        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        self.openai_api_key = openai_api_key
        self.inject_metadata_fields = inject_metadata_fields or Config.DEFAULT_METADATA_FIELDS
        self.chroma_persist_dir = chroma_persist_dir

        # Validate API key if semantic chunking is requested
        if self.use_semantic_chunking and not self.openai_api_key:
            print(
                "Warning: OpenAI API key is required for semantic chunking. Disabling semantic chunking.")
            self.use_semantic_chunking = False

        # API key is required regardless for embeddings
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")

        # Include date_timestamp in metadata fields to store
        self.store_metadata_fields = list(MetaData.model_fields.keys())
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model=Config.EMBEDDING_MODEL
        )

    def load_email_file(self, filepath: str) -> Optional[EmailData]:
        """Load a single email file and convert to EmailData object."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                email_json = json.load(f)

            # Extract cleaned body
            cleaned_body = email_json.get('cleaned_body', '')
            if not cleaned_body:
                return None

            # Get date string
            date_str = email_json.get('date', '')

            # Convert date string to timestamp if possible
            date_timestamp = None
            if date_str:
                try:
                    # Parse email date format to datetime
                    dt = parsedate_to_datetime(date_str)
                    # Convert to Unix timestamp (seconds since epoch)
                    date_timestamp = dt.timestamp()
                except Exception as e:
                    print(f"Error parsing date '{date_str}': {e}")
                    # Fallback to current time if date parsing fails
                    date_timestamp = time.time()

            # Get sender string
            sender = email_json.get('from', 'Unknown')

            # Extract email address from sender
            email_address = extract_email_address(sender)

            # Create metadata object with timestamp and extracted email
            metadata = MetaData(
                id=email_json.get('id', ''),
                thread_id=email_json.get('threadId', ''),
                snippet=email_json.get('snippet', ''),
                subject=email_json.get('subject', 'No Subject'),
                sender=sender,  # Keep the original sender
                sender_email=email_address,  # Add extracted email address as a new field
                date=date_str,
                date_timestamp=date_timestamp
            )

            # Create email data object
            return EmailData(
                metadata=metadata,
                body=cleaned_body
            )
        except Exception as e:
            print(f"Error loading email file {filepath}: {e}")
            return None

    def process_all_emails(self) -> List[Document]:
        """Process all emails in the input directory into documents."""
        # Create output directory
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

        # Find email files
        print(f"Loading emails from {self.input_dir}...")
        email_files = [f for f in os.listdir(
            self.input_dir) if f.endswith('.json')]
        print(f"Found {len(email_files)} email files")

        all_documents = []
        total_chunks = 0

        # Process each email
        for i, filename in enumerate(email_files):
            if i % 10 == 0:
                print(f"Processing email {i+1}/{len(email_files)}...")

            filepath = os.path.join(self.input_dir, filename)
            email_data = self.load_email_file(filepath)

            if not email_data:
                continue

            # Process the email
            documents = self.process_single_email(email_data)
            all_documents.extend(documents)

            # Update chunk count
            total_chunks += len(email_data.body_chunks or [])

            # Debug output for first few emails
            if i < 5:
                print(f"\n--- Email {i+1} ---")
                print(f"Subject: {email_data.metadata.subject}")
                print(f"Date: {email_data.metadata.date}")
                print(f"Date Timestamp: {email_data.metadata.date_timestamp}")
                print(f"Original Length: {len(email_data.body)} characters")
                print(f"Number of chunks: {len(email_data.body_chunks or [])}")
                if email_data.body_chunks and email_data.body_chunks[0]:
                    print(
                        f"First chunk ({len(email_data.body_chunks[0])} chars): {email_data.body_chunks[0][:100]}...")

        print(
            f"\nProcessed {len(email_files)} emails into {total_chunks} chunks.")
        return all_documents

    # Add this method to your EmailProcessor class

    def process_single_email(self, email_data: EmailData) -> List[Document]:
        """Process a single email into documents."""
        # Chunk the email body
        chunks = TextProcessor.chunk_text(
            email_data.body,
            self.chunk_size,
            self.chunk_overlap,
            self.use_semantic_chunking,
            self.openai_api_key
        )

        email_data.body_chunks = chunks

        # Create documents
        return DocumentCreator.create_documents(
            email_data,
            self.inject_metadata_fields,
            self.store_metadata_fields
        )

    def store_in_chroma(self, documents: List[Document]) -> Chroma:
        """Store documents in ChromaDB."""
        print(f"Storing {len(documents)} documents in ChromaDB...")
        chroma_db = Chroma.from_documents(
            documents,
            embedding=self.embeddings,
            persist_directory=self.chroma_persist_dir
        )

        print(
            f"ChromaDB stored at: {os.path.abspath(self.chroma_persist_dir)}")
        return chroma_db

    def process(self) -> Chroma:
        """Run the full email processing pipeline."""
        documents = self.process_all_emails()
        return self.store_in_chroma(documents)


def main():
    parser = argparse.ArgumentParser(
        description="Process emails, chunk text, and store in ChromaDB"
    )
    parser.add_argument("--input_dir", type=str, default=Config.DEFAULT_INPUT_DIR,
                        help=f"Directory containing email JSON files (default: {Config.DEFAULT_INPUT_DIR})")
    parser.add_argument("--chunk_size", type=int, default=Config.DEFAULT_CHUNK_SIZE,
                        help=f"Maximum size of each chunk (default: {Config.DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=Config.DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap between chunks (default: {Config.DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--use_semantic_chunking", action="store_true",
                        help="Use semantic chunking in addition to character chunking (default: False)")
    parser.add_argument("--openai_api_key", type=str, required=True,
                        help="OpenAI API key for embeddings and optional semantic chunking")
    parser.add_argument("--chroma_dir", type=str, default=Config.DEFAULT_CHROMA_DIR,
                        help=f"Directory to persist ChromaDB (default: {Config.DEFAULT_CHROMA_DIR})")

    args = parser.parse_args()

    # Check if semantic chunking was requested but not available
    if args.use_semantic_chunking and not SEMANTIC_CHUNKER_AVAILABLE:
        print("Warning: Semantic chunking was requested but is not available.")
        print("Install langchain_experimental package to enable this feature.")
        print("Proceeding with recursive character splitting only.")
        args.use_semantic_chunking = False

    # Initialize and run the processor
    processor = EmailProcessor(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=args.use_semantic_chunking,
        openai_api_key=args.openai_api_key,
        chroma_persist_dir=args.chroma_dir
    )

    processor.process()


if __name__ == "__main__":
    main()
