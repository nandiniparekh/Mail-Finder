from fastapi.responses import JSONResponse
from fastapi import Request
import os
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import shutil

import chromadb
from chromadb.errors import InvalidCollectionException

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import Runnable
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet

# Import FastAPI and LangServe
from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from email.utils import parsedate_to_datetime

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import base64
import json


# Configure global variables
openai_api_key = ""
chroma_dir = os.getenv("CHROMA_DIR", "./chromaDB_recursive")
email_dir = "gmail_emails_batch"  # Directory for email data
encryption_key = b"KUwAEgxBHwPjcW-uLfD8yfCgWUzVn3IH1tlBEIV535Q="
fernet = Fernet(encryption_key)

# Database initialization functions


def initialize_email_database():
    """Initialize the email database if it doesn't exist"""
    # Check if Chroma directory exists
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print(
            f"ChromaDB already exists at {chroma_dir}, skipping initialization")
        return

    # Check if email directory exists
    if not os.path.exists(email_dir):
        print(
            f"Email directory {email_dir} doesn't exist, running email fetch script")
        # Run Gmail fetch script
        try:
            print("Starting email fetching process...")
            subprocess.run(
                [sys.executable, "get_emails.py"], check=True)
            print("Email fetching completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running email fetch script: {e}")
            raise Exception(
                "Failed to initialize email database: Email fetch failed")
    else:
        print(
            f"Email directory {email_dir} already exists, skipping email fetch")

    # Run email processing script
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        print("Running email processing script to create ChromaDB...")
        try:
            # Run email processing script with appropriate arguments
            subprocess.run([
                sys.executable,
                "chunk_emails.py",
                "--input_dir", email_dir,
                "--openai_api_key", openai_api_key,
                "--chroma_dir", chroma_dir
            ], check=True)
            print(f"ChromaDB successfully created at {chroma_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error running email processing script: {e}")
            raise Exception(
                "Failed to initialize email database: Email processing failed")


# Initialize components
embeddings = None
chroma_db = None
hybrid_searcher = None
high_ranked_keywords_from_query = []

# Initialize FastAPI app
app = FastAPI(
    title="Email Search API",
    description="API for searching emails using hybrid search and reranking",
    version="1.0.0",
)

# Allow requests from your Chrome extension
app.add_middleware(
    CORSMiddleware,
    # Replace with your extension ID
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom JSON serializer for NumPy types


def numpy_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# Define Pydantic models for input/output


@app.post("/trigger-email-fetch")
def trigger_email_fetch():
    """Endpoint to trigger the complete email fetch and processing pipeline"""
    # Declare all globals at the beginning of the function
    global embeddings, chroma_db, hybrid_searcher, chroma_dir

    try:
        # Delete the existing gmail_emails_batch folder if it exists
        if os.path.exists(email_dir):
            print(f"Deleting existing email directory: {email_dir}")
            shutil.rmtree(email_dir)

        # Check if ChromaDB directory exists and delete it
        if os.path.exists(chroma_dir):
            print(f"Deleting existing chroma_db directory: {chroma_dir}")
            shutil.rmtree(chroma_dir)

        # Release any resources before deletion
        chroma_db = None
        hybrid_searcher = None
        embeddings = None

        # Ensure directory is deleted
        import time
        time.sleep(1)  # Short pause to ensure resources are released

        print("Triggering email fetch script...")
        subprocess.run(
            [sys.executable, "get_emails.py"], check=True
        )

        print("Running email processing script to create ChromaDB...")
        subprocess.run([
            sys.executable,
            "chunk_emails.py",
            "--input_dir", email_dir,
            "--openai_api_key", openai_api_key,
            "--chroma_dir", chroma_dir
        ], check=True)

        # Reinitialize components after rebuilding the database
        print("Reinitializing components with new database...")

        # Use init_components() instead of manually initializing each component
        # This ensures proper initialization of all components
        init_components()

        chroma_dir = os.getenv("CHROMA_DIR", "./chromaDB_recursive")

        return {"status": "success", "message": "Email fetching and processing completed successfully"}
    except subprocess.CalledProcessError as e:
        print(f"Error running email processing: {e}")
        return {"status": "error", "message": f"Failed to process emails: {e}"}
    except Exception as e:
        print(f"Unexpected error during email fetch/processing: {e}")
        return {"status": "error", "message": f"Unexpected error: {e}"}


# Define Pydantic models for input/output


class SearchInput(BaseModel):
    """Input for email search"""
    query: str
    k: int = 5
    filter_criteria: Optional[Dict[str, Any]] = None
    search_method: str = "hybrid"
    hybrid_alpha: float = 0.5
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    top_n: int = 5
    generate_answer: bool = False
    answer_model: str = "gpt-4o"

    # Date and sender filters
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sender: Optional[str] = None

    # Previous queries for context
    previous_queries: Optional[List[str]] = None


class DocumentOutput(BaseModel):
    """Document output schema"""
    content: str
    metadata: Dict[str, Any]


class SearchOutput(BaseModel):
    """Output for email search"""
    results: List[DocumentOutput]
    answer: Optional[str] = None
    # Add this for tracking filtered results
    filtered_info: Optional[Dict[str, Any]] = None


class HybridSearcher:
    """Hybrid search combining vector similarity and BM25 lexical search."""

    def __init__(
        self,
        chroma_db: Chroma,
        alpha: float = 0.5,
        min_bm25_score: float = 1.0,
        include_metadata: bool = True
    ):
        """
        Initialize the hybrid searcher.

        Args:
            chroma_db: ChromaDB instance
            alpha: Weight for vector search scores (1-alpha for BM25)
            min_bm25_score: Minimum BM25 score to consider
            include_metadata: Whether to include metadata in BM25 index
        """
        self.chroma_db = chroma_db
        self.alpha = alpha
        self.min_bm25_score = min_bm25_score
        self.include_metadata = include_metadata

        # Initialize BM25 with all documents in the collection
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Initialize BM25 index with all documents in ChromaDB."""
        print("Initializing BM25 index...")

        # Get all documents from ChromaDB
        collection = self.chroma_db._collection
        results = collection.get()

        # Extract documents and their IDs
        self.documents = []
        self.doc_ids = []

        if not results['documents']:
            raise ValueError("No documents found in ChromaDB collection")

        # Prepare documents for BM25
        for i, (doc_id, document, metadata) in enumerate(zip(
            results['ids'],
            results['documents'],
            results['metadatas']
        )):
            # Add document ID for reference
            self.doc_ids.append(doc_id)

            # Combine content with metadata if requested
            if self.include_metadata and metadata:
                # Extract key metadata fields
                meta_text = ""
                for key, value in metadata.items():
                    if value and isinstance(value, str):
                        meta_text += f"{key}: {value} "

                # Combine metadata with document text
                full_text = f"{meta_text} {document}"
                self.documents.append(full_text)
            else:
                self.documents.append(document)

        # Tokenize documents
        tokenized_corpus = [self._tokenize(doc) for doc in tqdm(
            self.documents, desc="Tokenizing documents")]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index created with {len(self.documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, remove special chars, split by whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def search(
        self,
        query: str,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results to return
            filter_criteria: Filter criteria for vector search
            fetch_k: Number of documents to fetch from vector search (defaults to 2*k)

        Returns:
            List of Document objects
        """
        if fetch_k is None:
            fetch_k = 2 * k  # Default to fetching more docs than needed

        # Step 1: Vector search
        vector_docs = self.chroma_db.similarity_search(
            query,
            k=fetch_k,
            filter=filter_criteria
        )

        # Create a mapping from document content to Document object
        vector_doc_map = {doc.page_content: doc for doc in vector_docs}

        # Get vector search scores
        # Note: ChromaDB doesn't directly expose scores, so we'll use order as proxy
        # In a production system, you'd want to get actual scores
        vector_scores = {doc.page_content: (
            fetch_k - i) / fetch_k for i, doc in enumerate(vector_docs)}

        # Step 2: BM25 search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1] range
        max_bm25 = max(bm25_scores) if bm25_scores.any() else 1.0
        normalized_bm25_scores = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

        # Create a map of document content to BM25 score
        bm25_doc_scores = {}
        for i, score in enumerate(normalized_bm25_scores):
            if score >= self.min_bm25_score / max_bm25:  # Apply minimum score threshold
                content = self.documents[i]
                bm25_doc_scores[content] = score

        # Step 3: Combine scores for documents found in vector search
        combined_scores = {}
        for doc_content, vector_score in vector_scores.items():
            # Find the corresponding BM25 score
            # Note: This is a simple approach - in practice, you'd need exact matching
            bm25_score = 0
            for bm25_content, score in bm25_doc_scores.items():
                # Simple string matching - in production, use better matching logic
                if doc_content in bm25_content or bm25_content in doc_content:
                    bm25_score = score
                    break

            # Combine scores using weighted sum
            combined_scores[doc_content] = (
                self.alpha * vector_score + (1 - self.alpha) * bm25_score
            )

        # Sort documents by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top k documents
        results = []
        for i, (content, score) in enumerate(sorted_docs[:k]):
            doc = vector_doc_map.get(content)
            if doc:
                # Add score to metadata for debugging/display
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                # Convert numpy float to Python float to avoid serialization issues
                doc.metadata['hybrid_score'] = float(score)
                results.append(doc)

        return results


def init_components():
    """Initialize the necessary components for search"""
    global embeddings, chroma_db, hybrid_searcher

    try:
        # Initialize email database first
        if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
            print("Database does not exist. Initializing email database...")
            initialize_email_database()

        if not os.path.exists(chroma_dir):
            raise ValueError(
                f"ChromaDB directory {chroma_dir} does not exist after initialization")

        # Initialize embedding model
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )

        # Load ChromaDB
        chroma_db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )

        # Test the connection to ensure the collection is valid
        try:
            # Try to access the collection with a simple query
            _ = chroma_db._collection.get(limit=1)
        except Exception as e:
            print(f"Error accessing ChromaDB collection: {e}")
            # If there's an error, create a new collection
            print("Attempting to rebuild ChromaDB collection...")
            # Recreate the collection by reinitializing Chroma
            chroma_db = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings,
                # Use a new collection name
                collection_name=f"emails_{int(time.time())}"
            )

        # Initialize hybrid searcher with the (possibly new) collection
        hybrid_searcher = HybridSearcher(
            chroma_db=chroma_db,
            alpha=0.5
        )

        print("Components initialized successfully")
    except Exception as e:
        print(f"Error initializing components: {e}")
        # Re-raise the exception to be handled by the caller
        raise


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    return obj


def format_documents_for_response(docs: List[Document]) -> List[Dict[str, Any]]:
    """Format Document objects for JSON response"""
    results = []
    for doc in docs:
        # Convert any numpy types in metadata to Python native types
        metadata = convert_numpy_types(doc.metadata)

        results.append({
            "content": doc.page_content,
            "metadata": metadata
        })
    return results


def validate_response(content: List[str], high_ranked_keywords: List[str], threshold: float = 0.7) -> bool:
    """
    Validate the LLM response by comparing the high_ranked_keywords_from_query
    with the content using semantic similarity.
    """
    if not high_ranked_keywords or not content:
        print("Validation failed: Keywords or content is empty.")
        return False

    try:
        # Initialize the embedding model
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Generate embeddings for keywords and content
        keyword_embeddings = [embeddings.embed_query(
            keyword) for keyword in high_ranked_keywords]
        content_embedding = embeddings.embed_query(" ".join(content))

        # Compute the average similarity between the content and the keywords
        similarities = [
            cosine_similarity([content_embedding], [keyword_embedding])[0][0]
            for keyword_embedding in keyword_embeddings
        ]
        average_similarity = sum(similarities) / len(similarities)

        print(f"Average similarity score: {average_similarity}")

        # Check if the average similarity meets the threshold
        return average_similarity >= threshold

    except Exception as e:
        print(f"Error during response validation: {e}")
        return False


def generate_answer_from_docs(query: str, documents: List[Document], model_name: str = "gpt-4o", previous_queries: Optional[List[str]] = None) -> str:
    """Generate an answer based on the retrieved documents, formatted as a structured list of emails"""

    global high_ranked_keywords_from_query

    # Implement pre-processing deduplication of documents based on thread_id or email_id
    # This ensures we're not feeding duplicate emails to the LLM in the first place
    unique_doc_ids = set()
    deduplicated_documents = []

    for doc in documents:
        # Look for any identifier field that might represent a unique email
        thread_id = doc.metadata.get("thread_id")
        email_id = doc.metadata.get("email_id")
        doc_id = doc.metadata.get("id")

        # Use the first available identifier, prioritizing thread_id
        unique_id = thread_id or email_id or doc_id

        if unique_id and unique_id not in unique_doc_ids:
            unique_doc_ids.add(unique_id)
            deduplicated_documents.append(doc)

    # Use deduplicated documents for the rest of the processing
    documents = deduplicated_documents

    # Format context from Document format
    context = ""
    for doc in documents:
        # Convert numpy types to avoid serialization issues
        metadata = convert_numpy_types(doc.metadata)
        metadata_str = json.dumps(metadata, indent=2)
        context += f"Content: {doc.page_content}\nMetadata: {metadata_str}\n\n"

    # Add previous queries context if available
    query_context = ""
    if previous_queries and len(previous_queries) > 0:
        query_context = "Previous search queries:\n" + \
            "\n".join([f"- {q}" for q in previous_queries]) + "\n\n"

    system_prompt = """
    You are an AI assistant that helps users find information in their emails.

    You will be given:
    1. A user query about their emails
    2. Context from the most relevant email chunks
    3. Optionally, previous search queries the user has made

    Your task is to identify the top 5 most relevant UNIQUE emails (not chunks) from the context.
    Do NOT return the same email multiple times, even if multiple chunks from that email are provided.

    For each email in the context, extract these exact fields:
    - email_id: Use "thread_id" from metadata as the unique identifier. If "thread_id" is not available, use "id".
    - subject: The email subject line (from metadata)
    - sender: The sender of the email (from metadata)
    - date: The date of the email (from metadata)
    - snippet: A brief, relevant snippet from the email content (limit to ~100 characters)
    - relevance_score: A number from 0-10 indicating how relevant this email is to the query

    IMPORTANT: 
    - ABSOLUTELY ENSURE each email appears ONLY ONCE in your response, no matter how many chunks belong to it
    - If multiple chunks belong to the same email, choose the most relevant one for the snippet
    - Check ALL identifier fields (thread_id, id, email_id) to ensure uniqueness
    - Return ONLY up to 5 unique emails

    Return ONLY a raw JSON array without any markdown formatting, explanation or surrounding characters.
    """

    user_message = f"""
    ### Query:
    {query}

    {query_context}### Context from most relevant emails:
    {context}

    Remember to return ONLY the raw JSON array with AT MOST 5 UNIQUE emails based on their identifiers.
    """

    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=openai_api_key
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    # Invoke the LLM
    response = llm.invoke(messages)

    # Clean any potential markdown formatting from the response
    content = response.content.strip()

    # Remove markdown code block formatting if present
    if content.startswith("```") and content.endswith("```"):
        # Find the first newline to skip the language identifier line if it exists
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]

        # Remove the closing backticks
        if content.endswith("```"):
            content = content[:-3].strip()

    # More aggressive cleaning if needed
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    # Parse the response into JSON
    try:
        emails = json.loads(content)

        # Apply our own deduplication to be absolutely certain
        seen_ids = set()
        unique_emails = []

        for email in emails:
            # Check all possible ID fields
            email_id = email.get("email_id")

            # Skip if we've already seen this ID
            if email_id and email_id in seen_ids:
                continue

            # Add to unique emails and mark as seen
            if email_id:
                seen_ids.add(email_id)
            unique_emails.append(email)

        # Ensure we have at most 5 results
        unique_emails = unique_emails[:5]

        # Return the final JSON
        final_json = json.dumps(unique_emails)
        print(f"Final emails after deduplication: {len(unique_emails)}")
        return final_json

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {content}")
        return "[]"


def preprocess_query(query):
    global high_ranked_keywords_from_query
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    query = query.strip()

    try:
        system_prompt = """
            You are a Gmail query preprocessing assistant.
            Given a user's Gmail search query, perform the following tasks:
            1. Clean the query by expanding abbreviations, correcting grammatical inconsistencies, and improving overall clarity.
            2. Identify and rank the most relevant keywords and phrases in the query that will help locate the correct emails. Focus on entities like names, dates, subjects, and action-related phrases.
            
            Respond only with a JSON object in the following format:
            {
            "cleaned_query": "<cleaned and corrected version of the original query>",
            "high_ranked_keywords_from_query": ["<keyword1>", "<keyword2>", "..."]
            }

            IMPORTANT: Do NOT include backticks, language markers like ```json, or any other text outside of the raw JSON array.
            Return ONLY the JSON array with no formatting, explanation or surrounding characters of any kind.
        """

        user_message = f"User Query: {query}"

        # Initialize the LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_api_key
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        # Retry mechanism
        for attempt in range(2):  # Allow up to 2 attempts
            try:
                # Invoke the LLM
                response = llm.invoke(messages)
                response_content = response.content.strip()

                # Attempt to parse the response content
                query = json.loads(response_content)
                high_ranked_keywords_from_query = query.get(
                    "high_ranked_keywords_from_query")

                return query.get("cleaned_query")
            except json.JSONDecodeError as e:
                print(
                    f"Attempt {attempt + 1}: Failed to parse LLM response as JSON. Error: {e}")
                print(f"LLM Response: {response}")
                if attempt == 1:  # If it's the second attempt, raise an error
                    raise ValueError(
                        "LLM failed to provide a valid JSON response after 2 attempts.")

    except Exception as e:
        print(f"Error during query correction: {e}")
        # Fallback to the basic preprocessed query if LLM fails
        return query


def search_emails(input_data: Union[Dict[str, Any], SearchInput]) -> SearchOutput:
    # Convert dict to SearchInput if necessary
    if isinstance(input_data, dict):
        input_data = SearchInput(**input_data)

    try:
        # Simple test query to verify the collection exists and is accessible
        print("Testing ChromaDB collection access...")
        _ = chroma_db._collection.get(limit=1)
    except InvalidCollectionException:  # Use the imported error class
        # If collection doesn't exist, attempt to reinitialize components
        print("Collection not found, reinitializing components...")
        try:
            init_components()
        except Exception as e:
            error_msg = f"Failed to reinitialize components after collection error: {str(e)}"
            print(error_msg)
            return SearchOutput(
                results=[],
                answer=None,
                filtered_info={"error": error_msg}
            )
    except Exception as e:
        # For any other error, return an informative error message
        error_msg = f"Error accessing ChromaDB collection: {str(e)}"
        print(error_msg)
        return SearchOutput(
            results=[],
            answer=None,
            filtered_info={"error": error_msg}
        )

    # Extract parameters
    query = input_data.query
    previous_queries = input_data.previous_queries
    k = input_data.k
    filter_criteria = input_data.filter_criteria
    search_method = input_data.search_method
    hybrid_alpha = input_data.hybrid_alpha
    use_reranker = input_data.use_reranker
    reranker_model = input_data.reranker_model
    top_n = input_data.top_n
    generate_answer_flag = input_data.generate_answer
    answer_model = input_data.answer_model
    start_date = input_data.start_date
    end_date = input_data.end_date
    sender = input_data.sender

    # Process the query
    query = preprocess_query(query)

    # Update hybrid searcher parameters
    if hybrid_searcher:
        hybrid_searcher.alpha = hybrid_alpha

    # Initialize results variable with a default empty list
    results = []

    # -------------------------------------------------------------
    # Build ChromaDB filter according to documentation format
    # -------------------------------------------------------------

    # Start with an empty filter
    chroma_filter = {}

    # Build a list of conditions for $and operator if we have multiple filters
    conditions = []

    # Add sender filter if provided
    if sender:
        # Normalize the sender to lowercase for case-insensitive matching
        sender_lower = sender.lower().strip()

        # Check if the sender looks like an email address (contains @)
        is_email = '@' in sender_lower

        if is_email:
            # Try several approaches to find the email

            # 1. First try matching against sender_email field if it exists
            # NOTE: This will only work if your metadata has this field
            try:
                # Test for existence of sender_email by checking the collection metadata
                # This is just a diagnostic - remove in production
                print("Checking for sender_email field in metadata...")
                # Create an empty filter just to get sample documents
                test_results = chroma_db.similarity_search("test", k=1)
                if test_results and test_results[0].metadata:
                    fields = list(test_results[0].metadata.keys())
                    print(f"Available metadata fields: {fields}")
                    if 'sender_email' in fields:
                        print("sender_email field found!")
                        # Use sender_email field directly with exact match
                        conditions.append(
                            {"sender_email": {"$eq": sender_lower}})
                    else:
                        print(
                            "sender_email field NOT found, using regular sender field")
                        # Fallback to using the regular sender field
                        conditions.append(
                            {"sender": {"$in": [sender_lower, f"<{sender_lower}>"]}})
            except Exception as e:
                print(f"Error checking metadata fields: {e}")
                # Fallback to using the regular sender field
                conditions.append(
                    {"sender": {"$in": [sender_lower, f"<{sender_lower}>"]}})
        else:
            # For non-email senders, use the name with the regular sender field
            conditions.append({"sender": {"$eq": sender_lower}})

        print(f"Final sender search conditions: {conditions}")

    # Add date filters using the numeric timestamp field
    if start_date or end_date:
        from datetime import datetime, timezone, timedelta

        if start_date:
            # Handle ISO format string with explicit timezone
            if 'T' in start_date and start_date.endswith('Z'):
                # It's already a full ISO timestamp with UTC indicator
                start_dt = datetime.fromisoformat(
                    start_date.replace('Z', '+00:00'))
            else:
                # It's just a date, add time component and set to UTC
                start_dt = datetime.fromisoformat(
                    f"{start_date}T00:00:00").replace(tzinfo=timezone.utc)

            # Convert to Unix timestamp (seconds since epoch)
            start_timestamp = start_dt.timestamp()
            conditions.append({"date_timestamp": {"$gte": start_timestamp}})

        if end_date:
            # Handle ISO format string with explicit timezone
            if 'T' in end_date and end_date.endswith('Z'):
                # It's already a full ISO timestamp with UTC indicator
                end_dt = datetime.fromisoformat(
                    end_date.replace('Z', '+00:00'))
            else:
                # It's just a date, add time component and set to UTC
                end_dt = datetime.fromisoformat(
                    f"{end_date}T00:00:00").replace(tzinfo=timezone.utc)

            # Convert to Unix timestamp (seconds since epoch)
            end_timestamp = end_dt.timestamp()
            conditions.append({"date_timestamp": {"$lte": end_timestamp}})

    # If we have filter conditions, combine them with $and
    if conditions:
        if len(conditions) == 1:
            # If only one condition, no need for $and
            chroma_filter = conditions[0]
        else:
            # Multiple conditions need $and
            chroma_filter = {"$and": conditions}

    # Print the filter for debugging
    print(f"Using ChromaDB filter: {chroma_filter}")

    # Choose search method
    if search_method == "vector":
        # Standard vector search
        print("Using standard vector search.")
        if use_reranker:
            # Vector search + reranker
            print(
                f"Using CrossEncoderReranker with model: {reranker_model}, top_n={top_n}")
            try:
                # Get initial results with filter
                fetch_k = k * 3  # Fetch more for reranking

                # Get initial results with filter applied
                initial_results = chroma_db.similarity_search(
                    query,
                    k=fetch_k,
                    filter=chroma_filter if chroma_filter else None  # Use the filter if not empty
                )

                # Set up the cross-encoder reranker
                model = HuggingFaceCrossEncoder(model_name=reranker_model)

                # Create query-document pairs
                sentence_pairs = [[query, doc.page_content]
                                  for doc in initial_results]

                # Compute relevance scores
                scores = model.predict(sentence_pairs)

                # Convert to numpy array if not already
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)

                # Handle 2D arrays (binary classification models)
                if len(scores.shape) > 1 and scores.shape[1] > 1:
                    # Take the positive class probability
                    scores = scores[:, 1]

                # Get indices of top_n documents sorted by score
                top_indices = np.argsort(scores)[::-1][:top_n]

                # Create reranked document list
                results = [initial_results[i] for i in top_indices]

                # Add reranking scores to metadata
                for i, doc in enumerate(results):
                    # Convert numpy float to Python float to avoid serialization issues
                    doc.metadata['rerank_score'] = float(
                        scores[top_indices[i]])

                print(
                    f"Retrieved {len(results)} documents after vector search + reranking")
            except Exception as e:
                print(f"Error using CrossEncoderReranker: {e}")
                print("Falling back to regular search without reranking")
                results = chroma_db.similarity_search(
                    query, k=k, filter=chroma_filter if chroma_filter else None)
        else:
            # Simple vector search without reranking
            results = chroma_db.similarity_search(
                query, k=k, filter=chroma_filter if chroma_filter else None)

    elif search_method == "hybrid" or search_method == "hybrid-rerank":
        # Hybrid search
        print(f"Using hybrid search with alpha={hybrid_alpha}")

        # Get hybrid search results with filter
        fetch_k = k * 3 if use_reranker else k
        hybrid_results = hybrid_searcher.search(
            query=query,
            k=fetch_k,
            filter_criteria=chroma_filter if chroma_filter else None
        )

        # Apply reranking if requested
        if search_method == "hybrid-rerank" and use_reranker:
            print(f"Applying CrossEncoderReranker to hybrid results")
            try:
                # Set up the cross-encoder reranker
                model = HuggingFaceCrossEncoder(model_name=reranker_model)

                # Create query-document pairs
                sentence_pairs = [[query, doc.page_content]
                                  for doc in hybrid_results]

                # Compute relevance scores
                scores = model.predict(sentence_pairs)

                # Convert to numpy array if not already
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)

                # Handle 2D arrays (binary classification models)
                if len(scores.shape) > 1 and scores.shape[1] > 1:
                    # Take the positive class probability
                    scores = scores[:, 1]

                # Get indices of top_n documents sorted by score
                top_indices = np.argsort(scores)[::-1][:top_n]

                # Create reranked document list
                results = [hybrid_results[i] for i in top_indices]

                # Add reranking scores to metadata
                for i, doc in enumerate(results):
                    # Convert numpy float to Python float to avoid serialization issues
                    doc.metadata['rerank_score'] = float(
                        scores[top_indices[i]])

                print(
                    f"Retrieved {len(results)} documents after hybrid search + reranking")
            except Exception as e:
                print(f"Error using CrossEncoderReranker: {e}")
                print("Returning hybrid search results without reranking")
                results = hybrid_results[:k]
        else:
            # Return hybrid results without reranking
            results = hybrid_results[:k]
    else:
        raise ValueError(f"Unknown search method: {search_method}")

    # Store all results for reference
    all_results = results.copy()

    # -------------------------------------------------
    # DEDUPLICATION STEP: Group documents by thread_id/email_id
    # -------------------------------------------------
    # Create a dictionary to group documents by their thread_id
    thread_groups = {}

    for doc in results:
        # Get thread ID (or fallback to other IDs if thread_id is not available)
        thread_id = doc.metadata.get("thread_id")
        if not thread_id:
            # Fallback to other potential ID fields
            thread_id = doc.metadata.get("email_id") or doc.metadata.get("id")

        if not thread_id:
            # Skip documents without any ID (shouldn't happen with properly structured data)
            continue

        # If this is the first chunk from this thread, create a new entry
        if thread_id not in thread_groups:
            thread_groups[thread_id] = []

        # Add this document to its thread group
        thread_groups[thread_id].append(doc)

    # For each thread, select the most relevant chunk based on metadata
    deduplicated_results = []
    for thread_id, docs in thread_groups.items():
        if not docs:
            continue

        # Sort the chunks by their relevance/score if available
        # First try using rerank_score
        if any("rerank_score" in doc.metadata for doc in docs):
            docs.sort(key=lambda x: x.metadata.get(
                "rerank_score", 0), reverse=True)
        # Then try hybrid_score
        elif any("hybrid_score" in doc.metadata for doc in docs):
            docs.sort(key=lambda x: x.metadata.get(
                "hybrid_score", 0), reverse=True)

        # Take the top chunk as the representative for this thread
        deduplicated_results.append(docs[0])

    # Limit to 5 documents max after deduplication
    deduplicated_results = deduplicated_results[:5]

    # Use the deduplicated results from now on
    filtered_results = deduplicated_results

    # Format for response
    formatted_results = format_documents_for_response(filtered_results)

    # Generate answer if requested
    answer = None
    if generate_answer_flag and filtered_results:
        answer_json_str = generate_answer_from_docs(
            query=query,
            documents=filtered_results,
            model_name=answer_model,
            previous_queries=previous_queries
        )

        # Try to parse the JSON response to ensure it's valid
        try:
            # Parse the response to ensure it's valid JSON
            parsed_json = json.loads(answer_json_str)

            # Ensure we only have 5 results max
            if isinstance(parsed_json, list) and len(parsed_json) > 5:
                parsed_json = parsed_json[:5]
                answer_json_str = json.dumps(parsed_json)

            answer = answer_json_str
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON response: {e}")
            print(f"Raw response: {answer_json_str}")
            # Fallback to an empty array if parsing fails
            answer = "[]"

    # Create filtered_info object
    filtered_info = {
        "has_results_outside_filter": False,  # Since filtering is done at retrieval
        "total_results_before_filter": len(all_results),
        "total_results_after_filter": len(filtered_results),
        # Add count of unique threads
        "total_unique_threads": len(thread_groups)
    }

    # Return results with filtered_info
    return SearchOutput(
        results=[DocumentOutput(
            content=r["content"], metadata=r["metadata"]) for r in formatted_results],
        answer=answer,
        filtered_info=filtered_info
    )

# Create a runnable chain for LangServe


class EmailSearchChain(Runnable):
    """Runnable chain for email search with encrypted responses"""

    def invoke(self, input_data: Any, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Invoke the chain.

        Args:
            input_data: Search input parameters
            config: Optional runtime configuration
            **kwargs: Additional keyword arguments

        Returns:
            Search results with an encrypted version in metadata
        """
        # DEBUG - Print detailed information about input
        print(f"INVOKE - INPUT TYPE: {type(input_data)}")
        print(f"INVOKE - INPUT CONTENT: {input_data}")
        print(f"INVOKE - CONFIG: {config}")
        print(f"INVOKE - KWARGS: {kwargs}")

        # Handle different input types
        if isinstance(input_data, dict):
            result = search_emails(input_data)
        elif isinstance(input_data, SearchInput):
            result = search_emails(input_data)
        else:
            # Try to convert to dict as a last resort
            try:
                input_dict = dict(input_data)
                result = search_emails(input_dict)
            except:
                raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Convert to dictionary to handle any serialization issues
        if isinstance(result, SearchOutput):
            result_dict = result.model_dump()
        else:
            result_dict = result

        # Ensure all values are serializable
        result_serializable = convert_numpy_types(result_dict)

        # Convert the response to JSON and encrypt it
        response_json = json.dumps(result_serializable)
        encrypted_response = fernet.encrypt(response_json.encode())

        # Add the encrypted data to the result's metadata
        # But keep the original structure to satisfy LangServe validation
        for doc in result_serializable["results"]:
            # Add the encryption metadata if not already present
            if "metadata" not in doc:
                doc["metadata"] = {}
            # Add a field to indicate the response is encrypted elsewhere
            doc["metadata"]["encryption_status"] = "Response encrypted in encrypted_data field"

        # Add the encrypted data as a top-level field
        result_serializable["encrypted_data"] = encrypted_response.decode()

        return result_serializable


@app.get("/test")
def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "API is working"}

# Add health check endpoint


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

# Add a direct endpoint for testing without LangServe wrapper


# Now update your endpoint to use this function:

def encrypt_for_frontend(data_string, key_string):
    """
    Encrypt data using AES in a WebCrypto-compatible format

    Args:
        data_string: JSON string to encrypt
        key_string: Base64-encoded encryption key

    Returns:
        Base64-encoded string with IV prepended to ciphertext
    """
    try:
        # Convert the data to bytes
        data_bytes = data_string.encode('utf-8')

        # Generate a proper AES-256 key from the provided key string
        # We'll use SHA-256 to derive a 32-byte key
        import hashlib

        # Use the raw string as input to SHA-256 to get a 32-byte key
        key_bytes = hashlib.sha256(key_string.encode('utf-8')).digest()
        print(f"Using key of length: {len(key_bytes)} bytes")

        # Create 16 bytes IV (initialization vector)
        iv = get_random_bytes(16)

        # Create the AES cipher in CBC mode
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv)

        # Pad the data to match block size
        padded_data = pad(data_bytes, AES.block_size)

        # Encrypt the data
        encrypted_bytes = cipher.encrypt(padded_data)

        # Combine IV and encrypted data (WebCrypto expects IV at the beginning)
        result = iv + encrypted_bytes

        # Return as base64 encoded string
        return base64.b64encode(result).decode('utf-8')
    except Exception as e:
        print(f"Encryption error: {e}")
        # Return unencrypted data for debugging
        return "ENCRYPTION_FAILED"


@app.post("/api/email-search/invoke-direct")
async def email_search_direct(request: Request):
    """
    Custom endpoint that directly encrypts responses in a way compatible with WebCrypto
    """
    # Parse the request body
    request_data = await request.json()

    # Extract the input data
    input_data = request_data.get("input", {})

    # Validate the input data
    try:
        search_input = SearchInput(**input_data)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Invalid input data: {str(e)}"},
            status_code=400
        )

    # Perform the search
    result = search_emails(search_input)

    # Convert to dict and ensure all values are serializable
    if isinstance(result, SearchOutput):
        result_dict = result.model_dump()
        result_serializable = convert_numpy_types(result_dict)
    else:
        result_serializable = convert_numpy_types(result)

    # Convert the response to JSON
    response_json = json.dumps(result_serializable)

    try:
        key_string = "KUwAEgxBHwPjcW-uLfD8yfCgWUzVn3IH1tlBEIV535Q="

        # Encrypt the data using our fixed function
        encrypted_response = encrypt_for_frontend(response_json, key_string)

        if encrypted_response == "ENCRYPTION_FAILED":
            # Provide unencrypted response if encryption fails
            return JSONResponse(content={
                "error": "Encryption failed, using debug data",
                "_debug_plain": result_serializable
            })
        else:
            # Return both encrypted and debug data
            return JSONResponse(content={
                "encrypted_data": encrypted_response
            })

    except Exception as e:
        print(f"Exception in route handler: {e}")
        # Fallback to returning unencrypted data
        return JSONResponse(content={
            "error": f"Failed to process response: {str(e)}",
            "_debug_plain": result_serializable
        })

# Custom LangServe route handler function


def encrypt_response_middleware(app, langserve_path: str):
    """
    Middleware to encrypt LangServe responses.
    This intercepts responses from LangServe routes and encrypts them.
    """
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse

    # Get the original route handler
    original_route = None
    for route in app.routes:
        if hasattr(route, "path") and route.path == f"{langserve_path}/invoke":
            original_route = route
            break

    if not original_route:
        raise ValueError(f"Route {langserve_path}/invoke not found")

    # Store the original endpoint function
    original_endpoint = original_route.endpoint

    # Create a new endpoint function that wraps the original
    async def encrypted_endpoint(request: Request):
        # Call the original endpoint
        original_response = await original_endpoint(request)

        # If it's a JSONResponse, extract the data and encrypt it
        if isinstance(original_response, JSONResponse):
            response_data = original_response.body
            response_dict = json.loads(response_data)

            # Encrypt the JSON data
            encrypted_data = fernet.encrypt(json.dumps(response_dict).encode())

            # Return a new response with the encrypted data
            return JSONResponse(
                content={"encrypted_data": encrypted_data.decode()},
                status_code=original_response.status_code,
                headers=dict(original_response.headers)
            )

        # If it's not a JSONResponse, return the original response
        return original_response

    # Replace the original endpoint with our wrapped version
    original_route.endpoint = encrypted_endpoint

    # Update the app.routes to use the new endpoint
    for i, route in enumerate(app.routes):
        if route.path == f"{langserve_path}/invoke":
            app.routes[i].endpoint = encrypted_endpoint
            break


# Initialize components
init_components()

# Create the search chain instance
email_search_chain = EmailSearchChain()

# Add routes using LangServe (keep only one instance of this)
add_routes(
    app,
    email_search_chain,
    path="/api/email-search",
    input_type=SearchInput,
    output_type=SearchOutput,
)

# Apply the encryption middleware AFTER adding routes
encrypt_response_middleware(app, "/api/email-search")

# Add database status endpoint


@app.get("/db-status")
def db_status():
    """Check the status of the email database"""
    chroma_exists = os.path.exists(chroma_dir) and os.listdir(chroma_dir)
    emails_exist = os.path.exists(email_dir) and os.listdir(email_dir)

    # Get some statistics if available
    stats = {}
    if chroma_exists and chroma_db:
        try:
            collection = chroma_db._collection
            results = collection.get()
            stats["document_count"] = len(
                results['ids']) if 'ids' in results else 0
        except Exception as e:
            stats["error"] = str(e)

    if emails_exist:
        try:
            email_count = len([f for f in os.listdir(
                email_dir) if f.endswith('.json')])
            stats["email_count"] = email_count
        except Exception as e:
            stats["email_error"] = str(e)

    return {
        "chroma_db_exists": chroma_exists,
        "emails_exist": emails_exist,
        "statistics": stats
    }


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))

    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=port)
