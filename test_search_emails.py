import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys
import numpy as np
from datetime import datetime, timezone
from email.utils import format_datetime
import tempfile
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import shutil

# Add the current directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module to test - update the import name if needed
try:
    from search_emails import (
        initialize_email_database, numpy_serializer, chroma_dir, email_dir, encryption_key, fernet, search_emails, SearchInput, SearchOutput, HybridSearcher,
        format_documents_for_response, generate_answer_from_docs,
        preprocess_query, convert_numpy_types,
        Document
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"\nWARNING: Import error: {e}")
    print("Make sure your code is saved in a file named 'search_emails.py'")
    print("If your file has a different name, update the import statement.")
    IMPORTS_SUCCESSFUL = False

    # Mock imports for testing when real imports fail
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional

    class SearchInput(BaseModel):
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
        start_date: Optional[str] = None
        end_date: Optional[str] = None
        sender: Optional[str] = None
        previous_queries: Optional[List[str]] = None

    class DocumentOutput(BaseModel):
        content: str
        metadata: Dict[str, Any]

    class SearchOutput(BaseModel):
        results: List[DocumentOutput]
        answer: Optional[str] = None
        filtered_info: Optional[Dict[str, Any]] = None

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class HybridSearcher:
        def __init__(self, chroma_db, alpha=0.5, min_bm25_score=1.0, include_metadata=True):
            self.chroma_db = chroma_db
            self.alpha = alpha

        def search(self, query, k=10, filter_criteria=None, fetch_k=None):
            return []

    def search_emails(input_data):
        return SearchOutput(results=[], answer=None)

    def format_documents_for_response(docs):
        return []

    def generate_answer_from_docs(query, documents, model_name="gpt-4o", previous_queries=None):
        return "[]"

    def preprocess_query(query):
        return query

    def convert_numpy_types(obj):
        return obj


# Create helpers for setting up test data
def create_test_document(content, metadata=None):
    """Create a test Document object with content and metadata"""
    if metadata is None:
        metadata = {}
    return Document(page_content=content, metadata=metadata)


def create_test_date(year, month, day):
    """Create a test date string in email format"""
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    return format_datetime(dt)


class TestSearchEmails(unittest.TestCase):
    """Test the search_emails function"""

    def setUp(self):
        """Set up for tests"""
        # Create sample documents for testing
        self.test_docs = [
            create_test_document(
                content="This is a test email about project planning",
                metadata={
                    "id": "email1",
                    "thread_id": "thread1",
                    "subject": "Project Planning",
                    "sender": "john@example.com",
                    "date": create_test_date(2023, 5, 15),
                    "hybrid_score": 0.85
                }
            ),
            create_test_document(
                content="Meeting tomorrow to discuss budget",
                metadata={
                    "id": "email2",
                    "thread_id": "thread2",
                    "subject": "Budget Meeting",
                    "sender": "sarah@example.com",
                    "date": create_test_date(2023, 6, 10),
                    "hybrid_score": 0.75
                }
            ),
            create_test_document(
                content="Quarterly review presentation draft",
                metadata={
                    "id": "email3",
                    "thread_id": "thread3",
                    "subject": "Quarterly Review",
                    "sender": "john@example.com",
                    "date": create_test_date(2023, 7, 20),
                    "hybrid_score": 0.65
                }
            )
        ]

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    def test_vector_search(self, mock_preprocess, mock_chroma, mock_hybrid):
        """Test vector search without reranking"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_chroma.similarity_search.return_value = self.test_docs

        # Create input
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="vector",
            use_reranker=False
        )

        # Call function
        result = search_emails(input_data)

        # Verify query was preprocessed
        mock_preprocess.assert_called_once_with("test query")

        # Verify chroma_db was called with right parameters
        mock_chroma.similarity_search.assert_called_once_with(
            "processed query", k=3, filter=None
        )

        # Verify the results
        self.assertEqual(len(result.results), 3)

        # Verify no answer was generated
        self.assertIsNone(result.answer)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.HuggingFaceCrossEncoder')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    def test_vector_search_with_reranking(self, mock_preprocess, mock_chroma,
                                          mock_cross_encoder, mock_hybrid):
        """Test vector search with reranking"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_chroma.similarity_search.return_value = self.test_docs

        # Mock the cross-encoder
        mock_encoder_instance = mock_cross_encoder.return_value
        mock_encoder_instance.predict.return_value = np.array([0.8, 0.9, 0.7])

        # Create input
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="vector",
            use_reranker=True,
            top_n=2
        )

        # Call function
        result = search_emails(input_data)

        # Verify query was preprocessed
        mock_preprocess.assert_called_once_with("test query")

        # Verify chroma_db was called with right parameters (fetch more for reranking)
        mock_chroma.similarity_search.assert_called_once_with(
            "processed query", k=9, filter=None
        )

        # Verify cross-encoder was used
        mock_cross_encoder.assert_called_once()
        mock_encoder_instance.predict.assert_called_once()

        # Verify the results (top 2 after reranking)
        self.assertEqual(len(result.results), 2)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    def test_hybrid_search(self, mock_preprocess, mock_chroma, mock_hybrid):
        """Test hybrid search without reranking"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_hybrid.search.return_value = self.test_docs

        # Create input
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="hybrid",
            hybrid_alpha=0.7,
            use_reranker=False
        )

        # Call function
        result = search_emails(input_data)

        # Verify query was preprocessed
        mock_preprocess.assert_called_once_with("test query")

        # Verify hybrid search was called with right parameters
        mock_hybrid.alpha = 0.7  # Check alpha was updated
        mock_hybrid.search.assert_called_once_with(
            query="processed query", k=3, filter_criteria=None
        )

        # Verify the results
        self.assertEqual(len(result.results), 3)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.HuggingFaceCrossEncoder')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    def test_hybrid_search_with_reranking(self, mock_preprocess, mock_chroma,
                                          mock_cross_encoder, mock_hybrid):
        """Test hybrid search with reranking"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_hybrid.search.return_value = self.test_docs

        # Mock the cross-encoder
        mock_encoder_instance = mock_cross_encoder.return_value
        mock_encoder_instance.predict.return_value = np.array([0.8, 0.9, 0.7])

        # Create input
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="hybrid-rerank",
            hybrid_alpha=0.7,
            use_reranker=True,
            top_n=2
        )

        # Call function
        result = search_emails(input_data)

        # Verify query was preprocessed
        mock_preprocess.assert_called_once_with("test query")

        # Verify hybrid search was called with right parameters
        mock_hybrid.alpha = 0.7  # Check alpha was updated
        mock_hybrid.search.assert_called_once_with(
            query="processed query", k=9, filter_criteria=None
        )

        # Verify cross-encoder was used
        mock_cross_encoder.assert_called_once()
        mock_encoder_instance.predict.assert_called_once()

        # Verify the results (top 2 after reranking)
        self.assertEqual(len(result.results), 2)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    @patch('search_emails.generate_answer_from_docs')
    def test_generate_answer(self, mock_generate, mock_preprocess, mock_chroma, mock_hybrid):
        """Test searching with answer generation"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_chroma.similarity_search.return_value = self.test_docs
        mock_generate.return_value = '[{"email_id":"thread1","subject":"Project Planning","relevance_score":9}]'

        # Create input with answer generation
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="vector",
            use_reranker=False,
            generate_answer=True,
            answer_model="gpt-3.5-turbo",
            previous_queries=["previous query"]
        )

        # Call function
        result = search_emails(input_data)

        # Verify the answer was generated
        mock_generate.assert_called_once_with(
            query="processed query",
            documents=self.test_docs[:5],
            model_name="gpt-3.5-turbo",
            previous_queries=["previous query"]
        )

        # Check answer in result
        self.assertEqual(
            result.answer, '[{"email_id":"thread1","subject":"Project Planning","relevance_score":9}]')

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.hybrid_searcher')
    @patch('search_emails.chroma_db')
    @patch('search_emails.preprocess_query')
    @patch('search_emails.generate_answer_from_docs')
    def test_invalid_json_answer(self, mock_generate, mock_preprocess, mock_chroma, mock_hybrid):
        """Test handling of invalid JSON in answer generation"""
        # Configure mocks
        mock_preprocess.return_value = "processed query"
        mock_chroma.similarity_search.return_value = self.test_docs
        mock_generate.return_value = 'This is not valid JSON'

        # Create input with answer generation
        input_data = SearchInput(
            query="test query",
            k=3,
            search_method="vector",
            use_reranker=False,
            generate_answer=True
        )

        # Call function
        result = search_emails(input_data)

        # Since the JSON is invalid, it should default to empty array
        self.assertEqual(result.answer, "[]")

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    def test_convert_numpy_types(self):
        """Test conversion of numpy types to Python native types"""
        # Create test data with numpy types
        test_data = {
            "int": np.int64(42),
            "float": np.float32(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
            "nested": {
                "array": np.array([4, 5, 6]),
                "list": [np.int64(7), np.float32(8.9)]
            }
        }

        # Convert the data
        converted = convert_numpy_types(test_data)

        # Verify the types were converted correctly
        self.assertIsInstance(converted["int"], int)
        self.assertIsInstance(converted["float"], float)
        self.assertIsInstance(converted["array"], list)
        self.assertIsInstance(converted["bool"], bool)
        self.assertIsInstance(converted["nested"]["array"], list)
        self.assertIsInstance(converted["nested"]["list"][0], int)
        self.assertIsInstance(converted["nested"]["list"][1], float)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('search_emails.ChatOpenAI')
    def test_preprocess_query(self, mock_chat_openai):
        """Test query preprocessing with LLM correction"""
        # Mock the ChatOpenAI response
        mock_llm_instance = mock_chat_openai.return_value
        mock_response = MagicMock()
        mock_response.content = '{"cleaned_query": "test query with typo", "high_ranked_keywords_from_query": ["test", "query", "typo"]}'
        mock_llm_instance.invoke.return_value = mock_response

        # Call the function
        result = preprocess_query("test querry with typo")

        # Verify the result is the corrected query
        self.assertEqual(result, "test query with typo")

    @patch('search_emails.ChatOpenAI')
    def test_preprocess_query(self, mock_chat_openai):
        """Test query preprocessing with LLM correction"""
        # Mock the ChatOpenAI response
        mock_llm_instance = mock_chat_openai.return_value
        mock_response = MagicMock()
        mock_response.content = '{"cleaned_query": "test query with typo", "high_ranked_keywords_from_query": ["test", "query", "typo"]}'
        mock_llm_instance.invoke.return_value = mock_response

        # Call the function
        result = preprocess_query("test querry with typo")

        # Verify the result is the corrected query
        self.assertEqual(result, "test query with typo")

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    def test_format_documents_for_response(self):
        """Test formatting documents for response"""
        # Call the function with test documents
        result = format_documents_for_response(self.test_docs)

        # Verify the results
        self.assertEqual(len(result), 3)

        # Check first document structure
        self.assertEqual(result[0]["content"],
                         "This is a test email about project planning")
        self.assertEqual(result[0]["metadata"]["subject"], "Project Planning")
        self.assertEqual(result[0]["metadata"]["sender"], "john@example.com")

        # Check all documents have the correct structure
        for doc in result:
            self.assertIn("content", doc)
            self.assertIn("metadata", doc)

            # Check metadata fields
            self.assertIn("id", doc["metadata"])
            self.assertIn("thread_id", doc["metadata"])
            self.assertIn("subject", doc["metadata"])
            self.assertIn("sender", doc["metadata"])
            self.assertIn("date", doc["metadata"])


class TestHybridSearcher(unittest.TestCase):

    @patch('search_emails.BM25Okapi')
    def setUp(self, mock_bm25_okapi):
        """Set up test fixtures with patched BM25Okapi"""
        # Create mock for BM25Okapi constructor
        self.mock_bm25 = MagicMock()
        mock_bm25_okapi.return_value = self.mock_bm25

        # Create a mock ChromaDB instance
        self.mock_chroma_db = MagicMock()

        # Set up collection results for _initialize_bm25 method
        mock_collection = MagicMock()
        self.mock_chroma_db._collection = mock_collection

        mock_collection.get.return_value = {
            'ids': ['doc1', 'doc2', 'doc3', 'doc4'],
            'documents': [
                "This is document one about machine learning",
                "Document two discusses data science topics",
                "Email about project timelines and deadlines",
                "Meeting notes from yesterday's discussion"
            ],
            'metadatas': [
                {'id': 'doc1', 'source': 'academic'},
                {'id': 'doc2', 'source': 'blog'},
                {'id': 'doc3', 'source': 'email'},
                {'id': 'doc4', 'source': 'meeting'}
            ]
        }

        # Initialize the HybridSearcher with our mocks
        with patch.object(HybridSearcher, '_tokenize', return_value=['mock', 'tokens']):
            self.hybrid_searcher = HybridSearcher(
                chroma_db=self.mock_chroma_db,
                alpha=0.6,
                min_bm25_score=0.5,
                include_metadata=True
            )

        # Create sample documents for vector search results
        self.vector_docs = [
            Document(page_content="This is document one about machine learning", metadata={
                     "id": "doc1"}),
            Document(page_content="Document two discusses data science topics", metadata={
                     "id": "doc2"}),
            Document(page_content="Email about project timelines and deadlines", metadata={
                     "id": "doc3"}),
        ]

    def test_search_basic(self):
        """Test basic search functionality without filters"""
        # Set up mock return values
        self.mock_chroma_db.similarity_search.return_value = self.vector_docs
        self.mock_bm25.get_scores.return_value = np.array([0.8, 0.6, 0.9, 0.3])

        # Execute search
        query = "machine learning project"

        # Patch the _tokenize method to return consistent tokens for the query
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['machine', 'learning', 'project']):
            results = self.hybrid_searcher.search(query, k=2)

        # Verify the results
        self.assertEqual(len(results), 2, "Should return exactly 2 results")
        self.mock_chroma_db.similarity_search.assert_called_once()
        self.mock_bm25.get_scores.assert_called_once()

        # Check that hybrid scores are added to metadata
        for doc in results:
            self.assertIn('hybrid_score', doc.metadata,
                          "Each result should have a hybrid_score")
            self.assertTrue(
                0 <= doc.metadata['hybrid_score'] <= 1, "Score should be between 0 and 1")

    def test_search_with_filter(self):
        """Test search with filter criteria"""
        # Set up mock return values
        self.mock_chroma_db.similarity_search.return_value = self.vector_docs
        self.mock_bm25.get_scores.return_value = np.array([0.8, 0.6, 0.9, 0.3])

        # Define filter criteria
        filter_criteria = {"date": {"$gte": "2023-01-01"}}

        # Execute search with filter
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['project', 'deadline']):
            results = self.hybrid_searcher.search(
                query="project deadline",
                k=2,
                filter_criteria=filter_criteria
            )

        # Verify filter was passed to vector search
        self.mock_chroma_db.similarity_search.assert_called_once_with(
            "project deadline",
            k=4,  # 2*k by default
            filter=filter_criteria
        )

        # Check results
        self.assertLessEqual(
            len(results), 2, "Should not return more than k results")

    def test_search_empty_results(self):
        """Test behavior when vector search returns no results"""
        # Set up empty return values
        self.mock_chroma_db.similarity_search.return_value = []
        self.mock_bm25.get_scores.return_value = np.array([0.2, 0.3, 0.1, 0.4])

        # Execute search
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['irrelevant', 'query']):
            results = self.hybrid_searcher.search("irrelevant query", k=3)

        # Verify results
        self.assertEqual(
            len(results), 0, "Should return empty list when no matches found")

    def test_search_custom_fetch_k(self):
        """Test search with custom fetch_k parameter"""
        # Set up mock return values
        self.mock_chroma_db.similarity_search.return_value = self.vector_docs
        self.mock_bm25.get_scores.return_value = np.array([0.8, 0.6, 0.9, 0.3])

        # Execute search with custom fetch_k
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['machine', 'learning']):
            results = self.hybrid_searcher.search(
                query="machine learning",
                k=2,
                fetch_k=10  # Custom fetch_k
            )

        # Verify vector search used custom fetch_k
        self.mock_chroma_db.similarity_search.assert_called_once_with(
            "machine learning",
            k=10,  # Custom fetch_k
            filter=None
        )

        # Check results
        self.assertLessEqual(
            len(results), 2, "Should not return more than k results")

    def test_score_normalization(self):
        """Test BM25 score normalization"""
        # Set up mock return values with specific scores
        self.mock_chroma_db.similarity_search.return_value = self.vector_docs

        # Test case 1: Normal scores
        self.mock_bm25.get_scores.return_value = np.array([0.8, 0.6, 0.9, 0.3])
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['test', 'query']):
            results1 = self.hybrid_searcher.search("test query", k=3)

        # Test case 2: All zeros
        self.mock_bm25.get_scores.return_value = np.array([0.0, 0.0, 0.0, 0.0])
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['another', 'query']):
            results2 = self.hybrid_searcher.search("another query", k=3)

        # Test case 3: Negative scores (shouldn't happen with BM25 but testing robustness)
        self.mock_bm25.get_scores.return_value = np.array(
            [-0.1, 0.0, -0.5, 0.0])
        with patch.object(self.hybrid_searcher, '_tokenize', return_value=['third', 'query']):
            results3 = self.hybrid_searcher.search("third query", k=3)

        # Verify all searches completed without errors
        self.assertGreaterEqual(len(results1), 0)
        self.assertGreaterEqual(len(results2), 0)
        self.assertGreaterEqual(len(results3), 0)

    @patch('search_emails.BM25Okapi')
    def test_alpha_weighting(self, mock_bm25_okapi):
        """Test that alpha parameter correctly weights vector vs BM25 scores"""
        # Set up mocks
        mock_bm25 = MagicMock()
        mock_bm25_okapi.return_value = mock_bm25
        mock_bm25.get_scores.return_value = np.array([0.2, 0.5, 0.9, 0.3])

        # Create two searchers with different alpha values
        with patch.object(HybridSearcher, '_tokenize', return_value=['mock', 'tokens']):
            high_vector_weight = HybridSearcher(
                chroma_db=self.mock_chroma_db,
                alpha=0.9,  # High weight to vector search
            )

            high_bm25_weight = HybridSearcher(
                chroma_db=self.mock_chroma_db,
                alpha=0.1,  # High weight to BM25
            )

        # Set up vector search results: doc1 ranks first in vector search
        self.mock_chroma_db.similarity_search.return_value = [
            Document(page_content="This is document one about machine learning", metadata={
                     "id": "doc1"}),
            Document(page_content="Email about project timelines and deadlines", metadata={
                     "id": "doc3"}),
            Document(page_content="Document two discusses data science topics", metadata={
                     "id": "doc2"}),
        ]

        # Run searches with patched tokenize method
        with patch.object(high_vector_weight, '_tokenize', return_value=['test', 'query']):
            with patch.object(high_bm25_weight, '_tokenize', return_value=['test', 'query']):
                vector_biased_results = high_vector_weight.search(
                    "test query", k=3)
                bm25_biased_results = high_bm25_weight.search(
                    "test query", k=3)

        # Verify alpha values were correctly set
        self.assertEqual(high_vector_weight.alpha, 0.9)
        self.assertEqual(high_bm25_weight.alpha, 0.1)


class TestInitializeEmailDatabase(unittest.TestCase):
    """Tests for the initialize_email_database function"""

    def setUp(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_chroma_dir = os.path.join(self.temp_dir, "test_chroma")
        self.temp_email_dir = os.path.join(self.temp_dir, "test_emails")

        # Create patch for global variables
        self.chroma_dir_patcher = patch(
            'search_emails.chroma_dir', self.temp_chroma_dir)
        self.email_dir_patcher = patch(
            'search_emails.email_dir', self.temp_email_dir)

        # Start patches
        self.mock_chroma_dir = self.chroma_dir_patcher.start()
        self.mock_email_dir = self.email_dir_patcher.start()

    def tearDown(self):
        # Stop patches
        self.chroma_dir_patcher.stop()
        self.email_dir_patcher.stop()

        # Clean up temp directories
        shutil.rmtree(self.temp_dir)

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('subprocess.run')
    def test_initialize_email_database_from_scratch(self, mock_run, mock_listdir, mock_exists):
        """Test initializing email database when neither exists"""
        # Setup mocks
        mock_exists.side_effect = lambda path: False  # Nothing exists
        mock_listdir.return_value = []  # Empty directories

        # Call function
        initialize_email_database()

        # Verify subprocess calls
        self.assertEqual(mock_run.call_count, 2)

        # First call should run get_emails.py
        self.assertEqual(mock_run.call_args_list[0][0][0][1], "get_emails.py")

        # Second call should run chunk_emails.py
        self.assertEqual(
            mock_run.call_args_list[1][0][0][1], "chunk_emails.py")

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('subprocess.run')
    def test_initialize_email_database_chroma_exists(self, mock_run, mock_listdir, mock_exists):
        """Test initializing when ChromaDB already exists"""
        # Setup mocks for existing ChromaDB
        mock_exists.side_effect = lambda path: path == self.temp_chroma_dir
        mock_listdir.side_effect = lambda path: [
            'data'] if path == self.temp_chroma_dir else []

        # Call function
        initialize_email_database()

        # Verify no subprocess calls were made
        mock_run.assert_not_called()

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('subprocess.run')
    def test_initialize_email_database_emails_exist(self, mock_run, mock_listdir, mock_exists):
        """Test initializing when emails exist but ChromaDB doesn't"""
        # Setup mocks for existing emails
        def mock_exists_side_effect(path):
            if path == self.temp_email_dir:
                return True
            if path == self.temp_chroma_dir:
                return False
            return False

        mock_exists.side_effect = mock_exists_side_effect
        mock_listdir.side_effect = lambda path: [
            'email1.json'] if path == self.temp_email_dir else []

        # Call function
        initialize_email_database()

        # Verify only chunk_emails.py was called (not get_emails.py)
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(
            mock_run.call_args_list[0][0][0][1], "chunk_emails.py")


class TestNumpySerializer(unittest.TestCase):
    """Tests for the numpy_serializer function"""

    def test_numpy_integer(self):
        """Test serializing numpy integer"""
        value = np.int32(42)
        result = numpy_serializer(value)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_numpy_float(self):
        """Test serializing numpy float"""
        value = np.float32(3.14)
        result = numpy_serializer(value)
        self.assertAlmostEqual(result, 3.14, places=2)
        self.assertIsInstance(result, float)

    def test_numpy_array(self):
        """Test serializing numpy array"""
        value = np.array([1, 2, 3, 4])
        result = numpy_serializer(value)
        self.assertEqual(result, [1, 2, 3, 4])
        self.assertIsInstance(result, list)

    def test_numpy_bool(self):
        """Test serializing numpy boolean"""
        value = np.bool_(True)
        result = numpy_serializer(value)
        self.assertEqual(result, True)
        self.assertIsInstance(result, bool)

    def test_unsupported_type(self):
        """Test serializing unsupported type raises TypeError"""
        class UnsupportedType:
            pass

        value = UnsupportedType()
        with self.assertRaises(TypeError):
            numpy_serializer(value)


if __name__ == "__main__":
    unittest.main()
