import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import sys
import tempfile
import shutil

# Add the current directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module to test
try:
    from chunk_emails import (
        Config, TextProcessor, DocumentCreator, EmailProcessor,
        MetaData, EmailData
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"\nWARNING: Import error: {e}")
    print("Make sure your code is saved in a file named 'chunk_emails.py'")
    IMPORTS_SUCCESSFUL = False

# Mock classes for testing when imports fail
if not IMPORTS_SUCCESSFUL:
    from pydantic import BaseModel

    class Config:
        DEFAULT_CHUNK_SIZE = 500
        DEFAULT_CHUNK_OVERLAP = 50
        DEFAULT_CHROMA_DIR = "./chromaDB"
        DEFAULT_INPUT_DIR = "gmail_emails_batch"
        DEFAULT_METADATA_FIELDS = ["subject", "sender", "date"]
        EMBEDDING_MODEL = "text-embedding-3-small"

    class MetaData(BaseModel):
        id: str
        thread_id: str
        snippet: str = None
        subject: str = None
        sender: str = None
        date: str = None

    class EmailData(BaseModel):
        metadata: MetaData
        body: str = None
        body_chunks: list = None

    class TextProcessor:
        @staticmethod
        def clean_chunk(chunk):
            return chunk.strip()

        @classmethod
        def chunk_text(cls, text, chunk_size=500, chunk_overlap=50,
                       use_semantic_chunking=False, openai_api_key=None):
            if not text:
                return []
            # Simple implementation for testing
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-chunk_overlap)]

    class DocumentCreator:
        @staticmethod
        def create_documents(email, inject_metadata_fields, store_metadata_fields):
            return []

    class EmailProcessor:
        def __init__(self, input_dir=None, chunk_size=500, chunk_overlap=50,
                     use_semantic_chunking=False, openai_api_key=None,
                     inject_metadata_fields=None, chroma_persist_dir=None):
            self.input_dir = input_dir or Config.DEFAULT_INPUT_DIR
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.use_semantic_chunking = use_semantic_chunking
            self.openai_api_key = openai_api_key
            self.inject_metadata_fields = inject_metadata_fields or Config.DEFAULT_METADATA_FIELDS
            self.chroma_persist_dir = chroma_persist_dir or Config.DEFAULT_CHROMA_DIR
            self.store_metadata_fields = [
                "id", "thread_id", "snippet", "subject", "sender", "date"]

        def load_email_file(self, filepath):
            return None

        def process_single_email(self, email_data):
            return []

        def process_all_emails(self):
            return []

        def store_in_chroma(self, documents):
            return None

        def process(self):
            return None


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
class TestConfig(unittest.TestCase):
    """Test the Config class."""

    def test_config_values(self):
        """Verify that Config has the expected values."""
        self.assertEqual(Config.DEFAULT_CHUNK_SIZE, 500)
        self.assertEqual(Config.DEFAULT_CHUNK_OVERLAP, 50)
        self.assertEqual(Config.DEFAULT_CHROMA_DIR, "./chromaDB")
        self.assertEqual(Config.DEFAULT_INPUT_DIR, "gmail_emails_batch")
        self.assertEqual(Config.DEFAULT_METADATA_FIELDS,
                         ["subject", "sender", "date"])
        self.assertEqual(Config.EMBEDDING_MODEL, "text-embedding-3-small")


class TestTextProcessor(unittest.TestCase):
    """Test the TextProcessor class."""

    def test_clean_chunk(self):
        """Test cleaning of chunks."""
        # Test removal of [link] in chunks
        self.assertEqual(TextProcessor.clean_chunk(
            "This is a [link] test"), "This is a  test")

        # Test stripping whitespace
        self.assertEqual(TextProcessor.clean_chunk("  test  "), "test")

        # Test handling empty strings
        self.assertEqual(TextProcessor.clean_chunk(""), "")

    @patch('chunk_emails.RecursiveCharacterTextSplitter.split_text')
    def test_chunk_text_basic(self, mock_split_text):
        """Test basic text chunking without semantic splitting."""
        # Mock the character splitter
        mock_split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]

        # Test with normal text
        result = TextProcessor.chunk_text("Test text", 100, 10, False, None)

        # Verify character splitter was called correctly
        mock_split_text.assert_called_once()

        # Verify results were processed correctly
        self.assertEqual(result, ["Chunk 1", "Chunk 2", "Chunk 3"])

    def test_chunk_text_empty_input(self):
        """Test chunking with empty input."""
        result = TextProcessor.chunk_text("", 100, 10, False, None)
        self.assertEqual(result, [])

        result = TextProcessor.chunk_text(None, 100, 10, False, None)
        self.assertEqual(result, [])

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    @patch('chunk_emails.SEMANTIC_CHUNKER_AVAILABLE', True)
    @patch('chunk_emails.SemanticChunker')
    @patch('chunk_emails.OpenAIEmbeddings')
    @patch('chunk_emails.RecursiveCharacterTextSplitter')
    def test_chunk_text_semantic(self, mock_char_splitter, mock_embeddings, mock_semantic_chunker):
        """Test semantic chunking path."""
        # Setup mocks
        mock_semantic_instance = mock_semantic_chunker.return_value
        mock_semantic_instance.split_text.return_value = [
            "Semantic Chunk 1", "Semantic Chunk 2"]

        mock_char_instance = mock_char_splitter.return_value
        mock_char_instance.split_text.side_effect = lambda x: [
            f"{x}_part1", f"{x}_part2"]

        # Call with semantic chunking enabled
        result = TextProcessor.chunk_text(
            "Test text", 100, 10, True, "fake-api-key"
        )

        # Verify embeddings were created
        mock_embeddings.assert_called_once()

        # Verify semantic chunker was used
        mock_semantic_instance.split_text.assert_called_once()

        # Verify character splitting was applied to each semantic chunk
        self.assertEqual(mock_char_instance.split_text.call_count, 2)

        # Verify results (should be a flat list of all character chunks from all semantic chunks)
        expected = [
            "Semantic Chunk 1_part1",
            "Semantic Chunk 1_part2",
            "Semantic Chunk 2_part1",
            "Semantic Chunk 2_part2"
        ]
        self.assertEqual(result, expected)


class TestDocumentCreator(unittest.TestCase):
    """Test the DocumentCreator class."""

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    def test_create_documents(self):
        """Test creation of documents from email data."""
        # Create test email data
        metadata = MetaData(
            id="test-id",
            thread_id="test-thread",
            subject="Test Subject",
            sender="test@example.com",
            date="2023-01-01"
        )

        email = EmailData(
            metadata=metadata,
            body="Test body",
            body_chunks=["Chunk 1", "Chunk 2"]
        )

        # Test with injecting subject and sender
        docs = DocumentCreator.create_documents(
            email,
            inject_metadata_fields=["subject", "sender"],
            store_metadata_fields=["id", "subject", "sender"]
        )

        # Verify the results
        # Two chunks should create two documents
        self.assertEqual(len(docs), 2)

        # Check first document
        self.assertTrue("Test Subject" in docs[0].page_content)
        self.assertTrue("test@example.com" in docs[0].page_content)
        self.assertTrue("Chunk 1" in docs[0].page_content)

        # Check metadata was stored correctly
        self.assertEqual(docs[0].metadata.get("id"), "test-id")
        self.assertEqual(docs[0].metadata.get("subject"), "Test Subject")
        self.assertEqual(docs[0].metadata.get("sender"), "test@example.com")

        # Check that thread_id is not in metadata (not in store_metadata_fields)
        self.assertNotIn("thread_id", docs[0].metadata)

    @unittest.skipIf(not IMPORTS_SUCCESSFUL, "Module import failed")
    def test_create_documents_no_injection(self):
        """Test document creation without metadata injection."""
        # Create test email data
        metadata = MetaData(
            id="test-id",
            thread_id="test-thread",
            subject="Test Subject",
            sender="test@example.com",
            date="2023-01-01"
        )

        email = EmailData(
            metadata=metadata,
            body="Test body",
            body_chunks=["Chunk 1", "Chunk 2"]
        )

        # Test with no metadata injection
        docs = DocumentCreator.create_documents(
            email,
            inject_metadata_fields=[],
            store_metadata_fields=["id", "subject", "sender"]
        )

        # Verify the results
        self.assertEqual(len(docs), 2)

        # Content should just be the chunk without metadata
        self.assertEqual(docs[0].page_content, "Chunk 1")
        self.assertEqual(docs[1].page_content, "Chunk 2")


class TestEmailProcessor(unittest.TestCase):
    """Test the EmailProcessor class."""

    def setUp(self):
        """Set up for tests."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "emails")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")

        # Create the input directory
        os.makedirs(self.input_dir, exist_ok=True)

        # Create a sample email file
        self.sample_email = {
            "id": "test-id",
            "threadId": "test-thread",
            "snippet": "Test snippet",
            "subject": "Test Subject",
            "from": "test@example.com",
            "date": "2023-01-01",
            "cleaned_body": "This is a test email body with some content to be chunked."
        }

        # Save the sample email
        with open(os.path.join(self.input_dir, "test-id.json"), "w") as f:
            json.dump(self.sample_email, f)

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of EmailProcessor."""
        # Test with minimal parameters (API key only)
        processor = EmailProcessor(openai_api_key="fake-api-key")

        # Verify default values
        self.assertEqual(processor.input_dir, Config.DEFAULT_INPUT_DIR)
        self.assertEqual(processor.chunk_size, Config.DEFAULT_CHUNK_SIZE)
        self.assertEqual(processor.chunk_overlap, Config.DEFAULT_CHUNK_OVERLAP)
        self.assertFalse(processor.use_semantic_chunking)
        self.assertEqual(processor.openai_api_key, "fake-api-key")
        self.assertEqual(processor.inject_metadata_fields,
                         Config.DEFAULT_METADATA_FIELDS)
        self.assertEqual(processor.chroma_persist_dir,
                         Config.DEFAULT_CHROMA_DIR)

    def test_init_with_parameters(self):
        """Test initialization with all parameters."""
        processor = EmailProcessor(
            input_dir="custom_dir",
            chunk_size=300,
            chunk_overlap=30,
            use_semantic_chunking=True,
            openai_api_key="fake-api-key",
            inject_metadata_fields=["custom_field"],
            chroma_persist_dir="custom_chroma"
        )

        # Verify custom values
        self.assertEqual(processor.input_dir, "custom_dir")
        self.assertEqual(processor.chunk_size, 300)
        self.assertEqual(processor.chunk_overlap, 30)
        self.assertTrue(processor.use_semantic_chunking)
        self.assertEqual(processor.openai_api_key, "fake-api-key")
        self.assertEqual(processor.inject_metadata_fields, ["custom_field"])
        self.assertEqual(processor.chroma_persist_dir, "custom_chroma")

    def test_init_missing_api_key(self):
        """Test initialization without API key."""
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            EmailProcessor()

    def test_load_email_file(self):
        """Test loading an email file."""
        processor = EmailProcessor(
            input_dir=self.input_dir,
            openai_api_key="fake-api-key"
        )

        # Load the sample email
        email_data = processor.load_email_file(
            os.path.join(self.input_dir, "test-id.json"))

        # Verify the email data
        self.assertIsNotNone(email_data)
        self.assertEqual(email_data.metadata.id, "test-id")
        self.assertEqual(email_data.metadata.thread_id, "test-thread")
        self.assertEqual(email_data.metadata.subject, "Test Subject")
        self.assertEqual(email_data.metadata.sender, "test@example.com")
        self.assertEqual(email_data.metadata.date, "2023-01-01")
        self.assertEqual(
            email_data.body, "This is a test email body with some content to be chunked.")

    def test_load_email_file_missing_body(self):
        """Test loading an email file with missing body."""
        # Create a sample email with no cleaned_body
        sample_email_no_body = {
            "id": "no-body-id",
            "threadId": "test-thread",
            "subject": "Test Subject",
            "from": "test@example.com",
            "date": "2023-01-01"
        }

        # Save the sample email
        with open(os.path.join(self.input_dir, "no-body-id.json"), "w") as f:
            json.dump(sample_email_no_body, f)

        processor = EmailProcessor(
            input_dir=self.input_dir,
            openai_api_key="fake-api-key"
        )

        # Load the sample email
        email_data = processor.load_email_file(
            os.path.join(self.input_dir, "no-body-id.json"))

        # Should return None for emails with no body
        self.assertIsNone(email_data)

    @patch('chunk_emails.TextProcessor.chunk_text')
    @patch('chunk_emails.DocumentCreator.create_documents')
    def test_process_single_email(self, mock_create_docs, mock_chunk_text):
        """Test processing a single email."""
        # Setup mocks
        mock_chunk_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_create_docs.return_value = ["Doc 1", "Doc 2"]

        # Create test email data
        metadata = MetaData(
            id="test-id",
            thread_id="test-thread",
            subject="Test Subject",
            sender="test@example.com",
            date="2023-01-01"
        )

        email = EmailData(
            metadata=metadata,
            body="Test body"
        )

        processor = EmailProcessor(openai_api_key="fake-api-key")

        # Process the email
        docs = processor.process_single_email(email)

        # Verify the chunking was called
        mock_chunk_text.assert_called_once_with(
            "Test body",
            processor.chunk_size,
            processor.chunk_overlap,
            processor.use_semantic_chunking,
            processor.openai_api_key
        )

        # Verify the documents were created
        mock_create_docs.assert_called_once()

        # Verify the chunks were added to the email data
        self.assertEqual(email.body_chunks, ["Chunk 1", "Chunk 2"])

        # Verify the documents were returned
        self.assertEqual(docs, ["Doc 1", "Doc 2"])

    @patch('chunk_emails.EmailProcessor.load_email_file')
    @patch('chunk_emails.EmailProcessor.process_single_email')
    @patch('os.makedirs')
    @patch('os.listdir')
    def test_process_all_emails(self, mock_listdir, mock_makedirs, mock_process_single, mock_load_file):
        """Test processing all emails."""
        # Setup mocks
        mock_listdir.return_value = [
            "email1.json", "email2.json", "not-json.txt"]

        # Create test email data
        email1 = EmailData(
            metadata=MetaData(
                id="email1",
                thread_id="thread1",
                subject="Subject 1",
                sender="sender1@example.com",
                date="2023-01-01"
            ),
            body="Body 1",
            body_chunks=["Chunk 1", "Chunk 2"]
        )

        email2 = EmailData(
            metadata=MetaData(
                id="email2",
                thread_id="thread2",
                subject="Subject 2",
                sender="sender2@example.com",
                date="2023-01-02"
            ),
            body="Body 2",
            body_chunks=["Chunk 3"]
        )

        # Mock the load_email_file method
        mock_load_file.side_effect = [email1, email2]

        # Mock the process_single_email method
        mock_process_single.side_effect = [["Doc 1", "Doc 2"], ["Doc 3"]]

        processor = EmailProcessor(
            input_dir=self.input_dir,
            openai_api_key="fake-api-key"
        )

        # Process all emails
        docs = processor.process_all_emails()

        # Verify the emails were loaded
        self.assertEqual(mock_load_file.call_count, 2)

        # Verify the emails were processed
        self.assertEqual(mock_process_single.call_count, 2)

        # Verify the documents were combined
        self.assertEqual(docs, ["Doc 1", "Doc 2", "Doc 3"])

    @patch('chunk_emails.Chroma.from_documents')
    def test_store_in_chroma(self, mock_from_documents):
        """Test storing documents in ChromaDB."""
        # Setup mock
        mock_chroma = MagicMock()
        mock_from_documents.return_value = mock_chroma

        processor = EmailProcessor(
            openai_api_key="fake-api-key",
            chroma_persist_dir=self.chroma_dir
        )

        # Store documents
        result = processor.store_in_chroma(["Doc 1", "Doc 2"])

        # Verify Chroma.from_documents was called correctly
        mock_from_documents.assert_called_once_with(
            ["Doc 1", "Doc 2"],
            embedding=processor.embeddings,
            persist_directory=self.chroma_dir
        )

        # Verify the result
        self.assertEqual(result, mock_chroma)

    @patch('chunk_emails.EmailProcessor.process_all_emails')
    @patch('chunk_emails.EmailProcessor.store_in_chroma')
    def test_process(self, mock_store, mock_process_all):
        """Test the full processing pipeline."""
        # Setup mocks
        mock_process_all.return_value = ["Doc 1", "Doc 2"]
        mock_chroma = MagicMock()
        mock_store.return_value = mock_chroma

        processor = EmailProcessor(openai_api_key="fake-api-key")

        # Run the process
        result = processor.process()

        # Verify process_all_emails was called
        mock_process_all.assert_called_once()

        # Verify store_in_chroma was called with the documents
        mock_store.assert_called_once_with(["Doc 1", "Doc 2"])

        # Verify the result
        self.assertEqual(result, mock_chroma)


if __name__ == "__main__":
    unittest.main()
