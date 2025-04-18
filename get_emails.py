import os
import json
import time
import datetime
from email.utils import parsedate
import base64
import re
from typing import List, Dict, Any, Optional, Callable

from pydantic import BaseModel
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ---- Pydantic Models ----

class MetaData(BaseModel):
    id: str
    thread_id: str
    snippet: Optional[str] = None
    subject: Optional[str] = None
    sender: Optional[str] = None
    date: Optional[str] = None


class EmailData(BaseModel):
    metadata: MetaData
    body: Optional[str] = None
    body_chunks: Optional[list] = None


# ---- Configuration ----

class Config:
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    MAX_RETRIES = 5
    MAX_BATCH_SIZE = 35  # Reduced batch size to avoid rate limits
    QUOTA_RESET_TIME = 0.5  # Time in seconds to pause when quota is exhausted
    REQUEST_INTERVAL = 0.2  # Time between batch requests
    OUTPUT_DIR = "gmail_emails_batch"
    INDEX_FILE = "gmail_index_batch.json"
    MAX_EMAILS = 1000
    CREDENTIALS_FILE = "credentials.json"
    TOKEN_FILE = "token.json"


# ---- Logger ----

class Logger:
    @staticmethod
    def log(message: str) -> None:
        """Print a message with a timestamp."""
        current_time = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{current_time}] {message}")


# ---- Email Cleaning ----

class EmailCleaner:
    @staticmethod
    def remove_redundant_content(email_body: str) -> str:
        """
        Remove redundant content (quoted or forwarded messages) from an email body.
        """
        # Remove quoted content (e.g., "On [date], [sender] wrote:")
        email_body = re.sub(r'On .*? wrote:', '', email_body, flags=re.DOTALL)

        # Remove forwarded message markers
        email_body = re.sub(r'Forwarded message.*?:', '',
                            email_body, flags=re.DOTALL)

        # Remove lines starting with ">"
        email_body = re.sub(r'^>.*$', '', email_body, flags=re.MULTILINE)

        # Remove excessive newlines
        email_body = re.sub(r'\n{3,}', '\n\n', email_body)

        return email_body.strip()

    @staticmethod
    def preserve_newlines(text: str) -> str:
        """Preserve important newlines while normalizing whitespace."""
        if not text:
            return ""

        # Replace single newlines with a special marker
        text = re.sub(r'(?<!\n)\n(?!\n)', ' NEWLINE_MARKER ', text)

        # Normalize other whitespace
        text = re.sub(r'\s+', ' ', text)

        # Restore newlines
        text = text.replace('NEWLINE_MARKER', '\n')

        return text.strip()

    @staticmethod
    def clean_separators(text: str) -> str:
        """Remove various separator characters."""
        # Remove multiple consecutive characters used as separators
        text = re.sub(r'[-=~_*]+', '', text)  # For - , = , ~ , _ , *
        return text

    @staticmethod
    def clean_email_body(text: str, remove_html: bool = True) -> str:
        """Clean up an email body by removing irrelevant content."""
        if not text:
            return ""

        # Save original newlines structure
        text = EmailCleaner.preserve_newlines(text)

        # Remove all URLs (e.g., links) but preserve surrounding text
        text = re.sub(r'http[s]?://\S+', '[link]', text)

        # Remove image placeholders
        text = re.sub(r'[\[]?.*\[.*\]', '', text)

        # Remove signature lines
        text = re.sub(r'\n+Sent from my [^\n]+$', '', text)

        # Remove closing phrases near the end of the email
        closing_phrases = [
            r'Best regards,?\s*[A-Z][a-zA-Z]*\s*$',
            r'Sincerely,?\s*[A-Z][a-zA-Z]*\s*$',
            r'Cheers,?\s*[A-Z][a-zA-Z]*\s*$',
            r'Regards,?\s*[A-Z][a-zA-Z]*\s*$',
            r'Thanks,?\s*[A-Z][a-zA-Z]*\s*$'
        ]

        # Only apply these if they appear in the last 20% of the email
        email_length = len(text)
        for phrase in closing_phrases:
            match = re.search(phrase, text)
            if match and match.start() > email_length * 0.8:
                text = text[:match.start()].strip()

        # Remove promotional content
        promo_patterns = [
            r'\bTip of the week:.*$',
            r'\bSponsored:.*$',
            r'\bTo unsubscribe from.*$',
            r'\bThis email was sent to [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            r'\bIf you no longer wish to receive.*$'
        ]

        for pattern in promo_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = EmailCleaner.clean_separators(text)

        # Final cleanup of excessive whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


# ---- Gmail API Handler ----

class GmailHandler:
    def __init__(self):
        self.service = None
        self.failed_message_ids = set()

    def authenticate(self) -> None:
        """Authenticate and set up Gmail API service."""
        Logger.log("Starting authentication process...")
        start_time = time.time()

        creds = None
        if os.path.exists(Config.TOKEN_FILE):
            Logger.log(
                "Found existing token file. Attempting to use saved credentials...")
            creds = Credentials.from_authorized_user_file(
                Config.TOKEN_FILE, Config.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                Logger.log("Credentials expired. Refreshing token...")
                creds.refresh(Request())
            else:
                Logger.log(
                    "No valid credentials found. Starting OAuth flow...")
                os.environ['BROWSER'] = 'none'
                flow = InstalledAppFlow.from_client_secrets_file(
                    Config.CREDENTIALS_FILE,
                    Config.SCOPES,
                    redirect_uri='http://localhost'
                )

                Logger.log(
                    "Starting local server for authentication. Please authenticate in your browser...")
                creds = flow.run_local_server(port=0)
                Logger.log("Successfully authenticated!")

            # Save the credentials for the next run
            Logger.log(f"Saving credentials to {Config.TOKEN_FILE}...")
            with open(Config.TOKEN_FILE, "w") as token:
                token.write(creds.to_json())

        Logger.log(
            f"Authentication completed in {time.time() - start_time:.2f} seconds")
        self.service = build("gmail", "v1", credentials=creds)

    def api_request_with_backoff(self, request_func: Callable, operation_name: str = "API request"):
        """Execute an API request with exponential backoff for rate limiting."""
        start_time = time.time()
        for n in range(Config.MAX_RETRIES):
            try:
                result = request_func()
                elapsed = time.time() - start_time
                if elapsed > 1.0:  # Only log slow requests
                    Logger.log(
                        f"{operation_name} completed in {elapsed:.2f} seconds")
                return result
            except HttpError as e:
                if hasattr(e, 'status_code') and e.status_code in [429, 500, 503]:
                    # Exponential backoff starting at 4 seconds
                    wait_time = 2 ** (n + 2)
                    Logger.log(
                        f"Rate limit hit during {operation_name}. Backing off for {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    Logger.log(f"HTTP error during {operation_name}: {e}")
                    raise
            except Exception as e:
                Logger.log(f"Unexpected error during {operation_name}: {e}")
                raise

        raise Exception(
            f"Failed {operation_name} after {Config.MAX_RETRIES} retries")

    def get_email_body(self, payload: Dict[str, Any]) -> str:
        """Extract the email body from the message payload."""
        start_time = time.time()
        body = None
        body_type = "unknown"

        try:
            if 'parts' in payload['payload']:
                for index, part in enumerate(payload['payload']['parts']):
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8')
            elif 'body' in payload['payload'] and 'data' in payload['payload']['body']:
                body = base64.urlsafe_b64decode(
                    payload['payload']['body']['data']).decode('utf-8')

        except Exception as e:
            Logger.log(f"Error extracting email body: {e}")
            Logger.log(f"Body extraction failed for {payload}")
            body = "Error extracting body: " + str(e)
            body_type = "error"

        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log if body extraction took some time
            Logger.log(
                f"Body extraction ({body_type}) took {elapsed:.2f} seconds")

        return body or "No body found"

    def get_message_ids(self) -> List[Dict[str, str]]:
        """Get message IDs from Gmail."""
        Logger.log("Getting email IDs (this may take a while)...")
        all_message_ids = []
        next_page_token = None
        remaining = Config.MAX_EMAILS
        page_count = 0

        while remaining > 0:
            page_count += 1
            # Gmail API allows up to 500 IDs per page
            batch_size = min(remaining, 500)
            page_start = time.time()

            Logger.log(
                f"Requesting page {page_count} of message IDs (max {batch_size})...")
            results = self.api_request_with_backoff(
                lambda: self.service.users().messages().list(
                    userId="me",
                    maxResults=batch_size,
                    pageToken=next_page_token
                ).execute(),
                f"list messages page {page_count}"
            )

            batch_message_ids = results.get("messages", [])
            all_message_ids.extend(batch_message_ids)

            next_page_token = results.get("nextPageToken")
            remaining = Config.MAX_EMAILS - len(all_message_ids)

            page_time = time.time() - page_start
            Logger.log(
                f"Page {page_count}: Retrieved {len(batch_message_ids)} message IDs in {page_time:.2f} seconds")
            Logger.log(
                f"Total progress: {len(all_message_ids)}/{Config.MAX_EMAILS} message IDs")

            if not next_page_token or not batch_message_ids:
                Logger.log(
                    "No more messages available or reached the end of the list")
                break

        # Ensure we have at most MAX_EMAILS messages
        all_message_ids = all_message_ids[:Config.MAX_EMAILS]
        Logger.log(f"Retrieved {len(all_message_ids)} email IDs in total")
        return all_message_ids


# ---- Email Processor ----

class EmailProcessor:
    def __init__(self, gmail_handler: GmailHandler):
        self.gmail_handler = gmail_handler
        self.index_file = {"emails": []}
        self.processed_count = 0
        self.total_count = 0

    def process_email_response(self, request_id: str, response: Dict[str, Any], exception: Exception, batch_data: Dict[str, Any]) -> None:
        """Process a single email response from a batch request."""
        if exception is not None:
            Logger.log(f"Error processing message {request_id}: {exception}")
            # Add to failed IDs for retry
            self.gmail_handler.failed_message_ids.add(request_id)
            return

        msg_id = request_id  # Use request_id as message ID

        try:
            # Extract headers
            headers = {header['name']: header['value']
                       for header in response['payload']['headers']}

            # Extract message body
            process_start = time.time()
            message_body = self.gmail_handler.get_email_body(
                response)

            # Clean the email body
            cleaned_body = EmailCleaner.clean_email_body(
                message_body, remove_html=True)

            body_length = len(message_body) if message_body else 0
            cleaned_length = len(cleaned_body) if cleaned_body else 0
            process_time = time.time() - process_start

            Logger.log(
                f"Email {msg_id} - Original body: {body_length} chars, Cleaned body: {cleaned_length} chars")

            # # Create the cleaned email Pydantic model
            # email_metadata = MetaData(
            #     id=response['id'],
            #     thread_id=response.get('threadId', ''),
            #     snippet=response.get('snippet', ''),
            #     subject=headers.get('Subject', 'No Subject'),
            #     sender=headers.get('From', 'Unknown'),
            #     date=headers.get('Date', 'Unknown')
            # )

            # # Create email object with both original and cleaned bodies
            # email_obj = {
            #     'id': response['id'],
            #     'threadId': response.get('threadId', ''),
            #     'labelIds': response.get('labelIds', []),
            #     'snippet': response.get('snippet', ''),
            #     'subject': headers.get('Subject', 'No Subject'),
            #     'from': headers.get('From', 'Unknown'),
            #     'to': headers.get('To', 'Unknown'),
            #     'date': headers.get('Date', 'Unknown'),
            #     'original_body': message_body,
            #     'cleaned_body': cleaned_body,
            # }

            # # Save email to file
            # with open(f"{Config.OUTPUT_DIR}/{msg_id}.json", "w", encoding='utf-8') as email_file:
            #     json.dump(email_obj, email_file, ensure_ascii=False)

            try:
                if not response.get('threadId', ''):
                    raise ValueError(
                        "The thread ID is empty. Cannot generate a valid file path.")

                existing_filepath = os.path.join(
                    Config.OUTPUT_DIR, f"{response.get('threadId', '')}.json")
            except ValueError as e:
                Logger.log(f"Error: {e}")
                return

            email_obj = {
                'id': response['id'],
                'threadId': response.get('threadId', ''),
                'labelIds': response.get('labelIds', []),
                'snippet': response.get('snippet', ''),
                'subject': headers.get('Subject', 'No Subject'),
                'from': headers.get('From', 'Unknown'),
                'to': headers.get('To', 'Unknown'),
                'date': headers.get('Date', 'Unknown'),
                'original_body': message_body,
                'cleaned_body': cleaned_body,
            }

            # Merge with existing email content or create a new file
            email_obj = merge_email_content(existing_filepath, email_obj)

            # Save email to file (if not already saved in merge_email_content)
            with open(existing_filepath, "w", encoding='utf-8') as email_file:
                json.dump(email_obj, email_file, ensure_ascii=False)

            # Add metadata to index
            self.index_file['emails'].append({
                'id': msg_id,
                'subject': email_obj['subject'],
                'from': email_obj['from'],
                'date': email_obj['date'],
                'cleaned_body_length': cleaned_length,
                'original_body_length': body_length,
            })

            # Update counts
            self.processed_count += 1

            # Periodic status update
            if self.processed_count % 10 == 0 or self.processed_count == self.total_count:
                Logger.log(
                    f"Processed {self.processed_count}/{self.total_count} emails")

        except Exception as e:
            Logger.log(f"Error processing message {msg_id} data: {e}")
            # Add to failed IDs for retry
            self.gmail_handler.failed_message_ids.add(msg_id)

    def process_message_individually(self, msg_id: str) -> bool:
        """Process a single message individually (for retry cases)."""
        Logger.log(f"Retrying message {msg_id} individually...")

        try:
            # Request full message content with backoff
            msg = self.gmail_handler.api_request_with_backoff(
                lambda: self.gmail_handler.service.users().messages().get(
                    userId="me", id=msg_id, format='full').execute(),
                f"individual retry for {msg_id}"
            )

            # Extract headers
            headers = {header['name']: header['value']
                       for header in msg['payload']['headers']}

            # Extract message body
            message_body = self.gmail_handler.get_email_body(msg)

            # Clean the email body
            cleaned_body = EmailCleaner.clean_email_body(
                message_body, remove_html=True)

            # # Create email object with both original and cleaned bodies
            # email_obj = {
            #     'id': msg['id'],
            #     'threadId': msg.get('threadId', ''),
            #     'labelIds': msg.get('labelIds', []),
            #     'snippet': msg.get('snippet', ''),
            #     'subject': headers.get('Subject', 'No Subject'),
            #     'from': headers.get('From', 'Unknown'),
            #     'to': headers.get('To', 'Unknown'),
            #     'date': headers.get('Date', 'Unknown'),
            #     'original_body': message_body,
            #     'cleaned_body': cleaned_body,
            # }

            # # Save email to file
            # with open(f"{Config.OUTPUT_DIR}/{msg_id}.json", "w", encoding='utf-8') as email_file:
            #     json.dump(email_obj, email_file, ensure_ascii=False)

            try:
                if not msg.get('threadId', ''):
                    raise ValueError(
                        "The thread ID is empty. Cannot generate a valid file path.")

                existing_filepath = os.path.join(
                    Config.OUTPUT_DIR, f"{msg.get('threadId', '')}.json")
            except ValueError as e:
                Logger.log(f"Error: {e}")
                return

            email_obj = {
                'id': msg['id'],
                'threadId': msg.get('threadId', ''),
                'labelIds': msg.get('labelIds', []),
                'snippet': msg.get('snippet', ''),
                'subject': headers.get('Subject', 'No Subject'),
                'from': headers.get('From', 'Unknown'),
                'to': headers.get('To', 'Unknown'),
                'date': headers.get('Date', 'Unknown'),
                'original_body': message_body,
                'cleaned_body': cleaned_body,
            }

            # Merge with existing email content or create a new file
            email_obj = merge_email_content(existing_filepath, email_obj)

            # Save email to file (if not already saved in merge_email_content)
            with open(existing_filepath, "w", encoding='utf-8') as email_file:
                json.dump(email_obj, email_file, ensure_ascii=False)

            # Add metadata to index
            self.index_file['emails'].append({
                'id': msg_id,
                'subject': email_obj['subject'],
                'from': email_obj['from'],
                'date': email_obj['date'],
                'cleaned_body_length': len(cleaned_body) if cleaned_body else 0,
                'original_body_length': len(message_body) if message_body else 0,
            })

            # Update counts and return success
            self.processed_count += 1
            Logger.log(f"Successfully processed individual message {msg_id}")
            return True

        except Exception as e:
            Logger.log(f"Failed to process message {msg_id} individually: {e}")
            return False

    def process_in_batches(self, message_ids: List[Dict[str, str]]) -> None:
        """Process emails in batches with rate limiting."""
        self.total_count = len(message_ids)
        batch_count = 0
        total_batches = (self.total_count +
                         Config.MAX_BATCH_SIZE - 1) // Config.MAX_BATCH_SIZE

        for i in range(0, self.total_count, Config.MAX_BATCH_SIZE):
            batch_count += 1
            batch_start = time.time()

            # Add delay between batches for rate limiting
            if batch_count > 1:
                Logger.log(
                    f"Waiting {Config.REQUEST_INTERVAL} seconds before next batch...")
                time.sleep(Config.REQUEST_INTERVAL)

            current_batch = message_ids[i:i+Config.MAX_BATCH_SIZE]

            Logger.log(
                f"Starting batch {batch_count}/{total_batches} with {len(current_batch)} emails...")

            # Create a batch request
            batch = self.gmail_handler.service.new_batch_http_request()

            # Add each message to the batch
            for msg_data in current_batch:
                msg_id = msg_data['id']
                batch.add(
                    self.gmail_handler.service.users().messages().get(
                        userId="me", id=msg_id, format='full'),
                    request_id=msg_id,
                    callback=lambda request_id, response, exception:
                        self.process_email_response(
                            request_id, response, exception, {})
                )

            # Execute the batch request with backoff
            Logger.log(
                f"Executing batch request for {len(current_batch)} emails...")
            batch_exec_start = time.time()

            try:
                # Apply backoff strategy for the batch request
                self.gmail_handler.api_request_with_backoff(
                    lambda: batch.execute(),
                    f"batch {batch_count} execution"
                )
            except Exception as e:
                Logger.log(f"Error executing batch {batch_count}: {e}")
                # Add all IDs in this batch to the failed list for individual retry
                for msg_data in current_batch:
                    self.gmail_handler.failed_message_ids.add(msg_data['id'])

            batch_exec_time = time.time() - batch_exec_start
            Logger.log(
                f"Batch execution completed in {batch_exec_time:.2f} seconds")

            # Save index after each batch
            with open(Config.INDEX_FILE, "w", encoding='utf-8') as outfile:
                json.dump(self.index_file, outfile,
                          ensure_ascii=False, indent=2)

            batch_time = time.time() - batch_start
            Logger.log(
                f"Completed batch {batch_count}/{total_batches} in {batch_time:.2f} seconds")

            # If we hit too many rate limits, pause to let quota reset
            if len(self.gmail_handler.failed_message_ids) > Config.MAX_BATCH_SIZE:
                Logger.log(
                    f"Too many rate limit errors. Pausing for {Config.QUOTA_RESET_TIME} seconds...")
                time.sleep(Config.QUOTA_RESET_TIME)

            # Calculate estimated time remaining
            if batch_count < total_batches:
                avg_time_per_batch = batch_time
                remaining_batches = total_batches - batch_count
                est_time_remaining = avg_time_per_batch * remaining_batches
                est_completion_time = datetime.datetime.now(
                ) + datetime.timedelta(seconds=est_time_remaining)
                Logger.log(
                    f"Estimated time remaining: {est_time_remaining/60:.1f} minutes")
                Logger.log(
                    f"Estimated completion time: {est_completion_time.strftime('%H:%M:%S')}")

    def retry_failed_messages(self) -> None:
        """Retry processing for failed messages individually."""
        if not self.gmail_handler.failed_message_ids:
            return

        Logger.log(
            f"Retrying {len(self.gmail_handler.failed_message_ids)} failed messages individually...")

        # Convert set to list for iteration
        failed_ids = list(self.gmail_handler.failed_message_ids)
        retry_successes = 0

        for i, msg_id in enumerate(failed_ids):
            # Add delay between requests for rate limiting
            if i > 0:
                time.sleep(Config.REQUEST_INTERVAL)

            if self.process_message_individually(msg_id):
                retry_successes += 1

            # Periodically show progress
            if (i + 1) % 10 == 0 or i == len(failed_ids) - 1:
                Logger.log(
                    f"Retry progress: {i+1}/{len(failed_ids)} ({retry_successes} successful)")

            # Save index file periodically during retries
            if (i + 1) % 20 == 0:
                with open(Config.INDEX_FILE, "w", encoding='utf-8') as outfile:
                    json.dump(self.index_file, outfile,
                              ensure_ascii=False, indent=2)

        Logger.log(
            f"Retried {len(failed_ids)} messages, {retry_successes} succeeded")

    def generate_statistics(self) -> None:
        """Generate statistics about the email processing."""
        Logger.log("Generating email cleaning statistics...")
        total_original_size = sum(email.get('original_body_length', 0)
                                  for email in self.index_file['emails'])
        total_cleaned_size = sum(email.get('cleaned_body_length', 0)
                                 for email in self.index_file['emails'])

        if total_original_size > 0:
            reduction_percentage = (
                (total_original_size - total_cleaned_size) / total_original_size) * 100
            Logger.log(
                f"Total original content size: {total_original_size} characters")
            Logger.log(
                f"Total cleaned content size: {total_cleaned_size} characters")
            Logger.log(f"Content reduction: {reduction_percentage:.2f}%")

        # Calculate storage usage
        Logger.log("Calculating storage usage...")
        total_size = sum(os.path.getsize(f"{Config.OUTPUT_DIR}/{f}")
                         for f in os.listdir(Config.OUTPUT_DIR))
        Logger.log(f"Total storage used: {total_size / (1024*1024):.2f} MB")

        if len(self.index_file['emails']) > 0:
            Logger.log(
                f"Average size per email: {total_size / len(self.index_file['emails']) / 1024:.2f} KB")


def merge_email_content(
    existing_filepath: str,
    new_email_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge the content of an existing email file with a new email object.
    Handle date and body switching based on the described logic.

    Args:
        existing_filepath: Path to the existing email file.
        new_email_obj: The new email object to merge.

    Returns:
        The merged email object.
    """
    if os.path.exists(existing_filepath):
        with open(existing_filepath, 'r', encoding='utf-8') as existing_file:
            existing_email = json.load(existing_file)

        Logger.log(f"Thread Path exists, {existing_filepath}")

        # Remove redundant content from the new email body
        new_email_obj['original_body'] = EmailCleaner.remove_redundant_content(
            new_email_obj['original_body'])
        new_email_obj['cleaned_body'] = EmailCleaner.remove_redundant_content(
            new_email_obj['cleaned_body'])

        # Compare timestamps
        existing_date = existing_email.get('date', '')
        new_date = new_email_obj.get('date', '')

        # Parse dates for comparison
        existing_datetime = datetime.datetime(
            *parsedate(existing_date)[:6]) if existing_date and parsedate(existing_date) else None
        new_datetime = datetime.datetime(
            *parsedate(new_date)[:6]) if new_date and parsedate(new_date) else None

        # Determine the earlier timestamp
        if existing_datetime and new_datetime:
            if new_datetime < existing_datetime:
                # New email is earlier: switch the date, keep the body
                existing_email['date'] = new_date
            else:
                # Existing email is earlier: keep the date, switch the body
                existing_email['original_body'] = new_email_obj['original_body']
                existing_email['cleaned_body'] = new_email_obj['cleaned_body']
        else:
            # If dates are missing or invalid, default to replacing the body
            existing_email['original_body'] = new_email_obj['original_body']
            existing_email['cleaned_body'] = new_email_obj['cleaned_body']

        # Save the updated email object back to the file
        with open(existing_filepath, 'w', encoding='utf-8') as updated_file:
            json.dump(existing_email, updated_file,
                      ensure_ascii=False, indent=2)

        return existing_email
    else:
        # If no existing file, create one with the new email object
        with open(existing_filepath, 'w', encoding='utf-8') as new_file:
            json.dump(new_email_obj, new_file, ensure_ascii=False, indent=2)

        return new_email_obj

# ---- Main Application ----


class GmailEmailProcessor:
    def __init__(self):
        self.gmail_handler = GmailHandler()
        self.email_processor = EmailProcessor(self.gmail_handler)

    def run(self):
        """Main function to process Gmail emails."""
        script_start_time = time.time()

        try:
            Logger.log(
                "=== Starting Gmail email retrieval script with rate-limited batch processing and cleaning ===")

            # Initialize service
            Logger.log("Authenticating with Gmail...")
            self.gmail_handler.authenticate()

            # Create output directory
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            Logger.log(f"Created output directory: {Config.OUTPUT_DIR}/")

            # Get message IDs
            message_ids = self.gmail_handler.get_message_ids()

            # Process emails in batches
            Logger.log(
                f"Processing emails in batches of {Config.MAX_BATCH_SIZE} with rate limiting and cleaning...")
            self.email_processor.process_in_batches(message_ids)

            # Retry failed messages
            self.email_processor.retry_failed_messages()

            # Generate statistics
            self.email_processor.generate_statistics()

            # Final stats
            script_end_time = time.time()
            total_script_time = script_end_time - script_start_time
            total_emails = len(self.email_processor.index_file['emails'])

            Logger.log("=== Email Retrieval and Cleaning Complete ===")
            Logger.log(
                f"Successfully retrieved, cleaned, and saved {total_emails} emails")
            Logger.log(
                f"Total script runtime: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")

            if total_emails > 0:
                avg_time_per_email = total_script_time / total_emails
                Logger.log(
                    f"Average time per email: {avg_time_per_email:.2f} seconds")
                Logger.log(
                    f"Emails per second: {total_emails/total_script_time:.2f}")

        except HttpError as error:
            Logger.log(f"Gmail API HTTP error: {error}")
        except Exception as e:
            Logger.log(f"An unexpected error occurred: {e}")
            import traceback
            Logger.log(traceback.format_exc())


if __name__ == "__main__":
    processor = GmailEmailProcessor()
    processor.run()
