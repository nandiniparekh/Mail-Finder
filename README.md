# Chrome Extension Gmail Integration

This repository contains an in-development Chrome extension that accesses your Gmail data via OAuth and the Gmail API. Follow the steps below to load the extension, generate the necessary API credentials, and use the `get_emails.py` script to set up all required permissions.

---

## 1. Clone the Repository and Load the Extension

### a. Clone the Repository

Clone the repository using Git or your preferred method:

```bash
git clone https://github.com/antonio2uofa/capstone
```

### b. Load the Unpacked Extension

1. Open Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer Mode** (top-right corner).
3. Click **Load Unpacked**.
4. Select the `extension` folder from your cloned repository.

### c. Find Your Chrome Extension ID

Once the extension is loaded, click **Details** on the extension card. Copy the **Extension ID** — you'll need this in the next step.

---

## 2. Create Gmail API Credentials in Google Cloud

### a. Configure the OAuth Consent Screen

Follow Google's guide to configure the consent screen:

👉 [Configure OAuth Consent Screen](https://developers.google.com/workspace/guides/configure-oauth-consent)

### b. Generate OAuth 2.0 Credentials

Follow this guide to create credentials for a Chrome extension:

👉 [Create OAuth Credentials for Chrome Extension](https://developers.google.com/workspace/guides/create-credentials#chrome-extension)

- Choose **"Chrome App"** as the application type.
- Add the extension's ID to the **Authorized JavaScript origins**:
  ```
  chrome-extension://<YOUR_EXTENSION_ID>
  ```
- No redirect URIs are needed for Chrome extensions.
- Download the `credentials.json` file.

---

## 3. Use `get_emails.py` to Authorize Gmail Access

### a. Place the Credentials File

Put the downloaded `credentials.json` file in the root of the repository.

### b. Run the Script

Run the script to trigger the OAuth flow and generate the necessary permissions:

```bash
python get_emails.py
```

- This will open a browser window where you'll log in and authorize access to your Gmail account.
- After successful authentication, a `token.json` file will be created, storing your access/refresh tokens.

---

## 4. Summary

✅ Clone the repo and load the Chrome extension in Developer Mode  
✅ Generate Gmail API credentials using your Chrome Extension ID  
✅ Place `credentials.json` in the repo and run `get_emails.py`  
✅ This will authorize Gmail access and allow the extension to function

Once setup is complete, your extension will be able to securely access and read your personal Gmail data through the Gmail API.

---

## 5. Add Your OpenAI API Key

To enable OpenAI-powered features in the extension, you must supply your own OpenAI API key.

### a. Locate the API Key

Navigate to the file where the key is defined:

```
search_emails.py
```

In that file, look for the line that sets `openai_api_key`:

```python
openai_api_key = "sk-..."
```

### b. Replace with Your Own Key

Replace the existing value with your own OpenAI API key, which you can generate here:

👉 [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

Example:

```python
openai_api_key = "sk-YourOwnKeyHere"
```

> ⚠️ **Do not share your API key publicly.** Keep it private and secure.

Once your OpenAI key is in place, the extension will be able to access and process email content using OpenAI's API.

---

## 6. Set Up Python Virtual Environment

To run the backend scripts (like `search_emails.py`), you should use a Python virtual environment to manage dependencies cleanly.

### a. Create a Virtual Environment

In your terminal, navigate to the project root and run:

```bash
python -m venv venv
```

This creates a folder named `venv` containing the virtual environment.

### b. Activate the Virtual Environment

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

Once activated, your terminal prompt will change to show the environment name.

### c. Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the dependencies needed by `get_emails.py`, including Gmail and OpenAI client libraries.

---

You're now ready to run the backend scripts in an isolated environment.

---

## 7. Run the Program and Use the Extension

### a. Start the Backend API

Make sure your virtual environment is **activated** (see step 6), then run:

```bash
python search_emails.py
```

This will launch the backend API that powers the Chrome extension.

### b. Use the Chrome Extension

1. In Google Chrome, click the **Extensions** button (puzzle icon) in the toolbar.
2. Click on the icon for **MailFinder** to open the app.
3. In the MailFinder popup:
   - Type your **search query** into the search bar.
   - Click the **filter icon** to choose from a set of filters (e.g., date, sender).
   - Click **Apply** to use selected filters, or just hit **Search** if you're not using any.

### c. View Your Emails

Search results will appear below. Click on any email in the list to open it directly in your Gmail web app.

---

✅ With the backend API running and your extension loaded, you now have a working Gmail search assistant powered by OpenAI and the Gmail API!

---

## 8. Running Tests

### a. Front-End Tests (Playwright)

Front-end tests are written using [Playwright](https://playwright.dev/). To run them, you need either:
- A **Unix-based environment** (Linux/macOS), or
- A working **Node.js installation** on any OS.

#### Install Playwright

1. Make sure you have Node.js installed:  
   👉 [Download Node.js](https://nodejs.org)

2. Navigate to the `extension/` directory:

```bash
cd extension
```

3. Install Playwright:

```bash
npm install -D @playwright/test
npx playwright install
```

4. Run the tests:

```bash
npx playwright test tests/server-test.js --headed
```

> ✅ `--headed` opens a visible browser so you can see the tests run.

---

### b. Backend Unit Tests with Coverage

To run unit tests and check code coverage:

1. Navigate to the project **root directory**:

```bash
cd path/to/project
```

2. Run the tests with coverage:

```bash
coverage run -m unittest discover
```

This will run all `unittest` tests and track coverage. You can generate a report afterward with:

```bash
coverage report
```

---

🧪 Now you're ready to test both the frontend and backend components of the project!
