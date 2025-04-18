const { chromium } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

/**
 * This script first launches Chrome with your extension,
 * then instructs you on how to manually start the recorder against that page
 */
async function prepareExtensionForRecording() {
  console.log('Starting browser with extension...');
  
  // Path to your extension directory - adjust this to your actual path
  const extensionPath = path.resolve(__dirname);
  console.log('Extension path:', extensionPath);
  
  // Launch a persistent context with a user data directory to ensure extensions load
  const userDataDir = path.join(__dirname, 'chrome-data-dir');
  if (!fs.existsSync(userDataDir)) {
    fs.mkdirSync(userDataDir, { recursive: true });
  }
  
  // Launch with your extension loaded
  const browser = await chromium.launchPersistentContext(userDataDir, {
    headless: false,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
    ]
  });
  
  // Create a page to navigate to chrome://extensions
  const page = await browser.newPage();
  await page.goto('chrome://extensions');
  
  // Wait for extensions to load
  await page.waitForTimeout(2000);
  
  // Find your extension ID
 let extensionId = await page.evaluate(() => {
    try {
      const manager = document.querySelector('extensions-manager');
      if (manager && manager.shadowRoot) {
        const itemList = manager.shadowRoot.querySelector('extensions-item-list');
        if (itemList && itemList.shadowRoot) {
          const items = itemList.shadowRoot.querySelectorAll('extensions-item');
          for (const item of items) {
            if (item.shadowRoot) {
              const name = item.shadowRoot.querySelector('.name');
              if (name && name.textContent.includes('Mail Finder')) {
                return item.getAttribute('id');
              }
            }
          }
        }
      }
    } catch (e) {
      console.error('Error finding extension:', e);
    }
    return null;
  });
  
  if (!extensionId) {
    extensionId = 'gcoajhcaekhbhbonmpogfoeojeijnkag';
  }
  
  console.log(`\n✅ Found extension ID: ${extensionId}`);
  
  // Save the extension ID to a file for reference
  fs.writeFileSync('extension-id.txt', extensionId);
  
  // Create a new tab for the popup
  const popupPage = await browser.newPage();
  const popupUrl = `chrome-extension://${extensionId}/popup.html`;
  await popupPage.goto(popupUrl);
  
  console.log('\n==================================================');
  console.log('🚀 Your extension popup is now loaded!');
  console.log('==================================================');
  console.log('\nTo record tests against this page:');
  console.log('\n1. Open a NEW terminal window');
  console.log(`2. Run this command: npx playwright codegen ${popupUrl}`);
  console.log('\nThe recorder will open and connect to your extension popup.');
  console.log('\nNote: Keep this terminal and browser window open while recording.');
  console.log('\nPress Ctrl+C in this terminal when you\'re done to close the browser.');
  
  // Keep the script running until manually terminated
  await new Promise(() => {}); // This will keep the script running until killed
}

// Run the function
prepareExtensionForRecording().catch(console.error);