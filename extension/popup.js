// import CryptoJS from "../node_modules/crypto-js";
const encryptionKey = "KUwAEgxBHwPjcW-uLfD8yfCgWUzVn3IH1tlBEIV535Q=";

async function decryptResponse(encryptedData) {
  try {
    console.log("Starting WebCrypto decryption");
    // Function to convert Base64 to ArrayBuffer
    function base64ToArrayBuffer(base64) {
      try {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
      } catch (e) {
        console.error("Base64 conversion error:", e);
        throw new Error("Invalid base64 input");
      }
    }
    
    // Function to convert ArrayBuffer to string
    function arrayBufferToString(buffer) {
      return new TextDecoder().decode(buffer);
    }
    
    if (!encryptedData || typeof encryptedData !== 'string' || encryptedData.length < 24) {
      throw new Error(`Invalid encrypted data: ${typeof encryptedData}, length: ${encryptedData ? encryptedData.length : 0}`);
    }
    
    // First, derive a proper key from the string
    // We need to hash the key string to get a consistent 32-byte key
    const keyData = new TextEncoder().encode(encryptionKey);
    const hashBuffer = await window.crypto.subtle.digest('SHA-256', keyData);
    
    // Import the key for use with AES-CBC
    const key = await window.crypto.subtle.importKey(
      "raw",
      hashBuffer,  // Use the hashed key
      { name: "AES-CBC" },
      false,
      ["decrypt"]
    );
    
    console.log("Key imported successfully");
    
    // Convert the encrypted data from Base64 to ArrayBuffer
    const encryptedBuffer = base64ToArrayBuffer(encryptedData);
    console.log("Encrypted data decoded, total length:", encryptedBuffer.byteLength);
    
    if (encryptedBuffer.byteLength <= 16) {
      throw new Error("Encrypted data too short, needs IV + content");
    }
    
    // Extract the IV (first 16 bytes)
    const iv = encryptedBuffer.slice(0, 16);
    console.log("IV extracted, length:", iv.byteLength);
    
    // Extract the actual ciphertext (everything after the IV)
    const ciphertext = encryptedBuffer.slice(16);
    console.log("Ciphertext extracted, length:", ciphertext.byteLength);
    
    // Decrypt the data
    console.log("Starting decryption operation");
    const decryptedBuffer = await window.crypto.subtle.decrypt(
      {
        name: "AES-CBC",
        iv: iv
      },
      key,
      ciphertext
    );
    console.log("Decryption operation completed");
    
    // Convert the decrypted data to a string
    const decryptedString = arrayBufferToString(decryptedBuffer);
    console.log("Decryption successful, string length:", decryptedString.length);
    
    // If the decrypted string is empty, that's suspicious
    if (!decryptedString || decryptedString.length === 0) {
      throw new Error("Decryption produced empty result");
    }
    
    // Try to parse as JSON
    try {
      console.log("Attempting to parse JSON");
      const result = JSON.parse(decryptedString);
      console.log("JSON parsing successful");
      return result;
    } catch (jsonError) {
      console.error("JSON parsing error:", jsonError);
      console.error("First 100 chars of decrypted string:", decryptedString.substring(0, 100));
      throw new Error("Decryption succeeded but produced invalid JSON");
    }
  } catch (error) {
    console.error("WebCrypto decryption error:", error);
    
    // Provide more specific error information
    if (error.name === "OperationError") {
      console.error("This is likely due to incorrect key, IV, or encrypted data format");
    } else if (error instanceof SyntaxError) {
      console.error("JSON parsing failed - decryption may have succeeded but produced invalid JSON");
    }
    
    throw error; // Re-throw to let the caller handle it
  }
}

// Cookie utility functions
function saveQueryToCookies(query) {
  // Get existing queries
  const previousQueries = getPreviousQueries();

  // Don't add duplicate queries
  if (previousQueries.length > 0 && previousQueries[0] === query) {
    return; // Already the most recent query
  }

  // Add current query to the beginning (most recent first)
  previousQueries.unshift(query);

  // Keep only the 2 most recent queries
  const limitedQueries = previousQueries.slice(0, 2);

  // Save to cookies (expires in 30 days)
  const expirationDate = new Date();
  expirationDate.setDate(expirationDate.getDate() + 30);

  document.cookie = `previousQueries=${encodeURIComponent(
    JSON.stringify(limitedQueries)
  )};expires=${expirationDate.toUTCString()};path=/`;

  // Update display immediately if the filter panel is visible
  const filterPanel = document.getElementById("filterPanel");
  if (filterPanel && filterPanel.classList.contains("active")) {
    updatePreviousQueriesDisplay();
  }
}

function getPreviousQueries() {
  const cookies = document.cookie.split(";");

  for (let cookie of cookies) {
    const [name, value] = cookie.trim().split("=");

    if (name === "previousQueries") {
      try {
        return JSON.parse(decodeURIComponent(value));
      } catch (error) {
        console.error("Error parsing previous queries from cookie:", error);
        return [];
      }
    }
  }

  return []; // Return empty array if no previous queries found
}

// Function to clear the cookie for testing purposes
function clearQueryHistory() {
  document.cookie =
    "previousQueries=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;";
  updatePreviousQueriesDisplay();
}

// Update the previous queries display
function updatePreviousQueriesDisplay() {
  const previousQueriesDisplay = document.getElementById(
    "previousQueriesDisplay"
  );
  const useContextToggle = document.getElementById("useContextToggle");
  const previousQueries = getPreviousQueries();

  console.log("Current previous queries:", previousQueries); // Debug log

  if (!previousQueriesDisplay) {
    console.error("previousQueriesDisplay element not found!");
    return;
  }

  if (previousQueries.length === 0) {
    previousQueriesDisplay.innerHTML = "<em>No previous queries</em>";
    if (useContextToggle) {
      useContextToggle.disabled = true;
    }
  } else {
    previousQueriesDisplay.innerHTML = previousQueries
      .map((query) => `<div class="previous-query">${query}</div>`)
      .join("");
    if (useContextToggle) {
      useContextToggle.disabled = false;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Element references
  const filterToggle = document.getElementById("filterToggle");
  const filterPanel = document.getElementById("filterPanel");
  const searchContainer = document.querySelector(".search-container");
  const clearFilters = document.getElementById("clearFilters");
  const applyFilters = document.getElementById("applyFilters");
  const startDate = document.getElementById("startDate");
  const endDate = document.getElementById("endDate");
  const senderInput = document.getElementById("senderInput");
  const queryForm = document.getElementById("queryForm");
  const searchInput = document.getElementById("searchInput");
  const resultsList = document.getElementById("resultsList");
  const resultsCount = document.getElementById("resultsCount");
  const useContextToggle = document.getElementById("useContextToggle");


  // Call immediately on load to set the proper initial size
  adjustExtensionHeight();

  // Only add the input event listener if both elements exist
  // Character limit handling
  if (searchInput) {
    // Set a reasonable max length if not already set
    if (!searchInput.hasAttribute('maxlength')) {
      searchInput.setAttribute('maxlength', '100');
    }
    
    // Get the actual max length
    const maxLength = parseInt(searchInput.getAttribute('maxlength'));
    
    // Find existing or create new character limit message
    let charLimitMessage = document.getElementById("charLimitMessage");
    if (!charLimitMessage) {
      charLimitMessage = document.createElement("div");
      charLimitMessage.id = "charLimitMessage";
      charLimitMessage.style.color = "#d93025";
      charLimitMessage.style.fontSize = "12px";
      charLimitMessage.style.marginTop = "4px";
      charLimitMessage.style.display = "none";
      charLimitMessage.textContent = `Character limit: ${maxLength} characters`;
      
      // Insert after search container
      const searchContainer = document.querySelector('.search-container');
      if (searchContainer) {
        searchContainer.appendChild(charLimitMessage);
      }
    }
    
    // Add the input event listener with proper check
    searchInput.addEventListener("input", () => {
      const currentLength = searchInput.value.length;
      const remainingChars = maxLength - currentLength;
      
      if (currentLength >= maxLength) {
        charLimitMessage.textContent = "Character limit reached";
        charLimitMessage.style.display = "block";
      } else if (currentLength > (maxLength * 0.8)) {
        // Show warning when approaching limit (80% or more)
        charLimitMessage.textContent = `${remainingChars} characters remaining`;
        charLimitMessage.style.display = "block";
      } else {
        charLimitMessage.style.display = "none";
      }
    });
  }
  // Display previous queries in the filter panel
  updatePreviousQueriesDisplay();

  // Watch for any DOM changes to adjust height automatically
  const container = document.querySelector(".container");
  if (container) {
    const observer = new MutationObserver(() => {
      adjustExtensionHeight();
    });

    // Start observing the container for all changes
    observer.observe(container, {
      childList: true,
      subtree: true,
      attributes: true,
    });
  }

  // Set max date to today for date inputs
  if (startDate && endDate) {
    const today = new Date().toISOString().split("T")[0];
    startDate.max = today;
    endDate.max = today;
  }

  // Track active filters
  let activeFilters = {
    startDate: "",
    endDate: "",
    sender: "",
    useContext: false,
  };

  // Toggle filter panel
  if (filterToggle && filterPanel) {
    filterToggle.addEventListener("click", () => {
      filterPanel.classList.toggle("active");

      // Update previous queries display when filter panel is opened
      if (filterPanel.classList.contains("active")) {
        updatePreviousQueriesDisplay();
      }

      // Adjust immediately for a snappier response
      adjustExtensionHeight();

      // And again after animation completes
      setTimeout(adjustExtensionHeight, 210);
    });
  }

  // Clear filters
  if (clearFilters && startDate && endDate && senderInput) {
    clearFilters.addEventListener("click", () => {
      startDate.value = "";
      endDate.value = "";
      senderInput.value = "";
      if (useContextToggle) {
        useContextToggle.checked = false;
      }
      updateActiveFilters();
      senderInput.classList.remove("invalid-input");

      // Adjust height after clearing
      adjustExtensionHeight();
    });
  }

  // Handle date change to ensure valid ranges
  if (startDate && endDate) {
    startDate.addEventListener("change", () => {
      if (endDate.value && startDate.value > endDate.value) {
        endDate.value = startDate.value;
      }
    });

    endDate.addEventListener("change", () => {
      if (startDate.value && endDate.value < startDate.value) {
        startDate.value = endDate.value;
      }
    });
  }

  // Validate sender email format when focus leaves the input
  if (senderInput) {
    senderInput.addEventListener("blur", () => {
      const email = senderInput.value.trim();
      if (email && !isValidEmail(email)) {
        // Simple UI feedback for invalid email
        senderInput.classList.add("invalid-input");
        // Add styling if not already present
        if (!document.getElementById("invalid-input-style")) {
          const style = document.createElement("style");
          style.id = "invalid-input-style";
          style.textContent = `
            .invalid-input {
              border-color: #d93025 !important;
              background-color: rgba(217, 48, 37, 0.05);
            }
          `;
          document.head.appendChild(style);
        }
      } else {
        senderInput.classList.remove("invalid-input");
      }
    });
  }

  // Email validation function
  function isValidEmail(email) {
    // Basic email validation regex
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    // Accept partial email addresses (just domain like gmail.com or just name)
    // This is useful for email search where users might want to search just by domain
    if (email.includes("@")) {
      return emailRegex.test(email);
    }
    // If it's just a domain or name part, it's acceptable
    return true;
  }

  // Apply filters
  if (applyFilters && filterPanel) {
    applyFilters.addEventListener("click", () => {
      // Validate inputs before applying
      const validInputs = validateFilterInputs();
      if (!validInputs) {
        return;
      }

      updateActiveFilters();
      filterPanel.classList.remove("active");

      // Adjust height after closing the panel
      adjustExtensionHeight();

      // If we have an active search query, refresh the search
      if (searchInput && searchInput.value.trim().length > 2) {
        searchEmails(searchInput.value.trim());
      }
    });
  }

  // Validate filter inputs
  function validateFilterInputs() {
    // Check date range
    if (startDate && endDate && startDate.value && endDate.value) {
      if (new Date(startDate.value) > new Date(endDate.value)) {
        alert("Start date must be before or equal to end date.");
        return false;
      }
    }

    // Check if dates are in the future
    const today = new Date();
    today.setHours(23, 59, 59, 999); // End of today

    if (startDate && startDate.value && new Date(startDate.value) > today) {
      alert("Start date cannot be in the future.");
      return false;
    }

    if (endDate && endDate.value && new Date(endDate.value) > today) {
      alert("End date cannot be in the future.");
      return false;
    }

    // Check sender email format if provided
    if (senderInput) {
      const sender = senderInput.value.trim();
      if (sender && sender.includes("@") && !isValidEmail(sender)) {
        alert("Please enter a valid email address or search term.");
        return false;
      }
    }

    return true;
  }

  // Update active filters
  function updateActiveFilters() {
    activeFilters = {
      startDate: startDate ? startDate.value : "",
      endDate: endDate ? endDate.value : "",
      sender: senderInput ? senderInput.value.trim() : "",
      useContext: useContextToggle ? useContextToggle.checked : false,
    };

    // Update visual indicator if any filters are active
    if (searchContainer) {
      const hasActiveFilters = Object.values(activeFilters).some(
        (value) =>
          value === true || (typeof value === "string" && value.length > 0)
      );
      searchContainer.classList.toggle("filter-active", hasActiveFilters);
    }
  }

  // Search form submit
  if (queryForm && searchInput && filterPanel) {
    queryForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = searchInput.value.trim();

      // Close filter panel if open
      if (filterPanel.classList.contains("active")) {
        filterPanel.classList.remove("active");
      }

      if (query.length > 2) {
        // Validate filter inputs before searching
        if (validateFilterInputs()) {
          // Do the search (saving to cookies happens inside searchEmails)
          searchEmails(query);
        }
      } else {
        alert("Please enter at least 3 characters to search.");
      }
    });
  }

  // Initialize
  updateActiveFilters();

  const reloadButton = document.getElementById("reloadButton");
  if (reloadButton && searchInput && resultsList && resultsCount && searchContainer && filterPanel) {
    reloadButton.addEventListener("click", () => {
      // Create a full-screen loading overlay
      const loadingOverlay = document.createElement("div");
      loadingOverlay.className = "loading-overlay";
      loadingOverlay.innerHTML = `
        <div class="loading-content">
          <div class="loading-spinner"></div>
          <p class="loading-message">Reloading data. This will take a minute, hang tight...</p>
        </div>
      `;
      
      // Add to the container
      const container = document.querySelector(".container");
      if (container) {
        container.appendChild(loadingOverlay);
      }
      
      // Hide the main form and results
      const queryForm = document.getElementById("queryForm");
      const resultsContainer = document.querySelector(".results-container");
      
      if (queryForm) queryForm.style.display = "none";
      if (resultsContainer) resultsContainer.style.display = "none";
      
      // Reset all form values
      searchInput.value = ""; // Clear the search bar
      if (startDate) startDate.value = ""; // Clear the start date filter
      if (endDate) endDate.value = ""; // Clear the end date filter
      if (senderInput) senderInput.value = ""; // Clear the sender filter
      if (useContextToggle) {
        useContextToggle.checked = false; // Uncheck the context toggle
      }
      resultsList.innerHTML = ""; // Clear any displayed results
      resultsCount.textContent = ""; // Clear the results count
      searchContainer.classList.remove("filter-active"); // Remove active filter indicator
      if (filterPanel) {
        filterPanel.classList.remove("active"); // Close the filter panel if open
      }

      // Adjust extension height for the overlay
      adjustExtensionHeight();

      // Call the /trigger-email-fetch endpoint
      fetch("http://127.0.0.1:8000/trigger-email-fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
        .then((response) => response.json())
        .then((data) => {
          // Remove the loading overlay
          if (loadingOverlay && loadingOverlay.parentNode) {
            loadingOverlay.parentNode.removeChild(loadingOverlay);
          }
          
          // Show the form again
          if (queryForm) queryForm.style.display = "block";
          
          // Show success message in results container
          if (resultsContainer) {
            resultsContainer.style.display = "block";
            resultsContainer.classList.add("active");
          }
          
          // Add success or failure message
          const message = document.createElement("li");
          message.className = "reload-message";
          
          if (data.status === "success") {
            message.innerHTML = `
              <div class="success-icon"></div>
              <span>We have your new emails! Happy searching :)</span>
            `;
            message.className += " success-message";
          } else {
            message.innerHTML = `
              <div class="error-icon"></div>
              <span>Error: ${data.message}</span>
            `;
            message.className += " error-message";
          }
          
          resultsList.innerHTML = "";
          resultsList.appendChild(message);

          // Adjust height after showing content again
          adjustExtensionHeight();
        })
        .catch((error) => {
          // Remove the loading overlay
          if (loadingOverlay && loadingOverlay.parentNode) {
            loadingOverlay.parentNode.removeChild(loadingOverlay);
          }
          
          // Show the form again
          if (queryForm) queryForm.style.display = "block";
          
          // Show error message in results container
          if (resultsContainer) {
            resultsContainer.style.display = "block";
            resultsContainer.classList.add("active");
          }
          
          // Add failure message
          const message = document.createElement("li");
          message.className = "reload-message error-message";
          message.innerHTML = `
            <div class="error-icon"></div>
            <span>Failed to trigger email fetch. Please try again.</span>
          `;
          
          resultsList.innerHTML = "";
          resultsList.appendChild(message);

          console.error("Error triggering email fetch:", error);

          // Adjust height after showing content again
          adjustExtensionHeight();
        });
    });
  }
});

// Function to adjust extension height - defined globally
function adjustExtensionHeight() {
  // Get container's natural height
  const container = document.querySelector(".container");
  if (!container) return;
  
  const containerHeight = container.scrollHeight;

  // Set a reasonable max height (for very large result lists)
  const maxHeight = 600;
  const newHeight = Math.min(containerHeight, maxHeight);

  // Apply height to html and body elements
  document.documentElement.style.height = `${newHeight}px`;
  document.body.style.height = `${newHeight}px`;
  document.documentElement.style.overflow = "hidden";
  document.body.style.overflow = "hidden";

  // Force container background to fill space properly
  container.style.minHeight = `${newHeight - 40}px`; // Account for padding
}

async function searchEmails(query) {
  // Get filters
  const startDate = document.getElementById("startDate");
  const endDate = document.getElementById("endDate");
  const senderInput = document.getElementById("senderInput");
  const useContextToggle = document.getElementById("useContextToggle");
  
  const startDateValue = startDate ? startDate.value : "";
  const endDateValue = endDate ? endDate.value : "";
  const senderValue = senderInput ? senderInput.value.trim() : "";
  const useContext = useContextToggle ? useContextToggle.checked : false;

  // Show loading indicator
  const resultsList = document.getElementById("resultsList");
  const resultsCount = document.getElementById("resultsCount");
  const resultsContainer = document.querySelector(".results-container");

  if (!resultsList || !resultsContainer) {
    console.error("Required elements not found for search display");
    return;
  }

  // Show results container if not already visible
  resultsContainer.classList.add("active");

  // Updated HTML for the search loading indicator
  resultsList.innerHTML = `
    <li class="loading search-loading">
      <div class="loading-spinner search-spinner"></div>
      <span>Searching emails...</span>
    </li>
  `;
  
  if (resultsCount) {
    resultsCount.textContent = "";
  }

  // Adjust extension height for loading indicator
  adjustExtensionHeight();

  // Get previous queries for context if enabled (BEFORE saving current query)
  let previousQueriesContext = [];
  if (useContext) {
    // Get the existing queries from cookies
    previousQueriesContext = getPreviousQueries();
    console.log("Using previous queries for context:", previousQueriesContext);
  }

  // Create payload with filters
  const payload = {
    input: {
      query: query,
      k: 10,
      search_method: "hybrid",
      hybrid_alpha: 0.7,
      generate_answer: true,
      // Create UTC dates at midnight
      start_date: startDateValue ? `${startDateValue}T00:00:00Z` : null,
      end_date: endDateValue ? `${endDateValue}T23:59:59Z` : null,
      sender: senderValue || null,
      previous_queries: previousQueriesContext.length > 0 && useContext ? previousQueriesContext : null,
    },
  };

  // AFTER creating the payload, now save the current query to cookies
  console.log("Saving query to cookies:", query);
  saveQueryToCookies(query);  // This should now be in scope

  // Use the custom direct endpoint
  fetch("http://127.0.0.1:8000/api/email-search/invoke-direct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then(async (data) => {
      console.log("Raw response data:", data);
      
      // First check for any error message from the server
      if (data.error) {
        console.warn("Server reported an error:", data.error);
      }
      
      // Check if we have debug data - use it first if available
      if (data._debug_plain) {
        console.log("Using debug data");
        displayResults(data._debug_plain);
        return;
      }
      
      // Check if the response contains encrypted data
      if (data.encrypted_data) {
        console.log("Received encrypted data, attempting to decrypt");
        
        try {
          // Decrypt the response using WebCrypto
          const decryptedData = await decryptResponse(data.encrypted_data);
          console.log("Decryption successful:", decryptedData);
          displayResults(decryptedData);
        } catch (decryptError) {
          console.error("Decryption error:", decryptError);
          
          // Try to provide more helpful error messaging
          let errorMessage = "Error decrypting the response.";
          if (decryptError.message && decryptError.message.includes("JSON")) {
            errorMessage = "Decryption succeeded but produced invalid data.";
          } else if (decryptError.name === "OperationError") {
            errorMessage = "Encryption key mismatch. Check that frontend and backend keys match.";
          }
          
          resultsList.innerHTML = `
            <li class="no-results">
              <p>${errorMessage} This is likely an issue with the encryption setup.</p>
            </li>
          `;
          adjustExtensionHeight();
        }
      } else {
        console.warn("No encrypted data or debug data found in response");
        // Handle missing data case
        resultsList.innerHTML = `
          <li class="no-results">
            <p>Invalid response format from server. Please check the server logs.</p>
          </li>
        `;
        adjustExtensionHeight();
      }
    })
    .catch((error) => {
      console.error("Error searching emails:", error);
      resultsList.innerHTML = `
        <li class="no-results">
          <p>Error searching emails: ${error.message}</p>
        </li>
      `;
      adjustExtensionHeight();
    });
}


function displayResults(response) {
  const resultsList = document.getElementById("resultsList");
  const resultsCount = document.getElementById("resultsCount");

  if (!resultsList) {
    console.error("Results list element not found");
    return;
  }

  // First, show the "Emails found!" success message
  resultsList.innerHTML = `
    <li class="search-success">
      <div class="success-icon"></div>
      <span>Emails found!</span>
    </li>
  `;

  // Adjust height for the success message
  adjustExtensionHeight();

  // Make sure we have response data
  if (!response || !response.results) {
    resultsList.innerHTML = `<li class="no-results">No response from server.</li>`;
    adjustExtensionHeight();
    return;
  }

  // Wait a short time to show the success message before displaying results
  setTimeout(() => {
    resultsList.innerHTML = "";

    console.log("API Response:", response);

    try {
      // Try to parse the structured JSON answer (array of email objects)
      let emailResults = [];
      let hasFilteredOutResults = false;

      if (response.results) {
        try {
          // emailResults = JSON.parse(response.results);
          emailResults = response.results;
          console.log("Parsed email results:", emailResults);

          // Check if we have results that were filtered out
          if (
            response.filtered_info &&
            response.filtered_info.has_results_outside_filter
          ) {
            hasFilteredOutResults = true;
          }

          // Ensure we have at most 5 results
          if (emailResults.length > 5) {
            emailResults = emailResults.slice(0, 5);
          }
        } catch (parseError) {
          console.error("Error parsing JSON:", parseError);
          console.log("Raw answer:", response.answer);
        }
      }

      // Update results count
      if (resultsCount) {
        resultsCount.textContent =
          emailResults.length > 0
            ? `${emailResults.length} ${
                emailResults.length === 1 ? "result" : "results"
              }`
            : "";
      }

      // Display "no results in range" message if applicable
      if (emailResults.length === 0 && hasFilteredOutResults) {
        // Remove any existing message
        const existingMessage = document.querySelector(".no-results-in-range");
        if (existingMessage) {
          existingMessage.remove();
        }

        const noResultsMessage = document.createElement("div");
        noResultsMessage.className = "no-results-in-range";
        noResultsMessage.innerHTML = `
          <p>No emails found with the current filters. There are results outside your filter criteria.</p>
          <button id="showAllResults" style="background: none; border: none; color: #1a73e8; padding: 0; font-size: 13px; cursor: pointer; margin-top: 4px;">Show all results</button>
        `;
        if (resultsList.parentNode) {
          resultsList.parentNode.insertBefore(noResultsMessage, resultsList);
        }

        // Add event listener to the "Show all results" button
        const showAllResults = document.getElementById("showAllResults");
        if (showAllResults) {
          showAllResults.addEventListener("click", () => {
            // Clear filters and search again
            const startDate = document.getElementById("startDate");
            const endDate = document.getElementById("endDate");
            const senderInput = document.getElementById("senderInput");
            const useContextToggle = document.getElementById("useContextToggle");
            const searchInput = document.getElementById("searchInput");
            
            if (startDate) startDate.value = "";
            if (endDate) endDate.value = "";
            if (senderInput) senderInput.value = "";
            if (useContextToggle) {
              useContextToggle.checked = false;
            }

            // Update filter state
            const searchContainer = document.querySelector(".search-container");
            if (searchContainer) {
              searchContainer.classList.remove("filter-active");
            }

            // Update active filters if function is available
            if (typeof updateActiveFilters === "function") {
              updateActiveFilters();
            }

            // Search again
            if (searchInput) {
              searchEmails(searchInput.value.trim());
            }

            // Remove the message
            noResultsMessage.remove();
          });
        }
      }

      // Display results
      if (emailResults && emailResults.length > 0) {
        // We have properly structured email results
        emailResults.forEach((email) => {
          let li = document.createElement("li");
          li.className = "email-result";

          // Format date if available
          let dateStr = "";
          if (email.metadata.date) {
            try {
              // Try to parse various date formats
              let date;
              if (email.metadata.date.includes(",")) {
                // Handle formats like "Tue, 25 Mar 2025 00:37:03 -0600"
                date = new Date(email.metadata.date);
              } else {
                // Handle formats like "2023-05-15"
                date = new Date(email.metadata.date);
              }
              dateStr = date.toLocaleDateString();
            } catch (e) {
              dateStr = email.metadata.date;
            }
          }

          // Create the HTML content
          li.innerHTML = `
            <strong>${email.metadata.subject || "(No Subject)"}</strong>
            <div class="email-meta">
              From: ${email.metadata.sender || "(Unknown)"} &middot; ${dateStr}
              ${
                email.metadata.relevance_score
                  ? ` &middot; Relevance: ${email.metadata.relevance_score}/10`
                  : ""
              }
            </div>
            <div class="email-snippet">${
              email.metadata.snippet || "(No preview available)"
            }</div>
          `;

          // Add click event to open in Gmail
          li.addEventListener("click", function () {
            if (email.metadata.subject && email.metadata.sender) {
              // Extract only the email address
              let senderRaw = email.metadata.sender ? email.metadata.sender : "(No Sender)";
              let emailMatch = senderRaw.match(/<([^>]+)>/);
              let sender = emailMatch ? emailMatch[1] : senderRaw;
              const queryStr = `subject:"${email.metadata.subject}" from:"${sender}"`;
              const gmailUrl = `https://mail.google.com/mail/u/0/#search/${encodeURIComponent(
                queryStr
              )}`;
              window.open(gmailUrl, "_blank");
            } else {
              console.error("Subject or sender not available for email:", email);
            }
          });

          resultsList.appendChild(li);
        });
      } else if (response.output.results && response.output.results.length > 0) {
        // Fallback to using raw chunks
        const limitedResults = response.output.results.slice(0, 5);

        limitedResults.forEach((chunk) => {
          let metadata = chunk.metadata;
          let li = document.createElement("li");
          li.className = "email-result";

          let dateStr = "";
          if (metadata.date) {
            try {
              const date = new Date(metadata.date);
              dateStr = date.toLocaleDateString();
            } catch (e) {
              dateStr = metadata.date;
            }
          }

          li.innerHTML = `
            <strong>${metadata.subject || "(No Subject)"}</strong>
            <div class="email-meta">
              From: ${metadata.sender || "(Unknown)"} &middot; ${dateStr}
            </div>
            <div class="email-snippet">
              ${chunk.content.substring(0, 100)}...
            </div>
          `;

          // Add click event to open in Gmail
          li.addEventListener("click", function () {
            if (metadata.subject && metadata.sender) {
              // Extract only the email address
              let senderRaw = metadata.sender ? metadata.sender : "(No Sender)";
              let emailMatch = senderRaw.match(/<([^>]+)>/);
              let sender = emailMatch ? emailMatch[1] : senderRaw;
              const queryStr = `subject:"${metadata.subject}" from:"${sender}"`;
              const gmailUrl = `https://mail.google.com/mail/u/0/#search/${encodeURIComponent(
                queryStr
              )}`;
              window.open(gmailUrl, "_blank");
            } else {
              console.error("Subject or sender not available for email:", chunk);
            }
          });

          resultsList.appendChild(li);
        });
      } else {
        resultsList.innerHTML = `<li class="no-results">No matching emails found.</li>`;
      }

      // Adjust height after results are displayed
      adjustExtensionHeight();
    } catch (error) {
      console.error("Error displaying results:", error);
      resultsList.innerHTML = `<li class="no-results">Error displaying results. Please try again.</li>`;
      adjustExtensionHeight();
    }
  }, 800); // Show success message for 800ms before displaying results
}