let lastEmailContent = null;

function checkForEmailContent() {
  const emailBody = document.querySelector('.a3s .aXjCH') || document.querySelector('.a3s.aiL');
  if (emailBody && emailBody.innerText) {
    let emailContent = emailBody.innerText;
    if (emailContent !== lastEmailContent) {
      lastEmailContent = emailContent;

      fetch('http://127.0.0.1:8000/api/spam-detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ emailContent: emailContent })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        const isSpam = data.isSpam ? 'Spam' : 'Not Spam';
        alert(`Spam detection result: ${isSpam}`);

        // Send result to background script
        chrome.runtime.sendMessage({ action: 'displayResult', emailContent: emailContent, result: isSpam });
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Failed to check spam');
      });
    }
  }
}

function startMonitoring() {
  setInterval(checkForEmailContent, 3000);
}

startMonitoring();
