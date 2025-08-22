let latestResult = null;

chrome.runtime.onInstalled.addListener(() => {
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && tab.url.includes('mail.google.com')) {
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['content.js']
      });
    }
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'displayResult') {
    latestResult = { emailContent: message.emailContent, result: message.result };
  }
});

chrome.runtime.onConnect.addListener(port => {
  if (port.name === "popup") {
    port.postMessage(latestResult);
  }
});
