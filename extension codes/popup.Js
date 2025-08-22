const port = chrome.runtime.connect({ name: "popup" });

port.onMessage.addListener(message => {
  if (message) {
    document.getElementById('emailContent').innerText = message.emailContent;
    document.getElementById('resultText').innerText = `Spam detection result: ${message.result}`;
    document.getElementById('feedback-modal').style.display = 'block';
  }
});

function sendFeedback(emailContent, prediction, actual) {
  fetch('http://127.0.0.1:8000/api/feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({ emailContent: emailContent, prediction: prediction, actual: actual })
  })
  .then(() => {
    alert('Thank you for your feedback!');
    document.getElementById('feedback-modal').style.display = 'none';
  })
  .catch(error => {
    console.error('Error:', error);
    alert('Failed to send feedback');
  });
}

document.getElementById('yes').onclick = () => sendFeedback(document.getElementById('emailContent').innerText, 1, 1);
document.getElementById('no').onclick = () => sendFeedback(document.getElementById('emailContent').innerText, 1, 0);
