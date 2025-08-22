# Phishing Email Detection System

## Overview
The **Phishing Email Detection System** is a machine learning-powered tool designed to detect and prevent phishing emails in real time. Phishing attacks attempt to steal sensitive information such as usernames, passwords, and credit card details by impersonating trusted entities.  

This system integrates **data collection, model training, and deployment via a Chrome extension**, providing real-time phishing detection and automated alerts. It is designed to be user-friendly while maintaining high security standards.

---

## Key Achievements
- Developed and deployed a **machine learning-based phishing email detection system** integrated with Gmail via a Chrome extension.  
- Implemented multiple models including **Logistic Regression, Neural Networks, Naive Bayes, and BiLSTM** for accurate classification.  
- Enabled **real-time phishing detection and automated blocking** of malicious links.  

### Contributions
- Designed and implemented a **Chrome extension** with content scripts, background scripts, and popup interface for seamless Gmail integration.  
- Engineered a **Tornado server backend** to handle detection requests and provide real-time results.  
- Managed **pre-trained ML models** to classify emails accurately.  
- Organized **end-to-end workflow** including server setup, extension integration, and small sample dataset testing.  

---

## Repository Structure
```
Phishing-Email-Detection-System/
│
├── server code/                 # Python server scripts for running detection
├── trained models/              # Pre-trained ML models (Logistic Regression, Naive Bayes, NN, BiLSTM)
├── extension codes/             # Chrome extension files
│   ├── background.js
│   ├── content.js
│   ├── manifest.json
│   ├── popup.css
│   └── popup.js
├── data sets/                   # Small sample datasets for demonstration
├── Building and Deploying an Email Spam Detector.pptx  # Project workflow and explanation
└── README.md
```

---

## How It Works
1. **Run the Server**  
   - Start the Python scripts from the `server code` folder.  
   - The server loads pre-trained models from the `trained models` folder and waits for requests.  

2. **Load the Chrome Extension**  
   - Open Chrome → go to Developer mode → load the `extension codes` folder as an unpacked extension.  
   - The extension interacts with Gmail to extract email content.  

3. **Real-Time Detection**  
   - The extension sends email content to the running Tornado server.  
   - The server analyzes emails using trained models and returns the phishing detection results.  
   - The extension displays alerts and phishing status in real time.  

4. **Dataset Testing**  
   - Use the small dataset in `data sets/` to test the system workflow without requiring the full dataset.  
   - Full datasets for model training are sourced from Kaggle.  

---

## Tools & Technologies
- **Programming Languages:** Python (server, ML, preprocessing), JavaScript (Chrome extension)  
- **ML Libraries:** Scikit-Learn (Logistic Regression, Naive Bayes), TensorFlow/Keras (Neural Networks, BiLSTM)  
- **Web & Server:** Tornado (backend server for real-time requests)  
- **Frontend:** HTML, CSS, AJAX (extension UI)  
- **Data Handling:** Pandas, Jupyter Notebooks  

---

## Notes
- The **PPT (`Building and Deploying an Email Spam Detector.pptx`)** explains the **project workflow, design, and methodology** in detail.  
- A **small sample dataset** is included for demonstration; full datasets are available on Kaggle for higher accuracy.  
- **Future Improvement:** A user feedback mechanism can be added to further enhance model accuracy while maintaining privacy.  

---

*This project demonstrates expertise in **machine learning, email security, Chrome extension development, real-time detection, and user-centric design**, making it a strong portfolio addition.*
