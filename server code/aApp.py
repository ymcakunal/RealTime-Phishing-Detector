import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the models and necessary data
with open('D:\\LR+NN+NB\\advance featured\\lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('D:\\LR+NN+NB\\advance featured\\nn_model.json', 'r') as json_file:
    nn_model_json = json_file.read()
nn_model = model_from_json(nn_model_json)
nn_model.load_weights('D:\\LR+NN+NB\\advance featured\\nn_model.h5')
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open('D:\\LR+NN+NB\\advance featured\\bilstm_model.json', 'r') as json_file:
    bilstm_model_json = json_file.read()
bilstm_model = model_from_json(bilstm_model_json)
bilstm_model.load_weights('D:\\LR+NN+NB\\advance featured\\bilstm_model.h5')
bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open('D:\\LR+NN+NB\\advance featured\\nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open('D:\\LR+NN+NB\\advance featured\\tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('D:\\LR+NN+NB\\advance featured\\tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_seq_length = 100  # Maximum length of a sequence

# Initialize feedback variables
feedback_data = {'like': 0, 'dislike': 0}

def predict_email(email_text):
    # Predict using NB model
    user_features_tfidf = vectorizer.transform([email_text])
    nb_prediction = nb_model.predict(user_features_tfidf)[0]
    
    # Predict using NN model
    user_seq = tokenizer.texts_to_sequences([email_text])
    user_pad = pad_sequences(user_seq, maxlen=max_seq_length, padding='post')
    nn_prediction = nn_model.predict(user_pad)[0][0]
    
    # Predict using BiLSTM model
    bilstm_prediction = bilstm_model.predict(user_pad)[0][0]
    
    # Combine predictions into meta-features
    user_meta_features = np.hstack((user_features_tfidf.toarray(), [[nn_prediction]], [[bilstm_prediction]]))
    
    # Predict using LR model
    lr_prediction = lr_model.predict(user_meta_features)[0]
    
    # Weighted majority voting for final prediction
    lr_weight = 0.4
    nn_weight = 0.2
    bilstm_weight = 0.2
    nb_weight = 0.2
    
    final_score = (lr_prediction * lr_weight) + (nn_prediction * nn_weight) + (bilstm_prediction * bilstm_weight) + (nb_prediction * nb_weight)
    final_prediction = 1 if final_score >= 0.5 else 0
    
    return final_prediction

st.title("Email Spam Detector")

query_params = st.experimental_get_query_params()
subject = query_params.get("subject", [""])[0]
body = query_params.get("body", [""])[0]

if subject and body:
    email_text = f"Subject: {subject}\n\nBody: {body}"
    result = predict_email(email_text)
    if result == 1:
        st.write("Consensus: The content is predicted to be spam.")
    else:
        st.write("Consensus: The content is not predicted to be spam.")
    
    if st.button("Like"):
        feedback_data['like'] += 1
        with open('D:\\LR+NN+NB\\advance featured\\feedback_data.json', 'w') as file:
            json.dump(feedback_data, file)
        st.write("Thank you for your feedback!")
    
    if st.button("Dislike"):
        feedback_data['dislike'] += 1
        with open('D:\\LR+NN+NB\\advance featured\\feedback_data.json', 'w') as file:
            json.dump(feedback_data, file)
        st.write("Thank you for your feedback!")
else:
    st.write("No email content provided.")
