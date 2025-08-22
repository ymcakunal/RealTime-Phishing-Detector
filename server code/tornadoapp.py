import os
import tornado.ioloop
import tornado.web
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression

# Define the directory path
base_path = 'D:\\LR+NN+NB\\advance featured\\final\\'

# Load models and necessary data
with open(base_path + 'lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open(base_path + 'nn_model.json', 'r') as json_file:
    nn_model_json = json_file.read()
nn_model = model_from_json(nn_model_json)
nn_model.load_weights(base_path + 'nn_model.h5')
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open(base_path + 'bilstm_model.json', 'r') as json_file:
    bilstm_model_json = json_file.read()
bilstm_model = model_from_json(bilstm_model_json)
bilstm_model.load_weights(base_path + 'bilstm_model.h5')
bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with open(base_path + 'nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open(base_path + 'tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open(base_path + 'tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

max_seq_length = 100  # Maximum length of a sequence

# Path to the feedback CSV file
feedback_file = base_path + 'feedback_data.csv'

# Initialize feedback data
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
else:
    feedback_df = pd.DataFrame(columns=['email_content', 'prediction', 'actual'])

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

def update_feedback(email_content, prediction, actual):
    new_feedback = pd.DataFrame({'email_content': [email_content], 'prediction': [prediction], 'actual': [actual]})
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    else:
        feedback_df = new_feedback
    feedback_df.to_csv(feedback_file, index=False)

class MainHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        email_content = self.get_argument("emailContent")
        result = predict_email(email_content)
        self.write({"isSpam": result == 1})

class FeedbackHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def options(self):
        self.set_status(204)
        self.finish()

    def post(self):
        email_content = self.get_argument("emailContent")
        prediction = int(self.get_argument("prediction"))
        actual = int(self.get_argument("actual"))
        
        update_feedback(email_content, prediction, actual)
        self.write({"status": "success"})

def retrain_models():
    global lr_model, nn_model, bilstm_model, nb_model
    
    # Load feedback data
    feedback_df = pd.read_csv(feedback_file)

    # Assign weights: reward correct predictions, punish incorrect ones
    feedback_df['weight'] = feedback_df.apply(lambda row: 1.1 if row['prediction'] == row['actual'] else 0.9, axis=1)

    # Prepare training data
    X_feedback = vectorizer.transform(feedback_df['email_content'])
    y_feedback = feedback_df['actual']
    sample_weights = feedback_df['weight']

    # Retrain models
    lr_model.fit(X_feedback, y_feedback, sample_weight=sample_weights)
    nn_model.fit(X_feedback, y_feedback, sample_weight=sample_weights, epochs=10, batch_size=32, verbose=1)
    bilstm_model.fit(X_feedback, y_feedback, sample_weight=sample_weights, epochs=10, batch_size=32, verbose=1)
    nb_model.fit(X_feedback, y_feedback, sample_weight=sample_weights)

    # Save the updated models
    with open(base_path + 'lr_model.pkl', 'wb') as file:
        pickle.dump(lr_model, file)
    nn_model_json = nn_model.to_json()
    with open(base_path + 'nn_model.json', 'w') as json_file:
        json_file.write(nn_model_json)
    nn_model.save_weights(base_path + 'nn_model.h5')
    bilstm_model_json = bilstm_model.to_json()
    with open(base_path + 'bilstm_model.json', 'w') as json_file:
        json_file.write(bilstm_model_json)
    bilstm_model.save_weights(base_path + 'bilstm_model.h5')
    with open(base_path + 'nb_model.pkl', 'wb') as file:
        pickle.dump(nb_model, file)

def make_app():
    return tornado.web.Application([
        (r"/api/spam-detect", MainHandler),
        (r"/api/feedback", FeedbackHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    
    # Schedule periodic retraining every 24 hours
    tornado.ioloop.PeriodicCallback(retrain_models, 86400000).start()
    
    tornado.ioloop.IOLoop.current().start()
