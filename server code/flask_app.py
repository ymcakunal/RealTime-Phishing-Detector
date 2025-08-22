import tornado.ioloop
import tornado.web
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and necessary data
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

def predict_email(email_text):
    user_features_tfidf = vectorizer.transform([email_text])
    nb_prediction = nb_model.predict(user_features_tfidf)[0]
    user_seq = tokenizer.texts_to_sequences([email_text])
    user_pad = pad_sequences(user_seq, maxlen=max_seq_length, padding='post')
    nn_prediction = nn_model.predict(user_pad)[0][0]
    bilstm_prediction = bilstm_model.predict(user_pad)[0][0]
    user_meta_features = np.hstack((user_features_tfidf.toarray(), [[nn_prediction]], [[bilstm_prediction]]))
    lr_prediction = lr_model.predict(user_meta_features)[0]
    lr_weight = 0.4
    nn_weight = 0.2
    bilstm_weight = 0.2
    nb_weight = 0.2
    final_score = (lr_prediction * lr_weight) + (nn_prediction * nn_weight) + (bilstm_prediction * bilstm_weight) + (nb_prediction * nb_weight)
    final_prediction = 1 if final_score >= 0.5 else 0
    return final_prediction

class MainHandler(tornado.web.RequestHandler):
    def post(self):
        email_content = self.get_argument("emailContent")
        result = predict_email(email_content)
        self.write({"isSpam": result == 1})

def make_app():
    return tornado.web.Application([
        (r"/api/spam-detect", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()
