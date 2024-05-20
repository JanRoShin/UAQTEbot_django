import keras
import re
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
import mysql.connector

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Connecting to the databse for getting the Questions and Rows as lists

# Establish a connection to your MySQL database
connection = mysql.connector.connect(
    host='127.0.0.1',
    database='qadb',
    user='root',
    password='BRT.mySQL2$24'
)

# Execute SQL query to retrieve data from MySQL database
query = "SELECT Questions, Answers FROM question_answer"
cursor = connection.cursor()
cursor.execute(query)

# Fetch data and store in lists
ques = []
ans = []
for row in cursor:
    ques.append(row[0])
    ans.append(row[1])

# Close the connection to the MySQL database
connection.close()

#######################################
# Text Preprocessing for the Input


def preprocess_text_input(text):
    contraction_mapping = {
        "arent": ["are not"],
        "cant": ["cannot"],
        "couldnt": ["could not"],
        "didnt": ["did not"],
        "doesnt": ["does not"],
        "dont": ["do not"],
        "hadnt": ["had not"],
        "hasnt": ["has not"],
        "havent": ["have not"],
        "hed": ["he had", "he would"],
        "hell": ["he will", "he shall"],
        "hes": ["he is", "he has"],
        "id": ["i had", "i would"],
        "ill": ["i will", "i shall"],
        "im": ["i am"],
        "ive": ["i have"],
        "isnt": ["is not"],
        "its": ["it is", "it has"],
        "lets": ["let us"],
        "mustnt": ["must not"],
        "shant": ["shall not"],
        "shed": ["she had", "she would"],
        "shell": ["she will", "she shall"],
        "shes": ["she is", "she has"],
        "shouldnt": ["should not"],
        "thats": ["that is", "that has"],
        "theres": ["there is", "there has"],
        "theyd": ["they had", "they would"],
        "theyll": ["they will", "they shall"],
        "theyre": ["they are"],
        "theyve": ["they have"],
        "wed": ["we had", "we would"],
        "were": ["we are"],
        "weve": ["we have"],
        "werent": ["were not"],
        "whatll": ["what will", "what shall"],
        "whatre": ["what are"],
        "whats": ["what is", "what has"],
        "whatve": ["what have"],
        "wheres": ["where is", "where has"],
        "whod": ["who had", "who would"],
        "wholl": ["who will", "who shall"],
        "whore": ["who are"],
        "whos": ["who is", "who has"],
        "whove": ["who have"],
        "wont": ["will not"],
        "wouldnt": ["would not"],
        "youd": ["you had", "you would"],
        "youll": ["you will", "you shall"],
        "youre": ["you are"],
        "youve": ["you have"]
    }

    def replace_contractions(match):
        contraction = match.group(0).lower()
        if contraction in contraction_mapping:
            options = contraction_mapping[contraction]
            return random.choice(options)
        else:
            return match.group(0)

    text = re.sub(r"\b(?:{})\b".format("|".join(
        contraction_mapping.keys())), replace_contractions, text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return text


cl_ques = [preprocess_text_input(question) for question in ques]
cl_ans = [preprocess_text_input(answer) for answer in ans]


#########################


# Preparation of the model for inference
max_question_length = max(len(question.split()) for question in cl_ques)
max_answer_length = max(len(answer.split()) for answer in cl_ans)

word2count = {}
for line in cl_ques + cl_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

vocab = {word: idx + 4 for idx, (word, _) in enumerate(word2count.items())}
vocab.update({'<PAD>': 0, '<EOS>': 1, '<OUT>': 2, '<SOS>': 3})

inv_vocab = {w: v for v, w in vocab.items()}

cl_ans = ['<SOS> ' + answer + ' <EOS>' for answer in cl_ans]

encoder_inp = pad_sequences([[vocab.get(word, vocab['<OUT>']) for word in line.split(
)] for line in cl_ques], maxlen=519, padding='post', truncating='post')
decoder_inp = pad_sequences([[vocab.get(word, vocab['<OUT>']) for word in line.split(
)] for line in cl_ans], maxlen=519, padding='post', truncating='post')

decoder_final_output = pad_sequences(
    decoder_inp[:, 1:], 519, padding='post', truncating='post')
decoder_final_output = to_categorical(
    decoder_final_output, num_classes=len(vocab))

VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE+1, output_dim=256)

enc_inp = Input(shape=(519,))
enc_embed = embed(enc_inp)
enc_lstm = LSTM(512, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_inp = Input(shape=(519,))
dec_embed = embed(dec_inp)
dec_lstm = LSTM(512, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(len(vocab), activation='softmax')
dense_op = dense(dec_op)

# Load the trained model
model_path = 'C:/Users/Account/Desktop/django/UAQTEbot_django/application/Models/wo-att_mod-20.keras'
model = load_model(model_path)

print("TensorFlow version:", tf.__version__)
print(keras.__version__)

# Show the model architecture
model.summary()

enc_model = Model(enc_inp, enc_states)

decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = dec_lstm(
    dec_embed, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

dec_model = Model([dec_inp] + decoder_states_inputs,
                  [decoder_outputs] + decoder_states)

####################################

# Function to generate response


def generate_response(input_text):
    preprocessed_input = preprocess_text_input(input_text)
    # Convert input text to sequence
    input_seq = []
    for word in preprocessed_input.split():
        try:
            input_seq.append(vocab[word])
        except KeyError:
            input_seq.append(vocab['<OUT>'])
    input_seq = pad_sequences([input_seq], maxlen=519, padding='post')
    # Initialize target sequence
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = vocab['<SOS>']
    # Initialize stop condition
    stop_condition = False
    decoded_translation = ''
    # Encoder states
    encoder_states = enc_model.predict(input_seq)

    while not stop_condition:
        # Predict next word
        decoder_outputs, h, c = dec_model.predict(
            [target_seq] + encoder_states)
        decoder_concat_input = dense(decoder_outputs)
        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '
        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word
        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 100:
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index
        encoder_states = [h, c]

    return decoded_translation.strip()


#####################################
# Example usage
input_text = "Hello"
response = generate_response(input_text)
print("Response:", response)
