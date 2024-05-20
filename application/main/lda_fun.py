import textdistance as LDA
from tensorflow import keras
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
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
import mysql.connector

def preprocess_text_output(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def preprocess_text_input(text):
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Reformat Contractions
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

    text = re.sub(r"\b(?:{})\b".format("|".join(contraction_mapping.keys())), replace_contractions, text, flags=re.IGNORECASE)

    # Convert text to lowercase
    text = text.lower()

    return text

# Function to calculate Levenshtein distance between two strings
def ld_algo(s1, s2):
    return LDA.levenshtein.normalized_similarity(s1, s2)

# Function to connect to MySQL database and retrieve question-answer pairs
def retrieve_db_for_lda():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="BRT.mySQL2$24",
            database="qadb_lda"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT Questions, Answers FROM question_answer")
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return {question: answer for question, answer in rows}
    except Exception as e:
        print("Error retrieving data from database:", e)
        return {}

def retrieve_data_for_model():
    try:
        # Establish a connection to your MySQL database
        connection = mysql.connector.connect(
            host='127.0.0.1',
            database='qadb',
            user='root',
            password='BRT.mySQL2$24'
        )

        # Execute SQL query to retrieve data from MySQL database
        query = "SELECT question, answer FROM question_answer"
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

        return ques, ans
    except Exception as e:
        print(f"Error retrieving data from MySQL database: {str(e)}")
        return [], []

ques, ans = retrieve_data_for_model()

cl_ques = [preprocess_text_input(question) for question in ques]
cl_ans = [preprocess_text_input(answer) for answer in ans]

# Function to preprocess text
def find_closest_match(user_input, database):
    user_input = preprocess_text_input(user_input)
    best_match = None
    best_score = float('inf')
    for question in database:
        processed_question = preprocess_text_input(question)
        score = 1 - ld_algo(user_input, processed_question)
        print(
            f"\nComparison for question '{processed_question}':\n Levenshtein distance = {score}\n")
        if score < best_score:
            best_score = score
            best_match = question
    return best_match
def handle_custom_responses(user_input):
    # Common greetings and responses
    greetings = ["hello", "hi", "hey"]
    thank_you = ["thank you", "thanks", "okay, thank you",
                 " okay, thanks", "okay thank you", "okay thanks"]
    okay = ["okay"]

    user_input_lower = user_input.lower()

    # Check if the lowercase user input matches any of the lowercase greetings
    if any(greeting in user_input_lower for greeting in greetings):
        greeting_responses = [
            "Hello! What would you like to ask?",
            "Hi there! What are you thinking right now?",
            "Hey! What are your questions?"
        ]
        # Randomly select a greeting response
        greet_response = random.choice(greeting_responses)
        return greet_response

    # Check if the lowercase user input matches any of the lowercase thank you phrases
    if any(thanks in user_input_lower for thanks in thank_you):
        return "You're welcome!"

    if any(approval in user_input_lower for approval in okay):
        okay_responses = [
            "Are there any more questions that you would like to ask?",
            "Is there anything more that I can help you with?"
        ]
        # Randomly select an approval response
        approve_response = random.choice(okay_responses)
        return approve_response

    return None  # If no custom response is triggered

# Function to get response from the database
def get_response_from_database(user_input, database):
    closest_question = find_closest_match(user_input, database)
    if closest_question:
        return database.get(closest_question, "Sorry, I couldn't find a suitable answer.")
    else:
        return "Sorry, I didn't understand that question."









max_question_length = max(len(question.split()) for question in cl_ques)
max_answer_length = max(len(answer.split()) for answer in cl_ans)

print("Maximum question length:", max_question_length)
print("Maximum answer length:", max_answer_length)

max_sequence_length = 100
lstm=512

# Vocabulary creation and tokenization
word2count = {}
for line in cl_ques + cl_ans:
    for word in line.split():
        word2count[word] = word2count.get(word, 0) + 1

vocab = {word: idx for idx, word in enumerate(word2count.keys())}
vocab.update({'<PAD>': len(vocab), '<EOS>': len(vocab)+1, '<OUT>': len(vocab)+2, '<SOS>': len(vocab)+3})

inv_vocab = {v: w for w, v in vocab.items()}

print(len(vocab))

# Convert text to sequences
def text_to_sequences(text, vocab):
    return [[vocab.get(word, vocab['<OUT>']) for word in line.split()] for line in text]



encoder_inp = pad_sequences(text_to_sequences(cl_ques, vocab), maxlen=max_sequence_length, padding='post', truncating='post')
decoder_inp = pad_sequences(text_to_sequences(['<SOS> ' + answer + ' <EOS>' for answer in cl_ans], vocab), maxlen=max_sequence_length, padding='post', truncating='post')
decoder_final_output = to_categorical(pad_sequences(decoder_inp[:, 1:], maxlen=max_sequence_length, padding='post', truncating='post'), num_classes=len(vocab))



# Train Word2Vec model
sentences = [text.split() for text in cl_ques + cl_ans]
word2vec_model = Word2Vec(sentences, vector_size=256, window=5, min_count=1, workers=4)

# Get Word2Vec embeddings and vocabulary
word_vectors = word2vec_model.wv
word_vocab = word_vectors.key_to_index

# Replace the existing embedding layer with Word2Vec embeddings
embedding_matrix = np.zeros((len(vocab), word_vectors.vector_size))
for word, i in vocab.items():
    if word in word_vocab:
        embedding_matrix[i] = word_vectors[word]



# Define model architecture
embedding_layer = Embedding(len(vocab), word_vectors.vector_size, trainable=True, weights=[embedding_matrix])
encoder_input = Input(shape=(519,))
encoder_embed = embedding_layer(encoder_input)
encoder_lstm = LSTM(lstm, return_sequences=True, return_state=True)
encoder_output, encoder_h, encoder_c = encoder_lstm(encoder_embed)
encoder_states = [encoder_h, encoder_c]

# Define attention layer
attention_layer = Attention()

decoder_input = Input(shape=(519,))
decoder_embed = embedding_layer(decoder_input)

decoder_lstm = LSTM(lstm, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
attention_output = attention_layer([decoder_output, encoder_output])
decoder_concat_input = tf.concat([decoder_output, attention_output], axis=-1)

decoder_dense = Dense(len(vocab), activation='softmax')
decoder_output = decoder_dense(decoder_output)






def load_seq2seq_model():
    seq2seq_model_path = 'C:/Users/JanRoShin/Desktop/UAQTEbot_django/application/Models/model20_10k_w_att.h5'
    
    try:
        # Load the Seq2Seq model
        model = load_model(seq2seq_model_path)
        print("Seq2Seq model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Seq2Seq model: {str(e)}")
        return None

seq2seq_model=load_seq2seq_model()

seq2seq_model.summary()



# Define a function for inference
def get_response_from_seq2seq_model(question, model, max_sequence_length, vocab, inv_vocab):
    # Preprocess the input question
    preprocessed_question = preprocess_text_input(question)
    # Tokenize the preprocessed question
    tokenized_question = [vocab.get(word, vocab['<OUT>']) for word in preprocessed_question.split()]
    # Pad the tokenized question sequence
    padded_question = pad_sequences([tokenized_question], maxlen=max_sequence_length, padding='post', truncating='post')
    
    # Initialize the decoder input with SOS token
    decoder_input = np.zeros((1, max_sequence_length))
    decoder_input[0, 0] = vocab['<SOS>']

    # Predict the entire sequence
    predictions = model.predict([padded_question, decoder_input])[0]
    
    # Generate the response from predictions
    answer = ''
    for pred in predictions:
        predicted_word_index = pred.argmax(axis=-1)
        predicted_word = inv_vocab.get(predicted_word_index, '<OUT>')
        if predicted_word == '<EOS>':
            break
        answer += predicted_word + ' '
    
    return answer.strip()

def start_chat(user_input):
    database = retrieve_db_for_lda()
    if not database:
        print("Error: Unable to retrieve data from database.")
        return None

    custom_response = handle_custom_responses(user_input)
    if custom_response:
        return custom_response

    database_response = get_response_from_database(user_input, database)
    seq2seq_model = load_seq2seq_model()
    seq2seq_response = get_response_from_seq2seq_model(user_input, seq2seq_model, max_sequence_length, vocab, inv_vocab)

    return database_response if database_response else seq2seq_response

# Start the chatbot
if __name__ == "__main__":
    start_chat()

