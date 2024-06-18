import textdistance as LDA
import re
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
import random
import numpy as np
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import mysql.connector
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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

    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r"-", " - ", text)

    text = re.sub(r"[-,.{}+=|?'()\:@]", "", text)

    text = text.replace('\n', '')

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

# Function to remove rows with missing values
def remove_rows_with_missing_values(df, columns):
    missing_values_mask = df[columns].isnull().any(axis=1)
    cleaned_df = df[~missing_values_mask]
    return cleaned_df

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

        # Fetch data and store in DataFrame
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['Questions', 'Answers'])

        print(df)

        df = remove_rows_with_missing_values(df, ['Questions', 'Answers'])

        questions = df["Questions"].tolist()
        answers = df["Answers"].tolist()

        # Check for missing values in 'Questions' and 'Answers' columns
        missing_questions = df['Questions'].isnull()
        missing_answers = df['Answers'].isnull()

        # Filter out missing values
        questions = df.loc[~missing_questions, 'Questions'].tolist()
        answers = df.loc[~missing_answers, 'Answers'].tolist()

        # Close the connection to the MySQL database
        connection.close()

        # Preprocess the questions and answers
        cl_ques = [preprocess_text_input(question) for question in questions]
        cl_ans = [preprocess_text_input(answer) for answer in answers]

        return cl_ques, cl_ans
    except Exception as e:
        print(f"Error retrieving data from MySQL database: {str(e)}")
        return [], []

cl_ques, cl_ans = retrieve_data_for_model()

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
    
# Preprocessing
answers_with_tags = ['<START> ' + answer + ' <END>' for answer in cl_ans]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(cl_ques + answers_with_tags)

# Ensure special tokens are in the tokenizer vocabulary
special_tokens = ['<START>', '<END>', '<OUT>']
for token in special_tokens:
    if token not in tokenizer.word_index:
        tokenizer.word_index[token] = len(tokenizer.word_index) + 1

VOCAB_SIZE = len(tokenizer.word_index) + 1

# Tokenize and pad questions
tokenized_questions = tokenizer.texts_to_sequences(cl_ques)
maxlen_questions = max([len(x) for x in tokenized_questions])
encoder_input_data = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')

# Tokenize and pad answers
tokenized_answers = tokenizer.texts_to_sequences(answers_with_tags)
maxlen_answers = max([len(x) for x in tokenized_answers])
decoder_input_data = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')

# Remove the start token for decoder output data
decoder_output_data = []
for seq in tokenized_answers:
    decoder_output_data.append(seq[1:])  # Remove the start token
decoder_output_data = preprocessing.sequence.pad_sequences(decoder_output_data, maxlen=maxlen_answers, padding='post')

LSTM_SIZE = 512
LR = 0.0001
EPOCH = 200
BATCH_SIZE = 64

# Define Word2Vec model parameters
VECTOR_SIZE = 1024 
WINDOW = 5 
MIN_COUNT = 1 
WORKERS = 4
SG = 1 

def load_seq2seq_model():
    seq2seq_model_path = 'C:/Users/JanRoShin/Desktop/GitHub/UAQTEbot_django/application\Models/T200.keras'
    
    try:
        # Load the Seq2Seq model
        model = load_model(seq2seq_model_path)
        print("Seq2Seq model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading Seq2Seq model: {str(e)}")
        return None

model1 = load_seq2seq_model()

#loss, accuracy = model1.evaluate([encoder_input_data, decoder_input_data], decoder_output_data)
#print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

model1.summary()

# Extract encoder inputs and states
encoder_inputs = model1.input[0]  # First input to the model
encoder_outputs, state_h_enc, state_c_enc = model1.layers[4].output  # LSTM layer output and states
encoder_states = [state_h_enc, state_c_enc]
    
# Define the encoder model
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

# Define the decoder inputs and states
decoder_inputs = model1.input[1]  # Second input to the model
decoder_state_input_h = tf.keras.layers.Input(shape=(LSTM_SIZE,), name='input_3')
decoder_state_input_c = tf.keras.layers.Input(shape=(LSTM_SIZE,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse embeddings and LSTM from the original model
decoder_embedding = model1.layers[3](decoder_inputs)
decoder_lstm = model1.layers[5]
decoder_dense = model1.layers[6]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense(decoder_outputs)

# Define the decoder model
decoder_model = tf.keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Convert string to tokens
def str_to_tokens(sentence, tokenizer, max_len_questions):
    words = sentence.lower().split()
    tokens_list = [tokenizer.word_index.get(word, tokenizer.word_index['<OUT>']) for word in words if word in tokenizer.word_index]
    return pad_sequences([tokens_list], maxlen=max_len_questions, padding='post')

def get_response_from_seq2seq_model(input_question, encoder_model, decoder_model, tokenizer, max_len_questions):
    # Preprocess the input question
    input_question = preprocess_text_input(input_question)
    
    # Tokenize the input question
    tokenized_question = tokenizer.texts_to_sequences([input_question])[0]
    
    # Pad the tokenized question sequence
    tokenized_question = pad_sequences([tokenized_question], maxlen=max_len_questions, padding='post')
    
    # Initialize the decoder input with the start token
    decoder_input = np.zeros((1, 1))
    decoder_input[0, 0] = tokenizer.word_index['<START>']
    
    # Initialize the states values
    states_values = encoder_model.predict(tokenized_question)
    
    # Initialize the stop condition flag and decoded translation
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition:
        # Predict the output sequence for the current decoder input and states values
        dec_outputs, h, c = decoder_model.predict([decoder_input] + states_values)
        
        # Get the index of the word with the highest probability in the output sequence
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        
        # Find the word corresponding to the sampled word index in the tokenizer
        sampled_word = tokenizer.index_word.get(sampled_word_index, '<OUT>')
        
        # Append the sampled word to the decoded translation
        decoded_translation += ' ' + sampled_word
        
        # Update the decoder input for the next iteration
        decoder_input = np.zeros((1, 1))
        decoder_input[0, 0] = sampled_word_index
        
        # Check if the stop condition is met
        if sampled_word == '<END>' or len(decoded_translation.split()) > 100:
            stop_condition = True
        
        # Update the states values for the next iteration
        states_values = [h, c]

        # Clean up the response by removing occurrences of "end" and "<OUT>"
        cleaned_response = clean_response(decoded_translation)
    
    return cleaned_response.strip()

def clean_response(response):
    # Remove occurrences of "end" and "<OUT>" from the response
    cleaned_response = response.replace(' end', '').replace('<OUT>', '').strip()
    return cleaned_response

def start_chat(user_input):
    database = retrieve_db_for_lda()
    if not database:
        print("Error: Unable to retrieve data from database.")
        return None

    custom_response = handle_custom_responses(user_input)
    if custom_response:
        return custom_response

    database_response = get_response_from_database(user_input, database)
    seq2seq_response = get_response_from_seq2seq_model(user_input, encoder_model, decoder_model, tokenizer, maxlen_questions)

    # Display both responses
    response = f"\nUAQTEbot-Seq2Seq Model response: \n\n{seq2seq_response}\n\n\nUAQTEbot-Levenshetin Distance Algorithm response: \n\n{database_response}"
    return response

# Start the chatbot
if __name__ == "__main__":
    user_input = input("You: ")
    print(user_input)
    response = start_chat(user_input)
    print(response)

