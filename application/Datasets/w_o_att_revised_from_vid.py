"""
from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/UAQTEbot/
"""

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Function to remove rows with missing values
def remove_rows_with_missing_values(df, columns):
    missing_values_mask = df[columns].isnull().any(axis=1)
    cleaned_df = df[~missing_values_mask]
    return cleaned_df

# Read the CSV file into a DataFrame
df = pd.read_csv("augmented_UAQTEbot_dataset.csv")
df = remove_rows_with_missing_values(df, ['Questions', 'Answers'])

ques = df['Questions'].tolist()
ans = df['Answers'].tolist()

# Text Preprocessing
def preprocess_text(text):
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
# Apply preprocessing to questions and answers
cl_ques = [preprocess_text(question) for question in ques]
cl_ans = [preprocess_text(answer) for answer in ans]



# Vocabulary creation
word2count = {}
for line in cl_ques + cl_ans:
    for word in line.split():
        word2count[word] = word2count.get(word, 0) + 1

vocab = {word: i + 4 for i, (word, _) in enumerate(word2count.items())}
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for i, token in enumerate(tokens):
    vocab[token] = len(vocab) + i

inv_vocab = {v: k for k, v in vocab.items()}


# Train Word2Vec model
sentences = [text.split() for text in cl_ques + cl_ans]
word2vec_model = Word2Vec(sentences, vector_size=256, window=55, min_count=1, workers=4)

# Get Word2Vec embeddings and vocabulary
word_vectors = word2vec_model.wv
word_vocab = word_vectors.key_to_index

# Replace the existing embedding layer with Word2Vec embeddings
embedding_matrix = np.zeros((len(vocab), word_vectors.vector_size))
for word, i in vocab.items():
    if word in word_vocab:
        embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Create a custom embedding layer using Word2Vec embeddings
embedding_layer = Embedding(len(vocab), word_vectors.vector_size, trainable=True)
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_matrix])

# Prepare encoder and decoder inputs
encoder_inp = [[vocab.get(word, vocab['<OUT>']) for word in line.split()] for line in cl_ques]
decoder_inp = [[vocab.get(word, vocab['<OUT>']) for word in line.split()] for line in cl_ans]

max_length = max(len(sequence) for sequence in encoder_inp + decoder_inp)

encoder_inp = pad_sequences(encoder_inp, maxlen=max_length, padding='post')
decoder_inp = pad_sequences(decoder_inp, maxlen=max_length, padding='post')

decoder_final_output = np.zeros((len(decoder_inp), max_length, len(vocab)), dtype='float32')
for i, sequence in enumerate(decoder_inp):
    for j, word_index in enumerate(sequence):
        decoder_final_output[i, j, word_index] = 1


# Define model architecture
enc_inp = Input(shape=(max_length,))
dec_inp = Input(shape=(max_length,))

# embed = Embedding(len(vocab) + 1, output_dim=256, trainable=True)
enc_embed = embedding_layer(enc_inp)
enc_lstm = LSTM(512, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_embed = embedding_layer(dec_inp)
dec_lstm = LSTM(512, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(len(vocab), activation='softmax')
dense_op = dense(dec_op)

# Compile the model with configurable learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = Model([enc_inp, dec_inp], dense_op)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=optimizer)



# Train model
model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=20)

# Save the trained model
model.save('my_model.keras')

# Evaluate model performance
loss, accuracy = model.evaluate([encoder_inp, decoder_inp], decoder_final_output)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')


# Define encoder and decoder models for inference
enc_model = Model(enc_inp, enc_states)

decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = dec_lstm(dec_embed, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
dec_model = Model([dec_inp] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Define function to calculate BLEU score
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

# Chat with the trained model
total_bleu_score = 0.0
num_samples = len(cl_ans)






while True:
    user_input = input("You: ")
    if user_input.lower() == 'q':
        break

    user_input = preprocess_text(user_input)
    preprocessed_input = [[vocab.get(word, vocab['<OUT>']) for word in user_input.split()]]
    preprocessed_input = pad_sequences(preprocessed_input, maxlen=max_length)

    enc_states_val = enc_model.predict(preprocessed_input)

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = vocab['<SOS>']

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + enc_states_val)
        decoder_concat_input = dense(dec_outputs)
        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '


        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 100:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        enc_states_val = [h, c]

    print("Chatbot: ", decoded_translation)
    bleu_score = calculate_bleu(cl_ans[i], decoded_translation)
    print("BLEU Score:", bleu_score)
    total_bleu_score += bleu_score

# Calculate average BLEU score over all samples
average_bleu_score = total_bleu_score / num_samples
print("Average BLEU Score:", average_bleu_score)