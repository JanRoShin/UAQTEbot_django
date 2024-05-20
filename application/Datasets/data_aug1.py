import pandas as pd
import random
from nltk.corpus import wordnet
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Load the dataset
dataset = pd.read_csv("UAQTEbot_dataset.csv")

# Function to replace verbs with synonyms


def replace_verb_synonyms(sentence):
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    for i, (word, tag) in enumerate(tagged_words):
        if tag.startswith('VB'):  # Check if the word is a verb
            synsets = wordnet.synsets(word, pos='v')
            if synsets:
                synonym = random.choice(synsets).lemma_names()[0]
                words[i] = synonym
    return ' '.join(words)

# Function to calculate contextual embeddings


def get_contextual_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="tf",
                       padding=True, truncation=True)
    outputs = model(inputs)
    last_hidden_states = outputs.last_hidden_state.numpy()
    return last_hidden_states


# Function to replace words with synonyms based on contextual similarity
def replace_with_synonym(sentence):
    tokens = tokenizer.tokenize(sentence)
    embeddings = get_contextual_embeddings(sentence)
    # Reduce dimensionality to match cosine_similarity
    mean_embedding = np.mean(embeddings, axis=1).reshape(1, -1)

    new_tokens = []
    for i, token in enumerate(tokens):
        synonyms = wordnet.synsets(token)
        if synonyms:
            max_similarity = -1
            best_synonym = None
            for syn in synonyms:
                syn_token = tokenizer.tokenize(syn.lemmas()[0].name())
                syn_embedding = get_contextual_embeddings(' '.join(syn_token))
                syn_mean_embedding = np.mean(syn_embedding, axis=1).reshape(
                    1, -1)  # Reduce dimensionality
                similarity = cosine_similarity(
                    mean_embedding, syn_mean_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synonym = syn_token
            if best_synonym:
                new_tokens.extend(best_synonym)
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    return tokenizer.convert_tokens_to_string(new_tokens)


# Specify the number of new rows to add
num_new_rows = 25

# Augment the dataset
augmented_data = []
total_rows = len(dataset)
for index, row in dataset.iterrows():
    question = row['Questions']
    answer = row['Answers']
    for _ in range(num_new_rows):
        augmented_question = replace_verb_synonyms(question)
        augmented_question = replace_with_synonym(augmented_question)

        augmented_answer = replace_verb_synonyms(answer)
        augmented_answer = replace_with_synonym(augmented_answer)

        augmented_data.append(
            {'Questions': augmented_question, 'Answers': augmented_answer})

    # Print progress
    print(f"Processed {index+1}/{total_rows} rows")

print("Data augmentation completed!")

# Save augmented data
augmented_dataset = pd.DataFrame(augmented_data)
augmented_dataset.to_csv("augmented_UAQTEbot_dataset.csv", index=False)
