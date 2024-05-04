import pandas as pd
import random
from nltk.corpus import wordnet
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the dataset
dataset = pd.read_csv("question_answer_dataset.csv")

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

# Specify the number of new rows to add
num_new_rows = 100

# Augment the dataset
augmented_questions = []
augmented_answers = []

for index, row in dataset.iterrows():
    question = row['Question']
    answer = row['Answer']
    for _ in range(num_new_rows):
        augmented_questions.append(replace_verb_synonyms(question))
        augmented_answers.append(replace_verb_synonyms(answer))

# Combine questions and answers into batches
augmented_data = [{'Question': q, 'Answer': a} for q, a in zip(augmented_questions, augmented_answers)]

# Save augmented data
augmented_dataset = pd.DataFrame(augmented_data)
augmented_dataset.to_csv("augmented_question_answer_dataset.csv", index=False)
