import pandas as pd
import random
from nltk.corpus import wordnet
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForMaskedLM, pipeline

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

# Function for paraphrasing using BERT


def paraphrase(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    masked_pipeline = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    masked_text = text.replace(text.split()[random.randint(
        0, len(text.split()) - 1)], tokenizer.mask_token)
    result = masked_pipeline(masked_text)
    return result[0]['sequence']


# Specify the number of new rows to add
num_new_rows = 100

# Augment the dataset
augmented_data = []
for index, row in dataset.iterrows():
    question = row['Questions']
    answer = row['Answers']
    for _ in range(num_new_rows):
        augmented_question = replace_verb_synonyms(question)
        augmented_question = paraphrase(augmented_question)

        augmented_answer = replace_verb_synonyms(answer)
        augmented_answer = paraphrase(augmented_answer)

        augmented_data.append(
            {'Questions': augmented_question, 'Answers': augmented_answer})

# Save augmented data
augmented_dataset = pd.DataFrame(augmented_data)
augmented_dataset.to_csv("augmented_UAQTEbot_dataset.csv", index=False)
