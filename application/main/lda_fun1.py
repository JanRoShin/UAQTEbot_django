import mysql.connector
import re
import textdistance as LDA
import random

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to calculate Levenshtein distance between two strings
def calculate_similarity(s1, s2):
    return LDA.levenshtein.normalized_similarity(s1, s2)

# Function to connect to MySQL database and retrieve question-answer pairs
def retrieve_question_answer_pairs():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="BRT.mySQL2$24",
            database="qadb"
        )
        cursor = connection.cursor()
        cursor.execute("SELECT questions, answers FROM question_answer")
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return {question: answer for question, answer in rows}
    except Exception as e:
        print("Error retrieving data from database:", e)
        return {}

# Function to find the closest question match in the database
def find_closest_question_match(user_input, database):
    user_input = preprocess_text(user_input)
    best_match = None
    best_score = float('inf')
    for question in database:
        question_preprocessed = preprocess_text(question)
        score = 1 - calculate_similarity(user_input, question_preprocessed)
        print(f"\nComparison for question '{question}':\n Levenshtein distance = {score}\n")
        if score < best_score:
            best_score = score
            best_match = question
    return best_match

# Function to get response from the database
def get_response(user_input, database):
    greetings = ["hello", "hi", "hey"]
    thank_you = ["thank you", "thanks", "okay, thank you", " okay, thanks", "okay thank you", "okay thanks"]
    okay = ["okay"]

    user_input_lower = user_input.lower()

    if any(greeting in user_input_lower for greeting in greetings):
        greeting_responses = [
            "Hello! What would you like to ask?",
            "Hi there! What are you thinking right now?",
            "Hey! What are your questions?"
        ]
        greet_response = random.choice(greeting_responses)
        return greet_response

    if any(thanks in user_input_lower for thanks in thank_you):
        return "You're welcome!"

    if any(approval in user_input_lower for approval in okay):
        okay_responses = [
            "Are there any more questions that you would like to ask?",
            "Is there anything more that I can help you with?"
        ]
        approve_response = random.choice(okay_responses)
        return approve_response

    closest_question = find_closest_question_match(user_input, database)
    if closest_question:
        return database.get(closest_question, "Sorry, I couldn't find a suitable answer.")
    else:
        return "Sorry, I didn't understand that question."

# Load your trained model
def load_trained_model(model_file):
    model = load_model(model_file)
    return model

    # Define a function to create vocabulary from text data
def create_vocab(text_data):
    vocab = {'<PAD>': 0, '<EOS>': 1, '<OUT>': 2, '<SOS>': 3}  # Initialize with special tokens
    word_num = len(vocab)
    for text in text_data:
        for word in text.split():
            if word not in vocab:
                vocab[word] = word_num
                word_num += 1
    return vocab

# Combine the preprocessed questions and answers into a single list
combined_text = cl_ques + cl_ans

# Create vocabulary from combined text data
vocab = create_vocab(combined_text)

# Function to get response from the trained model
def get_response_from_model(user_input, enc_model, dec_model, vocab, max_output_len=100):
    # Preprocess user input
    preprocessed_input = preprocess_text(user_input.lower())
    input_seq = [vocab.get(word, vocab['<OUT>']) for word in preprocessed_input.split()]
    input_seq = pad_sequences([input_seq], maxlen=enc_model.input_shape[1], padding='post')
    
    # Encode input sequence
    states_value = enc_model.predict(input_seq)
    
    # Initialize target sequence with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = vocab['<SOS>']
    
    # Initialize decoded translation
    decoded_translation = ''
    
    # Decode output sequence word by word
    for _ in range(max_output_len):
        output_tokens, h, c = dec_model.predict([target_seq] + states_value)
        sampled_word_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = [word for word, index in vocab.items() if index == sampled_word_index][0]
        
        if sampled_word == '<EOS>':
            break
        
        decoded_translation += sampled_word + ' '
        
        # Update target sequence for next iteration
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index
        
        # Update states for next iteration
        states_value = [h, c]
    
    return decoded_translation.strip()

# Main function to interact with the chatbot
def start_chat():
    database = retrieve_question_answer_pairs()
    if not database:
        print("Error: Unable to retrieve data from database.")
        return
    
    # Load your trained model
    model_file = 'my_model.keras'
    model = load_trained_model(model_file)

    print("Welcome to the ChatBot! Ask me anything or type 'exit' to quit.")
    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        # Get response from the database
        db_response = get_response(user_input, database)
        # Get response from the model
        model_response = get_response_from_model(user_input, model)
        
        print("\nChatBot (Database):", db_response)
        print("ChatBot (Model):", model_response)

# Start the chatbot
if __name__ == "__main__":
    start_chat()
