import mysql.connector
import re
import textdistance as LDA
import random


# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Function to calculate Levenshtein distance between two strings
def ld_algo(s1, s2):
    return LDA.levenshtein.normalized_similarity(s1, s2)

# Function to connect to MySQL database and retrieve question-answer pairs
def retrieve_db_QA():
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
def find_closest_match(user_input, database):
    user_input = preprocess_text(user_input)
    best_match = None
    best_score = float('inf')
    for question in database:
        question_preprocessed = preprocess_text(question)
        score = 1 - ld_algo(user_input, question_preprocessed)
        print(f"\nComparison for question '{question}':\n Levenshtein distance = {score}\n")
        if score < best_score:
            best_score = score
            best_match = question
    return best_match

# Function to get response from the database
def get_response(user_input, database):
    # Common greetings and responses
    greetings = ["hello", "hi", "hey"]
    thank_you = ["thank you", "thanks"]

    user_input_lower = user_input.lower()

    # Check if the lowercase user input matches any of the lowercase greetings
    if any(greeting in user_input_lower for greeting in greetings):
       # Define variations of greetings
        greeting_responses = [
            "Hello! What would you like to ask?",
            "Hi there! What are you thinking right now?",
            "Hey! What are your questions?"
        ]
        # Randomly select a greeting response
        response = random.choice(greeting_responses)
        return response

    # Check if the lowercase user input matches any of the lowercase thank you phrases
    if any(thanks in user_input_lower for thanks in thank_you):
        return "You're welcome!"

    closest_question = find_closest_match(user_input, database)
    if closest_question:
        return database.get(closest_question, "Sorry, I couldn't find a suitable answer.")
    else:
        return "Sorry, I didn't understand that question."
"""\
# Main function to interact with the chatbot
def start_chat():
    database = retrieve_db_QA()
    if not database:
        print("Error: Unable to retrieve data from database.")
        return

    print("Welcome to the ChatBot! Ask me anything or type 'exit' to quit.")
    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        response = get_response(user_input, database)
        print("\nChatBot:", response)

# Start the chatbot
if __name__ == "__main__":
    start_chat()
    """