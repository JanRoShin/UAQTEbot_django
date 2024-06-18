from django.http import JsonResponse
from django.shortcuts import render
from .lda_fun import start_chat

import os

def save_to_file(user_input, response):
    file_path = 'C:/Users/JanRoShin/Desktop/GitHub/UAQTEbot_django/application/Saved Conversations/conversation_history.txt'
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode) as file:
        file.write(f"User Input: {user_input}\n")
        file.write(f"Response: {response}\n\n\n")

def chat_view(request):
    if request.method == 'GET':
        user_input = request.GET.get('user_input', '')
        if user_input:
            bot_response = start_chat(user_input)
            save_to_file(user_input, bot_response)
            return JsonResponse({'bot_response': bot_response})
        else:
            return render(request, 'main/chat.html')

