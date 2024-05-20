import mysql.connector
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseNotFound
from django.http import JsonResponse
from django.shortcuts import render
from .helpers.function import try_me
from .lda_fun import preprocess_text_input, handle_custom_responses, get_response_from_database, get_response_from_seq2seq_model, start_chat, retrieve_db_for_lda as get_qa



"""
def index(request):
    context = {"data": "Jan Lance", "function_data": try_me()}

    _summary_

    call the context by the variable name they were in:
    e.g. in blogs website, no need for context['data']
    
    return render(request, "main/index.html", context)
"""



def chat_view(request):
    if request.method == 'GET':
        user_input = request.GET.get('user_input', '')
        if user_input:
            bot_response = start_chat(user_input)
            return JsonResponse({'bot_response': bot_response})
        else:
            return render(request, 'main/chat.html')


"""
def main(request):
    context = {"data": "hello this is main page", "function_data": try_me()}
    return render(request, "main/guest/index.html", context)


def settings(request):
    return render(request, "main/settings.html")



def chat_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')
        database = get_qa()  # Call function to retrieve database
        custom_response = custom_resp(user_input)
        if custom_response:
            return JsonResponse({'message': custom_response})
        else:
            # Call the function to get responses from database and model
            lda_response = lda_get_response(user_input, database)
            model_response = model_get_response(user_input)
            return JsonResponse({'lda_response': lda_response, 'model_response': model_response})
    else:
        return render(request, 'main/chat.html')
"""