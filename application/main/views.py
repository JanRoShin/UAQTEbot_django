from django.http import HttpResponseNotFound
from django.http import JsonResponse
from django.shortcuts import render
from .helpers.function import try_me
from .lda_fun import get_response as lda_get_response, retrieve_db_QA as get_qa

"""
def index(request):
    context = {"data": "Jan Lance", "function_data": try_me()}

    _summary_

    call the context by the variable name they were in:
    e.g. in blogs website, no need for context['data']
    
    return render(request, "main/index.html", context)
"""

def chat_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')
        database = get_qa()  # Call function to retrieve database
        response = lda_get_response(user_input, database)  # Call the LDA function
        return JsonResponse({'message': response})
    else:
        return render(request, 'main/chat.html')

"""
def main(request):
    context = {"data": "hello this is main page", "function_data": try_me()}
    return render(request, "main/guest/index.html", context)


def settings(request):
    return render(request, "main/settings.html")
"""