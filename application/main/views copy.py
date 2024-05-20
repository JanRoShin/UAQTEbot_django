from django.http import HttpResponseNotFound
from django.http import JsonResponse
from django.shortcuts import render
from .helpers.function import try_me
from .lda_fun  import get_response_from_database, get_response_from_seq2seq_model, retrieve_db_QA, handle_custom_responses


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
        database = retrieve_db_QA()  # Call function to retrieve database
        
        # Check for custom responses before proceeding with further processing
        custom_response = handle_custom_responses(user_input)
        if custom_response:
            return JsonResponse({'message': custom_response})
        
        # If no custom response, continue with processing user input
        else:
            # Call the function to get responses from database and model
            lda_response = get_response_from_database(user_input, database)
            model_response = get_response_from_seq2seq_model(user_input)
            
            # Compare LDA values and choose the best response
            lda_score = compute_lda_score(user_input, lda_response)
            model_score = compute_lda_score(user_input, model_response)
            
            if lda_score > model_score:
                best_response = lda_response
            else:
                best_response = model_response
                
            return JsonResponse({'message': best_response})
    else:
        return render(request, 'main/chat.html')



"""
def main(request):
    context = {"data": "hello this is main page", "function_data": try_me()}
    return render(request, "main/guest/index.html", context)


def settings(request):
    return render(request, "main/settings.html")
"""
