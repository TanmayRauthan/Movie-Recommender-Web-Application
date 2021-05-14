from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd
def home(request):
    return render(request,'home.html')

def result(request):
    correlation_matrix= joblib.load('finalised_model.sav')

    def get_similar(movie_name,rating):
        similar_score = correlation_matrix[movie_name]*(rating-2.5)
        similar_score = similar_score.sort_values(ascending=False)
        return similar_score

    movie=request.GET['m_name']
    rating=int(request.GET['rating_given'])
    similar_movies = pd.DataFrame()
    similar_movies = similar_movies.append(get_similar(movie,rating))
    cli=similar_movies.sum().sort_values(ascending=False).head(10)
    dfa=cli.to_frame()
    arr=[]
    for row in dfa.index:
        arr.append(row)

    print(arr[0])
    return render(request,'result.html',{'ans':arr})
