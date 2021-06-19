from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process


def home(request):
    return render(request,'home.html')

def result(request):
    model_knn= joblib.load('finalised_model2.sav')
    mat_movies_users= joblib.load('sparse.sav')
    df_movies= joblib.load('movie_data.sav')

    def recommender(movie_name,data,model, n_recommendations):
        model.fit(data)
        idx=process.extractOne(movie_name, df_movies['title'])[2]
        print('Movie Selected: ',df_movies['title'][idx], 'Index: ',idx)
        print('Searching for recommendations.....')
        distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
        similar_movies=[]
        for i in indices:
            for j in i:
                if(j!=idx):
                    similar_movies.append(df_movies['title'][j])
        return similar_movies

    movie_name=request.GET['m_name']
    arr=recommender(movie_name,mat_movies_users,model_knn,20)
    return render(request,'result.html',{'ans':arr})
