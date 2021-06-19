import numpy as np
import pandas as pd
import joblib
column_names=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep='\t',names=column_names)
df.head(5)

movie_titles = pd.read_csv("Movie_Id_Titles",header=0)
df=pd.merge(df,movie_titles,on='item_id')
df.head(5)

df.groupby('title')['rating'].count().sort_values(ascending=False).head(10)
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['Number_of_Ratings']=pd.DataFrame(df.groupby('title')['rating'].count())

moviemat = df.pivot_table(index='user_id',columns='title',values='rating',fill_value=0)
ratings.sort_values('Number_of_Ratings',ascending=False).head(10)
correlation_matrix=moviemat.corr(method='pearson')

filename='finalised_model.sav'
joblib.dump(correlation_matrix,filename)
