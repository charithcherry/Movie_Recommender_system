import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]


df=pd.read_csv("movies.csv")
print("data input done.....")
features=['keywords','cast','genres','director']

for feature in features:
    df[feature]=df[feature].fillna('')

def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("error",row)

print("Combining features ..........")
df['combined features']=df.apply(combine_features,axis=1)
print("Combining features done.......")
cv=CountVectorizer()
cm=cv.fit_transform(df['combined features'])
print("data processing.....")
cs=cosine_similarity(cm)
movie=input("Enter movie: ")


movie_index=get_index_from_title(movie)
sm=list(enumerate(cs[movie_index]))
sorted_sm=sorted(sm,key=lambda x:x[1],reverse=True)
print("processing done...")

i=0
for m in sorted_sm:
    print(get_title_from_index(m[0]))
    i=i+1
    if i==50:
        break
