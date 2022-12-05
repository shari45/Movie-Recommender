import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('mov.csv')
del df['Unnamed: 0']
v = CountVectorizer(max_features =5000, stop_words='english')
vectors=v.fit_transform(df['tags']).toarray()
vectors
movies=pickle.load(open('./movies.pkl','rb'))
movies=pd.DataFrame(movies)
#selected = st.selectbox('Enter or Select Movie Name',movies['title'].values)

similarity=pickle.load(open('./similarity.pkl','rb'))
def rec(movi):
    if movi not in df['title'].unique():
        return ('Thank you for the search, although I have trained this model on 5000 movies but it looks this movie does not exit in my database.\nPlease check if you spelled it correct. \nPlease use same name as movie poster. ')
    #movie_index = df[df['title']==movi].index[0]
    #if movie_index==null:
        #return('This movie is not in our database.\nPlease check if you spelled it correct.')
    
    else:
        movie_index = df[df['title']==movi].index[0]
        distances = similarity[movie_index]
        movies_list=sorted(list(enumerate(distances)), reverse=True ,key=lambda x:x[1])[1:10]
        l=[]
        for i in movies_list:
            movie_id=movies.iloc[i[0]].movie_id
            l.append((movies.iloc[i[0]].title))
        return l
#movies=pickle.load(open('./movies.pkl','rb'))
#movies=pd.DataFrame(movies)
#selected = st.selectbox('Enter or Select Movie Name',movies['title'].values)

#similarity=pickle.load(open('./similarity.pkl','rb'))
#recommend(ask_movie)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    #r = rcmd(movie)
    r= rec(movie)
    #aszmovie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    #app.run()
    app.run('localhost', 5000)