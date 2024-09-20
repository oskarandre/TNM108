import numpy as np
import pandas as pd
import PySimpleGUI as sg

from ast import literal_eval

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

#Import data set
movies = pd.read_csv("C:/Users/oskar/OneDrive - Linköpings universitet/Maskininlärning/Projekt/Kod/Databas/archive(small)/tmdb_5000_movies.csv",
                     usecols=["id","budget", "genres", "overview", "original_title", "vote_average", "vote_count"], low_memory=False)

cast = pd.read_csv("C:/Users/oskar/OneDrive - Linköpings universitet/Maskininlärning/Projekt/Kod/Databas/archive(small)/tmdb_5000_credits.csv",
                   usecols=['movie_id','cast','crew'], low_memory=False)

cast.columns = ['id','cast','crew']

#Merge the two sets
movies= movies.merge(cast,on='id')


#================================== FILTERING =========================================

#Define features in the film and extract the data
features = ['cast', 'crew', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)

# Find the director from the crew-feature. If director is not found, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the top 3 items or the complete list, whichever is larger
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]

        #See if there are more than three items. If there are, give just the first three. If not, give the whole list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing data
    return []


# Set up the details for the new director, cast, genre
movies['director'] = movies['crew'].apply(get_director)

features = ['cast', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)


# Calculated the weighted average score
def weighted_avg_m2(distribution, weights):
    weighted_sum = []
    for score, weight in zip(distribution, weights):
        weighted_sum.append(score * weight)

    return round(sum(weighted_sum) / sum(weights),2)



#==================================== GUI =============================================
sg.theme('Dark Gray 2')  

layout = [[sg.Text('Movie rating predictor 2000')],
      [sg.Text('Genre', size=(18, 1)), sg.InputText()],
      [sg.Text('Actor', size=(18, 1)), sg.InputText()],
      [sg.Text('Director', size=(18, 1)), sg.InputText()],
      [sg.Text('Short description of plot', size=(18, 1)), sg.InputText()],
      [sg.Submit(), sg.Cancel()]]

window = sg.Window('Movie rating predictor 2000', layout)

event, values = window.read()
window.close()
genres_input, cast_input, Director_input, overview_input  = values[0], values[1], values[2] ,values[3]        # get the data from the values dictionary



#================================= ADDING OWN MOVIE ===================================

new_row = pd.DataFrame({'budget': [666],
                        'genres': [genres_input],
                        'id' : [6969], 
                        'original_title': ["Dummy The Movie"], 
                        'overview': [overview_input],
                        'vote_average': [9.9], 
                        'vote_count': [1234], 
                        'cast': [cast_input], 
                        'crew': [' '],
                        'director': [Director_input]})

movies = pd.concat([new_row,movies], ignore_index=True)


# Removing movies with under 500 rating voted
movies = movies.copy().loc[movies['vote_count'] >= 500]

#================================= TF-IDF FOR PLOT =======================================

# TF-IDF Vectorizer. Remove stop words ('on', 'a', 'and', etc.)
TFIDF = TfidfVectorizer(stop_words='english')

#If not available fill with '' (empty string)
movies['overview'] = movies['overview'].fillna('')

#Create matrix with TFIDF
tfidf_matrix = TFIDF.fit_transform(movies['overview'])

#calculate the cosine similaritys 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Make a list that shows movie titles with their corresponding index numbers in reverse order. And remove dupes
indices = pd.Series(movies.index, index=movies['original_title']).drop_duplicates()


# Function that takes in own movie as input and outputs the 5 most similar movies
def get_recommendations(plot, cosine_sim):
    
    # Get the index of own movie
    idx = indices[plot]

    # Pairwsie compare similarity of all movies with own moive
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies according to how similar they are to each other.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Find indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the 5 most similar movies
    return movie_indices

#============================== CHARACTERISTICS: Credits, Genres and Keywords Based Recommender ========================================

# changes all words to small letters and removes spaces from names.
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:

        #Check if director exists. If not return empty
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
# Add clean_data to our features.
features = ['cast', 'director', 'genres']

for feature in features:
    movies[feature] = movies[feature].apply(clean_data)    

# Make soup of the features
def create_soup(x):
    # Own movie acts wierd and has to be this way (fast fix)
    if x['id']==6969:
        return ''.join(x['cast']) + ' ' + x['director'] + ' ' + ''.join(x['genres'])
    else: 
        return ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

movies['soup'] = movies.apply(create_soup, axis=1)   

# create the count matrix
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])   

#calculate the cosine similaritys
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our Data Frame and make reverse mapping like before
df2 = movies.reset_index()
indices = pd.Series(df2.index, index=df2['original_title'])


#==================================== PRINT STUFF ==================================================

#Get vote average from the 5 most similar movies
plot_indices = movies['vote_average'].iloc[get_recommendations('Dummy The Movie', cosine_sim)]
features_indices =  movies['vote_average'].iloc[get_recommendations('Dummy The Movie', cosine_sim2)]

#Most similar movie for each
plot_sim = movies['original_title'].iloc[get_recommendations('Dummy The Movie', cosine_sim)[0]]
features_sim = movies['original_title'].iloc[get_recommendations('Dummy The Movie', cosine_sim2)[0]]

# weights for movie rating
#weights = [8,4,2,1,1]
weights = [5,2,1,1,1] #best result

#Get a weighted score for each
plot_ave= weighted_avg_m2(plot_indices, weights)
feature_ave = weighted_avg_m2(features_indices, weights)

# Check if some is left empty in gui, if so, only use the score of usable one
if [genres_input, cast_input, Director_input] == ["","",""]:
    weights_avg = [(1), (0)]
elif overview_input == "":
    weights_avg = [(0), (1)]
else:
    weights_avg = [(3/5), (2/5)]    #Plot is worth 60%, features is worth 40%

#get the final weighted rating
total_ave = weighted_avg_m2([plot_ave,feature_ave], weights_avg)

#Test prints
# print(plot_ave)
# print(feature_ave)
# print(total_ave)

# print(movies['soup'][0])
# print(movies['overview'][0])
# print(plot_indices)
# print(features_indices)

#==================== GUI ==================================
layout = [[sg.Text('Movie rating predictor 2000')],
      [sg.Text('You got the score: '), sg.Text(total_ave)],
      [sg.Text('Your movie was similar to:')],
      [sg.Text(plot_sim), sg.Text('&'), sg.Text(features_sim)],
      [sg.Ok()]]

window = sg.Window('Movie rating predictor 2000', layout)

event, values = window.read()
window.close()


# Keywords: Science Fiction, Adventure, Fantasy, Comedy, Action, drama, horror, Animation, Thriller, Romance, crime, Romance, Mystery