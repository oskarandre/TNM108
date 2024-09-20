
from sklearn.base import np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_files
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

print("\n =================== START ============== \n")

moviedir = r'C:/Users/oskar/Downloads/movie_reviews'

# loading all files. 
movie = load_files(moviedir, shuffle=True)

# print(len(movie.data))

# # target names ("classes") are automatically generated from subfolder names
# print(movie.target_names)

# # First file seems to be about a Schwarzenegger movie. 
# print(movie.data[0][:500])

# # first file is in "neg" folder
# print(movie.filenames[0])

# # first file is a negative review and is mapped to 0 index 'neg' in target_names
# print(movie.target[0])


docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, 
        test_size = 0.20, random_state = 12)


# initialize CountVectorizer
movieVzer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000) # use top 3000 words only. 78.25% acc.
# movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. Higher accuracy

# fit and tranform using training text 
docs_train_counts = movieVzer.fit_transform(docs_train)


docs_train_counts.shape
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)
docs_train_tfidf.shape

#--------- Test data---------------

# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
print(sklearn.metrics.accuracy_score(y_test, y_pred))
print("\n")

cm = confusion_matrix(y_test, y_pred)

#----------- Trying the classifier on fake movie reviews ---------
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)         # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

pred = clf.predict(reviews_new_tfidf)

print("Accuracy ",np.mean(pred == movie.target_names))

for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

print("\n")
# create pipeline for vectorizer, tfidf, and classifier
# text_clf = Pipeline([
#  ('vect', CountVectorizer()),
#  ('tfidf', TfidfTransformer()),
#  ('clf', MultinomialNB())
# ])
text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier())
])



text_clf.fit(docs_train, y_train)  

# train on training set
# use grid search to find best parameters
parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3, 1e-4),
              'clf__random_state':(42, 43),
              }

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(docs_train, y_train)

print(gs_clf.best_score_)
print("\n")

print(gs_clf.best_params_)
print("\n")


# use best parameters to predict
y_pred = gs_clf.predict(docs_test)
print(sklearn.metrics.accuracy_score(y_test, y_pred))
print("\n")
#print("Accuracy ",np.mean(y_pred == movie.target_names))

# use best parameters to predict new reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']
pred = gs_clf.predict(reviews_new)
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

