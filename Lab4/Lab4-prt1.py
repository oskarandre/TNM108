d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB


vectorizer = CountVectorizer()

print(vectorizer)

my_stop_words={"the","is"}
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)

print("------- Vocabulary --------")
print(vectorizer.vocabulary)
print("\n ------------- Stop-words -------------")
print(vectorizer.stop_words)

smatrix = vectorizer.transform(Z)

print("\n--------- sparse matrix ----------- ")
print(smatrix)

matrix = smatrix.todense()
print("\n--------- sparse matrix (Dense format) ----------- ")
print(matrix)

tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# print idf values
feature_names = vectorizer.get_feature_names_out()
import pandas as pd
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])

print("\n--------- sorted in ascending order ----------- ")
print(df_idf)

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

print("\n--------- sorted in descending order ----------- ")
print(df)

# ===========  DOCUMENT SIMILARITY ===========

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)

print("\n--------- TF-IDF matrix ----------- ")
print(tfidf_matrix.shape)

cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print("\n--------- Cosine similarity ----------- ")
print(cos_similarity)

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0,2])
print("\n--------- Angle between 1st and 3rd doc ----------- ")
print(math.degrees(angle_in_radians))

# ================== Classifying text =====================

data = fetch_20newsgroups()
data.target_names

my_categories = ['rec.sport.baseball','rec.motorcycles','sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train', categories=my_categories)
test = fetch_20newsgroups(subset='test', categories=my_categories)

print("\n--------- Length of the training data ----------- ")
print(len(train.data))
print("\n--------- Length of test data ----------- ")
print(len(test.data))
print("\n--------- Sample ----------- ")
print(train.data[9])


cv = CountVectorizer()
X_train_counts=cv.fit_transform(train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)
model = MultinomialNB().fit(X_train_tfidf, train.target)

print("\n--------- TF-IDF MATRIX with naive beyers model ----------- ")
docs_new = ['Pierangelo is a really good baseball player','Maria rides her motorcycle', 
'OpenGL on the GPU is fast', 'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
 print('%r => %s' % (doc, train.target_names[category]))
