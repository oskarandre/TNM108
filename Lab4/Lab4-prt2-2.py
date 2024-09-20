from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

from sklearn.model_selection import GridSearchCV


categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)


text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

text_clf.fit(twenty_train.data, twenty_train.target)


docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ",np.mean(predicted == twenty_test.target))


# training SVM classifier

text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == twenty_test.target))


print(metrics.classification_report(twenty_test.target, predicted,
 target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))


#--------- GRID SEARCH-------

parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-2, 1e-3),
}


gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
 print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


print(gs_clf.cv_results_)
