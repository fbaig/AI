# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

#from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

# function to extract features
def extract_features(dataset, _stop_words='', vtype='count'):
    if vtype == 'count':
        print ('Extracting features using BOW')
        vectorizer = CountVectorizer(stop_words=_stop_words)
    elif vtype == 'tfidf':
        print ('Extracting features using TF-IDF')
        vectorizer = TfidfVectorizer(stop_words=_stop_words)
    else:
        sys.exit('Invalid feature extractor')

    vectors_train = vectorizer.fit_transform(dataset)
    return vectorizer, vectors_train

def train_classifier(train_vectors, train_dataset, ctype='nb'):
    if ctype == 'nb':
        print ('Training Naive Bayes Classifier')
        classifier = MultinomialNB(alpha=0.01)
        classifier.fit(train_vectors, train_dataset.target)
    elif ctype == 'svm':
        print ('Training SVM Classifier')
        sys.exit('SVM not implemented')
    elif ctype == 'lg':
        print ('Training Logistic Regression Classifier')
        sys.exit('Logistic not implemented')
    elif ctype == 'rf':
        print ('Training Random Forest Classifier')
        sys.exit('Random Forest not implemented')
        
    return classifier
    
#def preprocess(s):
    
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# # parse commandline arguments
# op = OptionParser()
# op.add_option("--lsa",
#               dest="n_components", type="int",
#               help="Preprocess documents with latent semantic analysis.")
# op.add_option("--no-minibatch",
#               action="store_false", dest="minibatch", default=True,
#               help="Use ordinary k-means algorithm (in batch mode).")
# op.add_option("--no-idf",
#               action="store_false", dest="use_idf", default=True,
#               help="Disable Inverse Document Frequency feature weighting.")
# op.add_option("--use-hashing",
#               action="store_true", default=False,
#               help="Use a hashing feature vectorizer")
# op.add_option("--n-features", type=int, default=10000,
#               help="Maximum number of features (dimensions)"
#                    " to extract from text.")
# op.add_option("--verbose",
#               action="store_true", dest="verbose", default=False,
#               help="Print progress reports inside k-means algorithm.")

# print(__doc__)
# op.print_help()

# (opts, args) = op.parse_args()
# if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)

############### Load Training Dataset #############
train_data_path = '/home/bmi-baig/Downloads/cse537-AI/AI/a3/selected_20NewsGroup/Training'
test_data_path = '/home/bmi-baig/Downloads/cse537-AI/AI/a3/selected_20NewsGroup/Test'
print ('Loading dataset from path: %s' % train_data_path)

train_dataset = load_files(container_path=train_data_path, encoding='latin-1')

print("%d documents" % len(train_dataset.data))
print("%d categories" % len(train_dataset.target_names))
print()
labels = train_dataset.target

# just to test
# 'category integer id' for each sample in data is stored in 'target' attribute as list
# print (train_dataset.data[1100])
# print (train_dataset.target_names[train_dataset.target[1100]])

# ############### Extract features ##################

# vectorizer, vectors = extract_features(dataset=train_dataset.data,
#                                  _stop_words='english',
#                                  vtype='tfidf') 
# ############## Train Naive Bayes (Classifier) #####
# classifier = train_classifier(train_vectors=vectors,
#                               train_dataset=train_dataset,
#                               ctype='nb')

class Config:
    def __init__(self, classifier, vectorizer='count', ngram=1, stopwords=None, lowercase=False):
        self.ngram = ngram
        self.stopwords = stopwords
        self.lowercase = lowercase
        self.classifier = classifier
        
        self.transformer = TfidfTransformer()
        # if vectorizer == 'count':
        #     self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, ngram))
        # elif vectorizer == 'tfidf':
        #     self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, ngram))
        if vectorizer == 'count':
            self.vectorizer = CountVectorizer(lowercase=lowercase, stop_words=stopwords, ngram_range=(1, ngram))
        elif vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(lowercase=lowercase, stop_words=stopwords, ngram_range=(1, ngram))
        else:
            self.vectorizer = None

    def __repr__(self):
        ret = 'n-gram: %d\nvectorizer: %s\n' % (self.ngram, type(self.vectorizer).__name__ )
        classifier_str = type(self.classifier).__name__
        if classifier_str == 'LogisticRegression':
            classifier_str += '(iterations: %s)' % str(self.classifier.max_iter)
        elif classifier_str == 'SVC':
            classifier_str += '(kernel: %s)' % str(self.classifier.kernel)
        elif classifier_str == 'RandomForestClassifier':
            classifier_str += '(# trees: %s, # features: %s)' % (str(self.classifier.n_estimators), str(self.classifier.max_features))

        ret += 'classifier: %s\n' % classifier_str
        return ret

# classifier models
'''
Naive Bayes
'''
NB = MultinomialNB()

'''
Logistic Regression
Hyperparameters:
    - Regularization constant
    - iterations count
'''
from sklearn.linear_model import LogisticRegression
_reg_constant = 1
_iter_count = 2
LR = LogisticRegression(C=_reg_constant, max_iter=_iter_count)

'''
SVM
Hyperparameters:
    - Regularization constant
    - Kernels (Linear, Polynomial, RBF)
'''
from sklearn import svm
_reg_constant = 2
_kernel = 'linear' # rbf, linear, poly
SVM = svm.SVC(C=_reg_constant, kernel=_kernel, degree=3)

'''
Random Forest
Hyperparameters:
    - Number of trees
    - Number of features
'''
from sklearn.ensemble import RandomForestClassifier
_trees_count = 10
_features_count = 8
RF = RandomForestClassifier(n_estimators=_trees_count, max_features=2)

# configurations
BASELINE_NB_UNI = Config(classifier=NB)
BASELINE_NB_BI = Config(classifier=NB, ngram=2)

BASELINE_LR_UNI = Config(classifier=LR)
BASELINE_LR_BI = Config(classifier=LR, ngram=2)

BASELINE_SVM_UNI = Config(classifier=SVM)
BASELINE_SVM_BI = Config(classifier=SVM, ngram=2)

BASELINE_RF_UNI = Config(classifier=RF)
BASELINE_RF_BI = Config(classifier=RF, ngram=2)

# configuration to use
conf = BASELINE_RF_BI

print (conf)

classifier = Pipeline([('vectorizer', conf.vectorizer),
                       ('transformer', conf.transformer),
                       ('classifier', conf.classifier)])

_ = classifier.fit(train_dataset.data, train_dataset.target)
############## Test Data ##########################
test_dataset = load_files(container_path=test_data_path, encoding='latin-1')
# vectors_test = vectorizer.transform(test_dataset.data)

my_test_set = ['nfl is boring ...', 'christ is a prophet of God', 'the resident doctor was not on duty that day']
############## Predict ############################
print ('Predicting Test data')

predict1 = classifier.predict(my_test_set)
print (train_dataset.target_names)
print (predict1)

predict = classifier.predict(test_dataset.data)

import numpy as np
print ('Accuracy: %f' % np.mean(predict == test_dataset.target))
print ('F1-score: %f' % metrics.f1_score(test_dataset.target, predict, average='macro'))





# print("Extracting features from the training dataset using a sparse vectorizer")
# t0 = time()
# if opts.use_hashing:
#     if opts.use_idf:
#         # Perform an IDF normalization on the output of HashingVectorizer
#         hasher = HashingVectorizer(n_features=opts.n_features,
#                                    stop_words='english', non_negative=True,
#                                    norm=None, binary=False)
#         vectorizer = make_pipeline(hasher, TfidfTransformer())
#     else:
#         vectorizer = HashingVectorizer(n_features=opts.n_features,
#                                        stop_words='english',
#                                        non_negative=False, norm='l2',
#                                        binary=False)
# else:
#     vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
#                                  min_df=2, stop_words='english',
#                                  use_idf=opts.use_idf)
# X = vectorizer.fit_transform(dataset.data)

# print("done in %fs" % (time() - t0))
# print("n_samples: %d, n_features: %d" % X.shape)
# print()

# if opts.n_components:
#     print("Performing dimensionality reduction using LSA")
#     t0 = time()
#     # Vectorizer results are normalized, which makes KMeans behave as
#     # spherical k-means for better results. Since LSA/SVD results are
#     # not normalized, we have to redo the normalization.
#     svd = TruncatedSVD(opts.n_components)
#     normalizer = Normalizer(copy=False)
#     lsa = make_pipeline(svd, normalizer)

#     X = lsa.fit_transform(X)

#     print("done in %fs" % (time() - t0))

#     explained_variance = svd.explained_variance_ratio_.sum()
#     print("Explained variance of the SVD step: {}%".format(
#         int(explained_variance * 100)))

#     print()
