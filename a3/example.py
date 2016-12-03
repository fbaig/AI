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


############ Classifier Models #############
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
#LR = LogisticRegression(C=_reg_constant, max_iter=_iter_count)
LR = LogisticRegression()

'''
SVM
Hyperparameters:
    - Regularization constant
    - Kernels (Linear, Polynomial, RBF)
'''
from sklearn import svm
_reg_constant = 2
_kernel = 'linear' # rbf, linear, poly
SVM = svm.SVC(kernel=_kernel)
#SVM = svm.SVC(C=_reg_constant, kernel=_kernel, degree=3)
#SVM = svm.SVC()

'''
Random Forest
Hyperparameters:
    - Number of trees
    - Number of features
'''
from sklearn.ensemble import RandomForestClassifier
_trees_count = 10
_features_count = 8
#RF = RandomForestClassifier(n_estimators=_trees_count, max_features=2)
RF = RandomForestClassifier()



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
    
def preprocess(s):
    words = s.lower().split()
    # using nltk remove stop words
    from nltk.corpus import stopwords
    words_wo_stopwords = [word for word in words if word not in stopwords.words('english')]
    # remove stemming

    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    words_to_process = [stemmer.stem(word) for word in words_wo_stopwords]

    return ' '.join(words_to_process)
    #print (words_to_process)
    

class Config:
    def __init__(self, classifier, vectorizer='count', ngram=1, stopwords=None, lowercase=False, preprocessor=None):
        self.ngram = ngram
        self.stopwords = stopwords
        self.lowercase = lowercase
        self.classifier = classifier
        
        self.transformer = TfidfTransformer()
        if vectorizer == 'count':
            self.vectorizer = CountVectorizer(lowercase=lowercase, stop_words=stopwords, ngram_range=(ngram, ngram), preprocessor=preprocessor)
        elif vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(lowercase=lowercase, stop_words=stopwords, ngram_range=(ngram, ngram), preprocessor=preprocessor)
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

        ret += 'classifier: %s' % classifier_str
        return ret

# configurations
BASELINE_NB_UNI = Config(classifier=NB)
BASELINE_NB_BI = Config(classifier=NB, ngram=2)

BASELINE_LR_UNI = Config(classifier=LR)
BASELINE_LR_BI = Config(classifier=LR, ngram=2)

BASELINE_SVM_UNI = Config(classifier=SVM)
BASELINE_SVM_BI = Config(classifier=SVM, ngram=2)

BASELINE_RF_UNI = Config(classifier=RF)
BASELINE_RF_BI = Config(classifier=RF, ngram=2)

def plot(datasize, f1values_nb, f1values_lr, f1values_svm, f1values_rf):
    import matplotlib.pyplot as plt
    plt.plot(datasize, f1values_nb, label='NB', linewidth=1.5, marker='s')
    plt.plot(datasize, f1values_lr, label='LR', linestyle='--', marker='o', linewidth=1.5)
    plt.plot(datasize, f1values_svm, label='SVM', linestyle=':', marker='+', linewidth=1.5)
    plt.plot(datasize, f1values_rf, label='RF', linestyle='-.', marker='*', linewidth=1.5)
    
    plt.xlabel('Data Size (# of documents)')
    plt.ylabel('F1 Score')
    plt.title('Classifier Performance with Unigram Representation')
    plt.legend()
    plt.show()

def classify_using(train_data_path, train_size, conf, test_data_path):

    train_dataset = load_files(container_path=train_data_path, encoding='latin-1')
    # maunally remove headers from training dataset
    for i, doc in enumerate(train_dataset.data):
        if i > train_size:
            train_dataset.data[i] = ''
            continue
        preprocessed = doc.split('\n')
        # Assuming 'Lines: <int>' is the last line in header
        body_size = 0
        for line in preprocessed:
            if line.startswith('Lines'):
                body_size = int(line.split(':')[1])+1
            elif body_size > 0:
                break
            continue

        preprocessed = preprocessed[-body_size:]
        train_dataset.data[i] = '\n'.join(preprocessed)

    classifier = Pipeline([('vectorizer', conf.vectorizer),
                           ('transformer', conf.transformer),
                           ('classifier', conf.classifier)])
    
    _ = classifier.fit(train_dataset.data, train_dataset.target)
    ############## Test Data ##########################
    test_dataset = load_files(container_path=test_data_path, encoding='latin-1')
    # vectors_test = vectorizer.transform(test_dataset.data)
    ############## Predict ############################
    #print ('Predicting Test data')
    #my_test_set = ['nfl is boring ...', 'christ is a prophet of God', 'the resident doctor was not on duty that day']
    #predict1 = classifier.predict(my_test_set)
    #print (labels)
    #print (predict1)
    
    predict = classifier.predict(test_dataset.data)
    
    #import numpy as np
    #print ('Accuracy: %f' % np.mean(predict == test_dataset.target))
    return metrics.f1_score(test_dataset.target, predict, average='macro')
        

def baseline():
    baseline_configs_uni = [BASELINE_NB_UNI, BASELINE_LR_UNI, BASELINE_SVM_UNI, BASELINE_RF_UNI]
    baseline_configs_bi = [BASELINE_NB_BI, BASELINE_LR_BI, BASELINE_SVM_BI, BASELINE_RF_BI]
    

    ############### Load Training Dataset #############
    train_dataset = load_files(container_path=train_data_path, encoding='latin-1')
    
    dataset_sizes = []
    for i in range(100, len(train_dataset.data), 100):
        dataset_sizes.append(i)

    dataset_sizes.append(len(train_dataset.data))
    print (dataset_sizes)
    
    #print (train_dataset.data[17])
    #print("%d documents" % len(train_dataset.data))
    #print("%d categories" % len(train_dataset.target_names))
    #labels = train_dataset.target

    classifier_f1_scores = []
    for conf in baseline_configs_uni:
        print ('Dataset path: %s' % test_data_path)
        print (conf)
        f1_scores = []
        for train_size in dataset_sizes:
            f1_score = classify_using(train_data_path, train_size, conf, test_data_path)
            f1_scores.append(f1_score)
            print ('F1-score: %f' % f1_score)
            
        classifier_f1_scores.append(f1_scores)
        #break

    plot(dataset_sizes, classifier_f1_scores[0], classifier_f1_scores[1], classifier_f1_scores[2], classifier_f1_scores[3])

if __name__ == '__main__':


    if len(sys.argv) == 1:
        train_data_path = '/home/bmi-baig/Downloads/cse537-AI/AI/a3/selected_20NewsGroup/Training'
        test_data_path = '/home/bmi-baig/Downloads/cse537-AI/AI/a3/selected_20NewsGroup/Test'
    else:
        train_data_path = sys.argv[1]
        test_data_path = sys.argv[2]
        
    #SVM = svm.SVC(C=_reg_constant, kernel=_kernel, degree=3)
    MBC = Config(classifier=svm.SVC(C=_reg_constant, kernel='linear', degree=100),
                 ngram=1,
                 lowercase=True,
                 #stopwords='english',
                 #preprocessor=preprocess,
                 vectorizer='tfidf')
    
    train_dataset = load_files(container_path=train_data_path, encoding='latin-1')
    
    print (classify_using(train_data_path, len(train_dataset.data), MBC , test_data_path))

    # mbc_config = Config(classifier=SVM, vectorizer='tfidf')

    #print (preprocess('this is the first line and it is indeed the first line'))
    # sys.exit('')
    
    

