# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

#from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files

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
print ('Loading dataset from path: %s' % train_data_path)

train_dataset = load_files(container_path=train_data_path, encoding='latin-1')

print("%d documents" % len(train_dataset.data))
print("%d categories" % len(train_dataset.target_names))
print()

labels = train_dataset.target

print (train_dataset.target_names)

############### Extract features ##################
t0 = time()
print ('Extracting features using TF-IDF')
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_dataset.data)
print("done in %fs" % (time() - t0))


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
