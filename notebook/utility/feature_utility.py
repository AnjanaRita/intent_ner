import os
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = '/home/genesis/.vectors/use_3/'
if not os.path.isdir(PATH):
    PATH = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'

use_embedding = hub.Module(PATH, trainable=False)


def saver(model, filename):
    file_name = os.path.join('model', filename)
    with open(file_name, 'wb') as fp:
        pickle.dump(model, fp)


def use_vectorizer(train_data):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        embedding = session.run(use_embedding(train_data))
    return embedding


def tfidf_vectorizer(train_data, test_data):
    tf_idf = TfidfVectorizer()
    X_train = tf_idf.fit_transform(train_data)
    X_test = tf_idf.transform(test_data)
    saver(tf_idf, 'tfidf.pkl')
    return X_train, X_test


def featurized_data(dataset, mode='tf-idf'):
    train_data, test_data = dataset.train, dataset.test
    y_train, y_test = train_data.intent.values, test_data.intent.values
    X_train, X_test = train_data.text.values, test_data.text.values
    if mode == 'tf-idf':
        X_train, X_test = tfidf_vectorizer(X_train, X_test)
    else:
        X_train, X_test = use_vectorizer(X_train), use_vectorizer(X_test)
    return X_train, X_test, y_train, y_test
