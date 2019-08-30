from sklearn.metrics import *
from sklearn.metrics import accuracy_score


def get_classfication_report(model, X_test, y_test):
    print('\n Classification Report :\n')
    print(classification_report(y_test, model.predict(X_test)))


def get_confusion_matrix(model, X_test, y_test):
    print('Confusion Matrix :\n')
    print(confusion_matrix(y_test, model.predict(X_test)))
