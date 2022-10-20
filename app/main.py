import os
import pickle

from fastapi import FastAPI

from commands import preprocess, tf_idf, grid_search, random_forest

app = FastAPI()

path = "optimized_random_forest.sav"
filename = "optimized_random_forest.sav"

''' check if saved file exists'''


def check_file():
    if os.path.exists(path):
        print("path exists")
        classifier = pickle.load(open(filename, "rb"))


    else:
        preprocess()

        X_train, X_test, y_train, y_test = tf_idf()

        grid_search(X_train, y_train)

        y_pred, classifier = random_forest(X_train, X_test, y_train, y_test)

        print("saving model")
        # save the entire model
        pickle.dump(classifier, open(filename, 'wb'))

    return classifier


check_file()

@app.get('/')
def get_root():
    return {'message': 'Welcome to Tf-IDF and Random Forest API'}

@app.get('/TF-IDF_RandomForest')
def TfIDF_RandomForest(text):

    return
