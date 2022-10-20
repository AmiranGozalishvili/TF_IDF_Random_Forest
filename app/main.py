import pickle

from fastapi import FastAPI

from commands import preprocess, tf_idf, grid_search, random_forest
from commands import scores

app = FastAPI()

path = "optimized_random_forest.sav"

preprocess()

X_train, X_test, y_train, y_test = tf_idf()

grid_search(X_train, y_train)

y_pred, classifier = random_forest(X_train, X_test, y_train, y_test)

print("saving model")
# save the entire model
pickle.dump(classifier, open(path, 'wb'))

json_file = scores(y_test, y_pred)


@app.get('/')
def get_root():
    return {'message': 'Welcome to Tf-IDF and Random Forest API'}


@app.get('/TF-IDF_RandomForest')
def TfIDF_RandomForest():
    return json_file
