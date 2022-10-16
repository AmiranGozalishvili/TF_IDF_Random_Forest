from fastapi import FastAPI
import nltk
import pandas as pd
from autocorrect import Speller
from nltk.corpus import stopwords
from tqdm.notebook import tqdm_notebook

nltk.download(info_or_id='stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import spacy
import re
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, plot_confusion_matrix, \
    plot_roc_curve
import pickle

from data import load_data

dataset = load_data()

from datapreprocess import cont_expand, clean_slang_terms

from stemm_and_lemmentize import stemming, lemmentization

'''
GOAL: Build a Random Forest model to predict Recommended vs Not-Recommended
classification using text from customer reviews.
'''

app = FastAPI()

# lemmatize reviews with spacy / dockerfile install pip spacy
nlp_lem = spacy.load('en_core_web_sm')

# initiate tqdm for pandas.apply() functions
tqdm_notebook.pandas()

# expand notebook display options for dataframes
pd.set_option('display.max_colwidth', 200)
pd.options.display.max_columns = 999
pd.options.display.max_rows = 300

# Text Pre-processing
# drop any rows with missing Review Text
dataset.dropna(axis=0, how='any', subset=['Review Text'], inplace=True)
# make all characters uniformly lowercase in order to ignoring case
dataset['Review Text'] = dataset['Review Text'].apply(lambda x: x.lower())

# expand contractations
dataset['Review Text'] = dataset['Review Text'].progress_apply(cont_expand)

# clean slang
dataset['Review Text'] = dataset['Review Text'].progress_apply(clean_slang_terms)

########

# autocorrect any misspelled words (takes about ~20 minutes)
spell_check = Speller(lang='en')
dataset['Review Text'] = dataset['Review Text'].progress_apply(lambda x: spell_check(str(x)))


# remove everything other than letters and apostrophes, and replace the removed with a space
# (^ characher indicates to complement the input)
dataset['Review Text'] = dataset['Review Text'].apply(lambda x: re.sub(pattern=r'[^a-z\']', repl=' ', string=x))

# remove any extra whitespace within the text
dataset['Review Text'] = dataset['Review Text'].apply(lambda x: re.sub(pattern=r'\s{2,}', repl=' ', string=x))

# manually edit the list of stopwords by removing certain important words
to_remove = ["not", "no", "but", "own", "more", "over", "under", "most", "again"]
new_stopwords = set(stopwords.words('english')) - set(to_remove)

# keeping root words with a stemming algorithm (takes about ~5 minutes)
dataset['Review Text'] = dataset['Review Text'].progress_apply(stemming)

# apply lemmentization (takes about ~10 minutes)
dataset['Review Text'] = dataset['Review Text'].progress_apply(lemmentization)

'''define a random state for reproducible results'''
random_state = 42

'''upsample the minority class ('not-recommended') by duplicating random samples
1 = recommended
0 = not-recommended'''

up_sample = resample(dataset[dataset['Recommended IND'] == 0],
                     replace=True,  # sample with replacement
                     n_samples=10000,  # to match majority class
                     random_state=random_state)  # set reproducible results
# combine the up-sampled minority class with the rest of the data
dataset_balanced = pd.concat([dataset[dataset['Recommended IND'] != 0], up_sample])
print('Before:')
print(dataset['Recommended IND'].value_counts(), '\n')

print('After:')
print(dataset_balanced['Recommended IND'].value_counts())

'''
TF-IDF 
'''

# define the corpus
corpus = dataset['Review Text']

# max_feature helps condense the sparse matrix by keeping only the top n number of words in the corpus' vocabulary
# in this case, the 1500 most frequent words are kept
cv = TfidfVectorizer(max_features=1500)

# create the matrix
X = cv.fit_transform(corpus).toarray()

# adding the dependent variable [recommended (1) vs not-recommended (0)]
y = dataset['Recommended IND'].values

'''Train/Test Split'''

# splitting the dataset into an 80% training and 20% test set

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=0,
                                                    shuffle=True)

'''Grid-Search'''

parameters = [{'criterion': ['gini', 'entropy'],
               'n_estimators': [1, 10, 50, 100, 300],
               # 'max_depth': [2, 3, 4, 5],
               # "min_samples_leaf":[1, 5, 10],
               # "min_samples_split" : [2, 4, 10, 12, 16]
               }]
# determine which combination of parameters leads to the best results
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy')

grid = grid_search.fit(X_train, y_train)

print('Best accuracy: {}'.format(grid.best_score_))
print('Best parameters: {}'.format(grid.best_params_))

'''Random Forest'''

# fit the optimized Random Forest model onto the training set
classifier = RandomForestClassifier(n_estimators=50,
                                    criterion='gini',
                                    random_state=random_state)
classifier.fit(X_train, y_train)

# predict the test results
y_pred = classifier.predict(X_test)

'''Model Metrics'''

# metrics
print("Accuracy:".ljust(18), round(accuracy_score(y_test, y_pred), 2))
print("Recall Score:".ljust(18), round(recall_score(y_test, y_pred), 2))
print("Percision Score:".ljust(18), round(precision_score(y_test, y_pred), 2))
print("F1-Score:".ljust(18), round(f1_score(y_test, y_pred), 2))

# confusion matrix
plot_confusion_matrix(classifier, X_test, y_test)

# ROC curve
plot_roc_curve(classifier, X_test, y_test)

''' save the model to disk'''

filename = "optimized_random_forest.sav"
pickle.dump(classifier, open(filename, 'wb'))

@app.get('/')
def get_root():
	return {'message': 'Welcome to TQ-IDF and Random Forest API'}

@app.get('/TQ-IDF_RandomForest')
def TQIDF_RandomForest(text):
    text_pred = classifier.predict(text)
    return (print("Accuracy:".ljust(18), round(accuracy_score(y_test, text_pred), 2)),
            print("Recall Score:".ljust(18), round(recall_score(y_test, text_pred), 2)),
            print("Percision Score:".ljust(18), round(precision_score(y_test, text_pred), 2)),
            print("F1-Score:".ljust(18), round(f1_score(y_test, text_pred), 2)))

