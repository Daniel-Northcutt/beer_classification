# Wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as stats


import json
import wrangle as w
import wrangle2

# Visualizing
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.model_selection import learning_curve

from sklearn.cluster import KMeans


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

################
df = w.acquire_beer_classification()
df = w.prepare_beer(df)
train, validate, test = w.split_data(df)
train, validate, test = w.scale(train, validate, test)
train['baseline_prediction'] = 0
baseline_accuracy = .714
#print(f'The baseline accuracy is: {baseline_accuracy:.2%}')
#####
df = wrangle2.acquire_beer_classification()
df = wrangle2.prepare_beer(df)
train, validate, test = wrangle2.split_data(df)
train, validate, test = wrangle2.scale(train, validate, test)
aseline_accuracy = .85
#print(f'The baseline accuracy is: {baseline_accuracy:.2%}')

##################
def feature_forest():
    ''' Random Forest on imputed Boolean values pulled via str.get from 
    the style column'''
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)

    X_column = ['sour', 'fruit_des',
    'wheat_des', 'smoke_des', 'chili_des', 'mead_des', 'lager_des', 'BBL',
    'american_hop', 'piney_flavor', 'belgian', 'imperial', 'honey',
    'esters', 'bitterness', 'oak', 'scaled_ibu',
    'scaled_srm', 'scaled_abv']
    y_column = 'style_collapsed'


    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')

    print(f'The precision is: {precision:.2%}')
    print(f'The recall is: {recall:.2%}')

##############

def target_forest():
    ''' Random Forest on scaled main 3 features 
    (scaled performed better than nonscalled)'''

    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)
    #train, validate, test = w.scale(train, validate, test)

    X_column = ['scaled_ibu',
        'scaled_srm', 'scaled_abv']

    y_column = 'style_collapsed'

    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')

    print(f'The precision is: {precision:.2%}')
    print(f'The recall is: {recall:.2%}')

##############
def target_forest_extra_target_variable(): 
    df = wrangle2.acquire_beer_classification()
    df = wrangle2.prepare_beer(df)
    train, validate, test = wrangle2.split_data(df)
    train, validate, test = wrangle2.scale(train, validate, test)
    X_column = ['ibu',
    'srm', 'abv']

    y_column = 'style_collapsed'

    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')

    print(f'The precision is: {precision:.2%}')
    print(f'The recall is: {recall:.2%}')

##########
def all_features(): 
    df = wrangle2.acquire_beer_classification()
    df = wrangle2.prepare_beer(df)
    train, validate, test = wrangle2.split_data(df)
    train, validate, test = wrangle2.scale(train, validate, test)

    X_column = ['ibu',
        'srm', 'abv', 'sour', 'fruit_des',
        'wheat_des', 'smoke_des', 'chili_des', 'mead_des', 'lager_des', 'BBL',
        'american_hop', 'piney_flavor', 'belgian', 'imperial', 'honey',
        'esters', 'bitterness', 'oak']

    y_column = 'style_collapsed'

    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')

    print(f'The precision is: {precision:.2%}')
    print(f'The recall is: {recall:.2%}')
############

### THE TEST ###

def test_time(): 
    df = wrangle2.acquire_beer_classification()
    df = wrangle2.prepare_beer(df)
    train, validate, test = wrangle2.split_data(df)
    train, validate, test = wrangle2.scale(train, validate, test)
    X_column = ['ibu',
    'srm', 'abv']

    y_column = 'style_collapsed'

    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    #print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    #print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')
    print()
    print(f'The test score is: {forest_time.score(X_test, y_test):.2%}\n')
    #print(f'The precision is: {precision:.2%}')
    #print(f'The recall is: {recall:.2%}')

###############
def test_time_less_features(): 
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = wrangle2.split_data(df)
    train, validate, test = wrangle2.scale(train, validate, test)
    X_column = ['scaled_ibu',
        'scaled_srm', 'scaled_abv']

    y_column = 'style_collapsed'

    # Specifying our train, validate, and test for our models

    X_train = train[X_column]
    y_train= train[y_column]

    X_validate = validate[X_column]
    y_validate= validate[y_column]

    X_test = test[X_column]
    y_test= test[y_column]


    # Create and fit the model
    forest_time = RandomForestClassifier(min_samples_leaf = 1, max_depth = 12, random_state= 123)
    forest_time.fit(X_train, y_train)
    forest_time_pred = forest_time.predict(X_train)

    #defined our parameters for precision and recall
    TP = confusion_matrix(y_train, forest_time_pred)[0][0]
    FP = confusion_matrix(y_train, forest_time_pred)[0][1]
    FN = confusion_matrix(y_train, forest_time_pred)[1][0]
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    #print(f'training score: {forest_time.score(X_train, y_train):.2%}\n')
    #print(f'validate score: {forest_time.score(X_validate, y_validate):.2%}\n')
    #print('The difference from Baseline: {:.2f}%'.format(d))
    print(f'The baseline accuracy is: {baseline_accuracy:.2%}\n')
    print()
    print(f'The test score is: {forest_time.score(X_test, y_test):.2%}\n')
    #print(f'The precision is: {precision:.2%}')
    #print(f'The recall is: {recall:.2%}')