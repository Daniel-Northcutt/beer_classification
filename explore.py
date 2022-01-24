# Wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

import scipy.stats as stats
########
def style_distribution():
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x='style_collapsed')
    plt.xticks(rotation='vertical')
    plt.title('Distribution of styles in dataset');

########

def relplot1():
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    
    style_counts = df['style_collapsed'].value_counts()
    to_remove_counts = style_counts[style_counts <=20].index
    df = df[~df.style_collapsed.isin(to_remove_counts)]
    
    sns.relplot(data=df, x='ibu', y='srm', hue='style_collapsed');
    plt.title("Relation of SRM and IBU")
    plt.show()
    ######
    sns.relplot(data=df, x='ibu', y='abv', hue='style_collapsed');
    plt.title("Relation of IBU and ABV")
    plt.show()
    ######
    sns.relplot(data=df, x='srm', y='abv', hue='style_collapsed');
    plt.title("Relation of SRM and ABV")
    plt.show()


########### LOOKING AT CLUSTERS ################
from sklearn.preprocessing import StandardScaler

def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return dataframe (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')

##################
### Cluster to show a better relationship between IBU and SRM features
    ### K means was used to determine cluster size

def ibu_srm_cluster():
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)

    X = train[['ibu', 'srm']]
    train, X_scaled, scaler, kmeans, centroids = create_cluster(train, X, 5)
    # use the functions I created above to plot the data

    create_scatter_plot('srm', 'ibu',train,kmeans, X_scaled, scaler)

##################
### Cluster to show a better relationship between IBU and ABV features
    ### K means was used to determine cluster size


def ibu_abv_cluster():
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)

    X = train[['ibu', 'abv']]
    train, X_scaled, scaler, kmeans, centroids = create_cluster(train, X, 7)
    create_scatter_plot('abv', 'ibu',train,kmeans, X_scaled, scaler)


##################
### Cluster to show a better relationship between SRM and ABV features
    ### K means was used to determine cluster size
def srm_abv_cluster():
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)

    X = train[['srm', 'abv']]
    train, X_scaled, scaler, kmeans, centroids = create_cluster(train, X, 5)
    create_scatter_plot('abv', 'srm',train,kmeans, X_scaled, scaler)

######## STATS #######

def chi_ibu():
    # IMPORTS #
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)
    ### Crosstab
    observed = pd.crosstab(train.style_collapsed, train.scaled_ibu)
    
    # Categorical use Chi2
    alpha = .5
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

    if p < alpha:
        print('''\n Our p value is less than our alpha and we can reject the null hypothesis
    There is a relationship between IBU and style classification.''')
    else:
        print('We cannot reject the null hypothesis')

##################


def chi_abv():
    # IMPORTS #
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)
    ### Crosstab
    observed = pd.crosstab(train.style_collapsed, train.scaled_abv)
    
    # Categorical use Chi2
    alpha = .5
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

    if p < alpha:
        print('''\n Our p value is less than our alpha and we can reject the null hypothesis
    There is a relationship between ABV and style classification.''')
    else:
        print('We cannot reject the null hypothesis')
##################
def chi_srm():
    # IMPORTS #
    df = w.acquire_beer_classification()
    df = w.prepare_beer(df)
    train, validate, test = w.split_data(df)
    train, validate, test = w.scale(train, validate, test)
    ### Crosstab
    observed = pd.crosstab(train.style_collapsed, train.scaled_srm)
    
    # Categorical use Chi2
    alpha = .5
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

    if p < alpha:
        print('''\n Our p value is less than our alpha and we can reject the null hypothesis
    There is a relationship between SRM and style classification.''')
    else:
        print('We cannot reject the null hypothesis')