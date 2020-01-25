# Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import sqrt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.metrics import *
from nltk.metrics.distance import jaro_similarity, jaro_winkler_similarity

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')



# Import data
train = pd.read_csv('train.csv', header=0, sep=",", encoding="ISO-8859-1")
test = pd.read_csv('test.csv', header=0, sep=",", encoding="ISO-8859-1")
attributes = pd.read_csv('attributes.csv', header=0, sep=",", encoding="ISO-8859-1")
desc = pd.read_excel('product_descriptions.xlsx', header=0, encoding="ISO-8859-1")

# DataFrame
target = train['relevance']
piv_train = train.shape[0]
df = pd.concat((train.drop('relevance', axis=1), test), axis=0, ignore_index=True)

# Some functions

# Param forest is the model
# Param name_f is a list with the names of the features
def plot_feature_importances(forest, name_f):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), " ", name_f[indices[f]])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def nlp_prep(col):
    punc = ['.', ',', ';', ')', '(']

    words = []
    for i in range(len(col)):
        token = sent_tokenize(col[i])
        list_words = []
        for j in range(len(token)):
            stem = word_tokenize(token[j].lower())
            # Filter punctuation
            for elt in punc:
                stem = list(filter(lambda a: a != elt, stem))
            list_words += stem
        words.append(list_words)
    return words

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


df["product_words"] = nlp_prep(df['product_title'])
df["research_words"] = nlp_prep(df['search_term'])

# New empty features lists
nbWords_search_terms = []
total_search_terms_length = []
nbWords_product_title = []
total_product_title_length = []
nb_same_words_research = []
nb_similar_words_research = []
max_len_same_words_research = []
mean_len_same_words_research = []
std_len_same_words_research = []
max_len_similar_words_research = []
mean_len_similar_words_research = []
std_len_similar_words_research = []

mean_Levenshtein_distance_title_search = []
mean_Levenshtein_distance_title_search_N1 = []
mean_Levenshtein_distance_title_search_N2 = []
std_Levenshtein_distance_title_search = []
std_Levenshtein_distance_title_search_N1 = []
std_Levenshtein_distance_title_search_N2 = []
min_Levenshtein_distance_title_search = []
min_Levenshtein_distance_title_search_N1 = []
min_Levenshtein_distance_title_search_N2 = []

mean_Jaro_distance_title_search = []
mean_Jaro_distance_title_search_N1 = []
mean_Jaro_distance_title_search_N2 = []
std_Jaro_distance_title_search = []
std_Jaro_distance_title_search_N1 = []
std_Jaro_distance_title_search_N2 = []
min_Jaro_distance_title_search = []
min_Jaro_distance_title_search_N1 = []
min_Jaro_distance_title_search_N2 = []

closest_distance_match = []
closest_distance_match_N1 = []
closest_distance_match_N2 = []
closest_distance_match_dev = []
closest_distance_match_dev_N1 = []
closest_distance_match_dev_N2 = []

for i in range(df.shape[0]): 
    # research_words 
    dist_Levenshtein = []
    dist_Jaro = []
    best_distance = []
    dist_Levenshtein_N1 = []
    dist_Jaro_N1 = []
    best_distance_N1 = []
    dist_Levenshtein_N2 = []
    dist_Jaro_N2 = []
    best_distance_N2 = []
    lenWord = []
    
    len_same_word = []
    len_similar_word =[]
    
    for word in df['research_words'].iloc[i]:
        lenWord.append(len(word))
        disL = []
        dis_L_normalised1 = []
        dis_L_normalised2 = []
        disJ = []
        dis_J_normalised1 = []
        dis_J_normalised2 = []
        
        perfect_match = False
        
        for s_word in df['product_words'].iloc[i]:
            # distances calculation
            dL = edit_distance(word, s_word)
            dJ = jaro_similarity(word, s_word)
            # another function: jaro_winkler_similarity()
            
            # computing
            disL.append(dL)
            dis_L_normalised1.append((max(len(s_word), len(word))-dL)/max(len(s_word), len(word)))
            dis_L_normalised2.append((min(len(s_word), len(word))-dL)/min(len(s_word), len(word)))
            
            disJ.append(dJ)
            dis_J_normalised1.append((max(len(s_word), len(word))-dJ)/max(len(s_word), len(word)))
            dis_J_normalised2.append((min(len(s_word), len(word))-dJ)/min(len(s_word), len(word)))
            
            if s_word == word:
                perfect_match = True
                len_same_word.append(len(word))
            
            if dis_L_normalised2[-1] > 0.7:
                len_similar_word.append(len(word))
                
        dist_Levenshtein.append(mean(disL))
        dist_Jaro.append(mean(disJ))
        dist_Levenshtein_N1.append(mean(dis_L_normalised1))
        dist_Levenshtein_N2.append(mean(dis_L_normalised2))
        dist_Jaro_N1.append(mean(dis_J_normalised1))
        dist_Jaro_N2.append(mean(dis_J_normalised2))

        if not perfect_match:
            best_distance.append(min(disL + disJ))
            best_distance_N1.append(min(dist_Levenshtein_N1+dist_Jaro_N1))
            best_distance_N2.append(min(dist_Levenshtein_N2+dist_Jaro_N2))
            
    # Création de features
    nbWords_search_terms.append(len(lenWord))
    total_search_terms_length.append(sum(lenWord))
    
    nb_same_words_research.append(len(len_same_word))
    nb_similar_words_research.append(len(len_similar_word))
    
    if len_same_word != []:
        max_len_same_words_research.append(max(len_same_word))
        mean_len_same_words_research.append(mean(len_same_word))    
        if len(len_same_word) > 1:
            std_len_same_words_research.append(stdev(len_same_word))
        else:
            std_len_same_words_research.append(-1)
    else:
        max_len_same_words_research.append(0)
        mean_len_same_words_research.append(0)
        std_len_same_words_research.append(0)
    
    if len_similar_word != []:
        max_len_similar_words_research.append(max(len_similar_word))
        mean_len_similar_words_research.append(mean(len_similar_word))
        if len(len_similar_word) > 1:
            std_len_similar_words_research.append(stdev(len_similar_word))
        else:
            std_len_similar_words_research.append(-1)
    else:
        max_len_similar_words_research.append(0)
        mean_len_similar_words_research.append(0)
        std_len_similar_words_research.append(0)

    mean_Levenshtein_distance_title_search.append(mean(dist_Levenshtein))
    mean_Levenshtein_distance_title_search_N1.append(mean(dist_Levenshtein_N1))
    mean_Levenshtein_distance_title_search_N2.append(mean(dist_Levenshtein_N2))   
    min_Levenshtein_distance_title_search.append(min(dist_Levenshtein))
    min_Levenshtein_distance_title_search_N1.append(min(dist_Levenshtein_N1))
    min_Levenshtein_distance_title_search_N2.append(min(dist_Levenshtein_N2))

    mean_Jaro_distance_title_search.append(mean(dist_Jaro))
    mean_Jaro_distance_title_search_N1.append(mean(dist_Jaro_N1))
    mean_Jaro_distance_title_search_N2.append(mean(dist_Jaro_N2))
    min_Jaro_distance_title_search.append(min(dist_Jaro))
    min_Jaro_distance_title_search_N1.append(min(dist_Jaro_N1))
    min_Jaro_distance_title_search_N2.append(min(dist_Jaro_N2))
    
    if len(dist_Levenshtein) > 1:
        std_Levenshtein_distance_title_search.append(stdev(dist_Levenshtein))
        std_Levenshtein_distance_title_search_N1.append(stdev(dist_Levenshtein_N1))
        std_Levenshtein_distance_title_search_N2.append(stdev(dist_Levenshtein_N2))
        
        std_Jaro_distance_title_search.append(stdev(dist_Jaro))
        std_Jaro_distance_title_search_N1.append(stdev(dist_Jaro_N1))
        std_Jaro_distance_title_search_N2.append(stdev(dist_Jaro_N2))
    else :
        std_Levenshtein_distance_title_search.append(-1)
        std_Levenshtein_distance_title_search_N1.append(-1)
        std_Levenshtein_distance_title_search_N2.append(-1)
        
        std_Jaro_distance_title_search.append(-1)
        std_Jaro_distance_title_search_N1.append(-1)
        std_Jaro_distance_title_search_N2.append(-1)
        
    if best_distance != []:
        closest_distance_match.append(min(best_distance))
        closest_distance_match_N1.append(min(best_distance_N1))
        closest_distance_match_N2.append(min(best_distance_N2))
        if len(best_distance) > 1:
            closest_distance_match_dev.append(stdev(best_distance))
            closest_distance_match_dev_N1.append(stdev(best_distance_N1))
            closest_distance_match_dev_N2.append(stdev(best_distance_N2))
        else:
            closest_distance_match_dev.append(-1)
            closest_distance_match_dev_N1.append(-1)
            closest_distance_match_dev_N2.append(-1)
    else:
        closest_distance_match.append(0)
        closest_distance_match_N1.append(0)
        closest_distance_match_N2.append(0)
        closest_distance_match_dev.append(0)
        closest_distance_match_dev_N1.append(0)
        closest_distance_match_dev_N2.append(0)
    
    # product_words
    lengthProductTitle = []
    for word in df["research_words"].iloc[i]:
        lengthProductTitle.append(len(word))
        
    # Creation of features
    nbWords_product_title.append(len(lengthProductTitle))
    total_product_title_length.append(sum(lengthProductTitle))

# Adding features
df["nbWords_search_terms"] = nbWords_search_terms
df["total_search_terms_length"] = total_search_terms_length
df["total_search_terms_length"] = total_search_terms_length
df["nbWords_product_title"] = nbWords_product_title
df["total_product_title_length"] = total_product_title_length
df["nb_same_words_research"] = nb_same_words_research
df["nb_similar_words_research"] = nb_similar_words_research
df["max_len_same_words_research"] = max_len_same_words_research
df["mean_len_same_words_research"] = mean_len_same_words_research
df["std_len_same_words_research"] = std_len_same_words_research
df["max_len_similar_words_research"] = max_len_similar_words_research
df["mean_len_similar_words_research"] = mean_len_similar_words_research
df["std_len_similar_words_research"] = std_len_similar_words_research
df["mean_Levenshtein_distance_title_search"] = mean_Levenshtein_distance_title_search
df["mean_Levenshtein_distance_title_search_N1"] = mean_Levenshtein_distance_title_search_N1
df["mean_Levenshtein_distance_title_search_N2"] = mean_Levenshtein_distance_title_search_N2
df["std_Levenshtein_distance_title_search"] = std_Levenshtein_distance_title_search
df["std_Levenshtein_distance_title_search_N1"] = std_Levenshtein_distance_title_search_N1
df["std_Levenshtein_distance_title_search_N2"] = std_Levenshtein_distance_title_search_N2
df["min_Levenshtein_distance_title_search"] = min_Levenshtein_distance_title_search
df["min_Levenshtein_distance_title_search_N1"] = min_Levenshtein_distance_title_search_N1
df["min_Levenshtein_distance_title_search_N2"] = min_Levenshtein_distance_title_search_N2
df["mean_Jaro_distance_title_search"] = mean_Jaro_distance_title_search
df["mean_Jaro_distance_title_search_N1"] = mean_Jaro_distance_title_search_N1
df["mean_Jaro_distance_title_search_N2"] = mean_Jaro_distance_title_search_N2
df["std_Jaro_distance_title_search"] = std_Jaro_distance_title_search
df["std_Jaro_distance_title_search_N1"] = std_Jaro_distance_title_search_N1
df["std_Jaro_distance_title_search_N2"] = std_Jaro_distance_title_search_N2
df["min_Jaro_distance_title_search"] = min_Jaro_distance_title_search
df["min_Jaro_distance_title_search_N1"] = min_Jaro_distance_title_search_N1
df["min_Jaro_distance_title_search_N2"] = min_Jaro_distance_title_search_N2
df["closest_distance_match"] = closest_distance_match
df["closest_distance_match_N1"] = closest_distance_match_N1
df["closest_distance_match_N2"] = closest_distance_match_N2
df["closest_distance_match_dev"] = closest_distance_match_dev
df["closest_distance_match_dev_N1"] = closest_distance_match_dev_N1
df["closest_distance_match_dev_N2"] = closest_distance_match_dev_N2

df_model = df.drop(['product_title', 'search_term', 'product_words', 'research_words'], axis=1)
features = df_model.columns

X = df_model.iloc[0:piv_train].values
y = target.values

# Best parameters

RFReg = RandomForestRegressor(random_state=1,
                              n_jobs=-1) 

param_grid = {'max_features' : ["auto", "sqrt", "log2"],
              'min_samples_split' : np.linspace(0.1, 1.0, 10),
              'max_depth' : [1, 5, 10, 20, 50],
              'n_estimators' : [100, 200, 300, 500, 1000]}

n_splits = 10
CV_rfc = RandomizedSearchCV(estimator=RFReg,
                            param_distributions=param_grid,
                            n_jobs=-1,
                            cv=n_splits,
                            n_iter=50)

CV_rfc.fit(X, y)
report(CV_rfc.cv_results_)

# Cross-validation

kf = KFold(n_splits=n_splits, random_state=2019)
rmse = []

for train_index, cv_index in kf.split(X, y):
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]

    RFReg = RandomForestRegressor(random_state=1,
                                  n_jobs=-1, 
                                  max_features=CV_rfc.best_estimator_.max_features, 
                                  min_samples_split=CV_rfc.best_estimator_.min_samples_split, 
                                  max_depth=CV_rfc.best_estimator_.max_depth, 
                                  n_estimators=CV_rfc.best_estimator_.n_estimators)
    RFReg.fit(X_train, y_train)

    y_pred = RFReg.predict(X_cv)
    mse = mean_squared_error(y_cv, y_pred)
    rmse.append(sqrt(mse))
    print("mse: ", mse, " RMSE: ", sqrt(mse))    

print('RMSE: ' + str(sum(rmse) / n_splits), " std : ", str(stdev(rmse)))

# Model
regr = RandomForestRegressor(random_state=1,
                             n_jobs=-1, 
                             max_features=CV_rfc.best_estimator_.max_features, 
                             min_samples_split=CV_rfc.best_estimator_.min_samples_split,
                             max_depth=CV_rfc.best_estimator_.max_depth,
                             n_estimators=CV_rfc.best_estimator_.n_estimators)
regr.fit(X, y)

plot_feature_importances(regr, features)

test['relevance'] = regr.predict(df_model.iloc[piv_train:df_model.shape[0]].values)
test[['id', 'relevance']].to_csv('minimal_submission.csv', index=False)