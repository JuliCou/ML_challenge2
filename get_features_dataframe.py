# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from math import sqrt
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

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
            tok = word_tokenize(token[j].lower())
            # Filter punctuation
            for elt in punc:
                tok = list(filter(lambda a: a != elt, tok))
            list_words += tok
        words.append(list_words)
    return words

def str_stem(s): 
    if isinstance(s, str):
        # Uniformisation pour le systeme unitaire
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s) # supprime les space entre 2 chiffres : 21 . 0 devient 21.0
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s) # Uniformise les distances inch en [0-9+]in.
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s) # Uniformise les distances feet en [0-9+]ft.
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s) # same pour poids
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1deg. ", s)
        s = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1hr ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal.permin. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal.perhr ", s)
        # Deal with special characters (balises html)
        # https://www.commentcamarche.net/contents/489-caracteres-speciaux-html
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("&nbsp;"," ") #code html pour " "
        s = s.replace("&amp;","&")
        s = s.replace("&#39;","'")
        s = s.replace("/>/Agt/>","")
        s = s.replace("</a<gt/","")
        s = s.replace("gt/>","")
        s = s.replace("/>","")
        s = s.replace("<br","")
        s = s.replace("<.+?>","")
        s = s.replace("[ &<>)(_,;:!?\+^~@#\$]+"," ")
        s = s.replace("'s\\b","")
        s = s.replace("[']+","")
        s = s.replace("[\"]+","")
        s = s.replace("-"," ")
        s = s.replace("+"," ")
        # Remove text between paranthesis/brackets)
        s = s.replace("[ ]?[[(].+?[])]","")
        # remove sizes
        s = s.replace("size: .+$","")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b","")
        
        
        return " ".join([stemmer.stem(re.sub('[^A-Za-z0-9-./]', ' ', word)) for word in s.lower().split()])
    else:
        return "null"
    

df_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_all = pd.merge(df, desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

list_attributes = []
for id in df_all.product_uid:
    l = list(attributes[attributes.product_uid == id].value)
    li = [l[i] if type(l[i]) == 'str' else str(l[i]) for i in range(len(l))]
    list_attributes.append(' '.join(li))

df_all["list_attributes"] = list_attributes

# Tokenize words and stem
stemmer = PorterStemmer()
df_all['product_title'] = df_all['product_title'].apply(str_stem)
df_all['search_term'] = df_all['search_term'].apply(str_stem)
df_all['product_description'] = df_all['product_description'].apply(str_stem)
df_all['brand'] = df_all['brand'].apply(str_stem)
df_all["list_attributes"] = df_all["list_attributes"].apply(str_stem)

df_all["product_words"] = nlp_prep(df_all['product_title'])
df_all["research_words"] = nlp_prep(df_all['search_term'])
df_all["brand_words"] = nlp_prep(df_all['brand'])
df_all["product_description_words"] = nlp_prep(df_all['product_description'])
df_all["list_attributes_words"] = nlp_prep(df_all["list_attributes"])

stop_words = pd.read_excel('stopWords.xlsx', header=0, encoding="ISO-8859-1")
def filter_stop_words(vec, stop_words):
    for word in stop_words:
        vec = list(filter(lambda a: a != word, vec))
    return vec

df_all.product_description_words = filter_stop_words(df_all.product_description_words, stop_words)
df_all.list_attributes_words = filter_stop_words(df_all.list_attributes_words, stop_words)

# Création de quelques features
identifiant = df_all.product_uid.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.product_uid==id]))

nbre_caracteristiques = []
for id in identifiant:
    nbre_caracteristiques.append(len(attributes[attributes.product_uid==id]))

brand = []
for id in identifiant:
    sous_ens = attributes[attributes.product_uid == id]
    brand.append(len(sous_ens[sous_ens.name == "MFG Brand Name"]))

df_all["nbre_mots_brand"] = df_all.brand_words.apply(lambda x: len(x) if "null" not in x else 0)

# Ajout des features au dataset
df_ = pd.DataFrame(data={"product_uid" : identifiant,
                         "nbre_requetes":nbre_requetes,
                         "nbre_caracteristiques": nbre_caracteristiques,
                         "longueur_brand": brand})
df_all = pd.merge(df_all, df_, how='left', on='product_uid')


# Fonction pour créer des features en 
# Comparant avec les mots de recherche
def compare_research_words(df, column_to_be_compared, original_text):
    dic = {}
    dic["product_uid"] = df["product_uid"]  # for merge with df
    
    # New empty feature lists
    dic["nbWords_" + column_to_be_compared] = df[column_to_be_compared].apply(lambda x: len(x))
    dic["total_length_char_" + column_to_be_compared] = df[column_to_be_compared].apply(lambda x: len(x))
    
    dic["nb_same_words_research_" + column_to_be_compared] = []
    dic["nb_similar_words_research_" + column_to_be_compared] = []
    dic["max_len_same_words_research_" + column_to_be_compared] = []
    dic["mean_len_same_words_research_" + column_to_be_compared] = []
    dic["std_len_same_words_research_" + column_to_be_compared] = []
    dic["max_len_similar_words_research_" + column_to_be_compared] = []
    dic["mean_len_similar_words_research_" + column_to_be_compared] = []
    dic["std_len_similar_words_research_" + column_to_be_compared] = []
    # Distances between research and column_to_be_compared
    dic["mean_Levenshtein_distance_search_" + column_to_be_compared] = []
    dic["mean_Levenshtein_distance_search_N1_" + column_to_be_compared] = []
    dic["mean_Levenshtein_distance_search_N2_" + column_to_be_compared] = []
    dic["std_Levenshtein_distance_search_" + column_to_be_compared] = []
    dic["std_Levenshtein_distance_search_N1_" + column_to_be_compared] = []
    dic["std_Levenshtein_distance_search_N2_" + column_to_be_compared] = []
    dic["min_Levenshtein_distance_search_" + column_to_be_compared] = []
    dic["min_Levenshtein_distance_search_N1_" + column_to_be_compared] = []
    dic["min_Levenshtein_distance_search_N2_" + column_to_be_compared] = []

    dic["mean_Jaro_distance_search_" + column_to_be_compared] = []
    dic["mean_Jaro_distance_search_N1_" + column_to_be_compared] = []
    dic["mean_Jaro_distance_search_N2_" + column_to_be_compared] = []
    dic["std_Jaro_distance_search_" + column_to_be_compared] = []
    dic["std_Jaro_distance_search_N1_" + column_to_be_compared] = []
    dic["std_Jaro_distance_search_N2_" + column_to_be_compared] = []
    dic["min_Jaro_distance_search_" + column_to_be_compared] = []
    dic["min_Jaro_distance_search_N1_" + column_to_be_compared] = []
    dic["min_Jaro_distance_search_N2_" + column_to_be_compared] = []

    dic["closest_distance_match_" + column_to_be_compared] = []
    dic["closest_distance_match_N1_" + column_to_be_compared] = []
    dic["closest_distance_match_N2_" + column_to_be_compared] = []
    dic["closest_distance_match_dev_" + column_to_be_compared] = []
    dic["closest_distance_match_dev_N1_" + column_to_be_compared] = []
    dic["closest_distance_match_dev_N2_" + column_to_be_compared] = []
    
    # Calcul des features pour chaque individus
    for i in range(df_all.shape[0]):
        nb = len(df_all[column_to_be_compared].iloc[i])
        if nb == 0:
            dic["nb_same_words_research_" + column_to_be_compared].append(None)
            dic["nb_similar_words_research_" + column_to_be_compared].append(None)
            dic["max_len_same_words_research_" + column_to_be_compared].append(None)
            dic["mean_len_same_words_research_" + column_to_be_compared].append(None)
            dic["std_len_same_words_research_" + column_to_be_compared].append(None)
            dic["max_len_similar_words_research_" + column_to_be_compared].append(None)
            dic["mean_len_similar_words_research_" + column_to_be_compared].append(None)
            dic["std_len_similar_words_research_" + column_to_be_compared].append(None)
            # Distances between research and column_to_be_compared
            dic["mean_Levenshtein_distance_search_" + column_to_be_compared].append(None)
            dic["mean_Levenshtein_distance_search_N1_" + column_to_be_compared].append(None)
            dic["mean_Levenshtein_distance_search_N2_" + column_to_be_compared].append(None)
            dic["std_Levenshtein_distance_search_" + column_to_be_compared].append(None)
            dic["std_Levenshtein_distance_search_N1_" + column_to_be_compared].append(None)
            dic["std_Levenshtein_distance_search_N2_" + column_to_be_compared].append(None)
            dic["min_Levenshtein_distance_search_" + column_to_be_compared].append(None)
            dic["min_Levenshtein_distance_search_N1_" + column_to_be_compared].append(None)
            dic["min_Levenshtein_distance_search_N2_" + column_to_be_compared].append(None)

            dic["mean_Jaro_distance_search_" + column_to_be_compared].append(None)
            dic["mean_Jaro_distance_search_N1_" + column_to_be_compared].append(None)
            dic["mean_Jaro_distance_search_N2_" + column_to_be_compared].append(None)
            dic["std_Jaro_distance_search_" + column_to_be_compared].append(None)
            dic["std_Jaro_distance_search_N1_" + column_to_be_compared].append(None)
            dic["std_Jaro_distance_search_N2_" + column_to_be_compared].append(None)
            dic["min_Jaro_distance_search_" + column_to_be_compared].append(None)
            dic["min_Jaro_distance_search_N1_" + column_to_be_compared].append(None)
            dic["min_Jaro_distance_search_N2_" + column_to_be_compared].append(None)

            dic["closest_distance_match_" + column_to_be_compared].append(None)
            dic["closest_distance_match_N1_" + column_to_be_compared].append(None)
            dic["closest_distance_match_N2_" + column_to_be_compared].append(None)
            dic["closest_distance_match_dev_" + column_to_be_compared].append(None)
            dic["closest_distance_match_dev_N1_" + column_to_be_compared].append(None)
            dic["closest_distance_match_dev_N2_" + column_to_be_compared].append(None)
        else:
            # Listes vides
            dist_Levenshtein = []
            dist_Jaro = []
            best_distance = []
            dist_Levenshtein_N1 = []
            dist_Jaro_N1 = []
            best_distance_N1 = []
            dist_Levenshtein_N2 = []
            dist_Jaro_N2 = []
            best_distance_N2 = []
            len_same_word = []
            len_similar_word =[]

            for word in df_all['research_words'].iloc[i]:
                disL = []
                dis_L_normalised1 = []
                dis_L_normalised2 = []
                disJ = []
                dis_J_normalised1 = []
                dis_J_normalised2 = []
                perfect_match = False

                for s_word in df_all[column_to_be_compared].iloc[i]:
                    # distances calculation
                    dL = abs(edit_distance(word, s_word))
                    dJ = abs(jaro_similarity(word, s_word))
                    # another function: jaro_winkler_similarity()

                    # computing
                    disL.append(dL)
                    dis_L_normalised1.append(abs((max(len(s_word), len(word))-dL)/max(len(s_word), len(word))))
                    dis_L_normalised2.append(abs((min(len(s_word), len(word))-dL)/min(len(s_word), len(word))))

                    disJ.append(dJ)
                    dis_J_normalised1.append(abs((max(len(s_word), len(word))-dJ)/max(len(s_word), len(word))))
                    dis_J_normalised2.append(abs((min(len(s_word), len(word))-dJ)/min(len(s_word), len(word))))

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

            # Mise à jour des features
            dic["nb_same_words_research_" + column_to_be_compared].append(len(len_same_word))
            dic["nb_similar_words_research_" + column_to_be_compared].append(len(len_similar_word))

            if len_same_word != []:
                dic["max_len_same_words_research_" + column_to_be_compared].append(max(len_same_word))
                dic["mean_len_same_words_research_" + column_to_be_compared].append(mean(len_same_word))    
                if len(len_same_word) > 1:
                    dic["std_len_same_words_research_" + column_to_be_compared].append(stdev(len_same_word))
                else:
                    dic["std_len_same_words_research_" + column_to_be_compared].append(-1)
            else:
                dic["max_len_same_words_research_" + column_to_be_compared].append(0)
                dic["mean_len_same_words_research_" + column_to_be_compared].append(0)
                dic["std_len_same_words_research_" + column_to_be_compared].append(0)

            if len_similar_word != []:
                dic["max_len_similar_words_research_" + column_to_be_compared].append(max(len_similar_word))
                dic["mean_len_similar_words_research_" + column_to_be_compared].append(mean(len_similar_word))
                if len(len_similar_word) > 1:
                    dic["std_len_similar_words_research_" + column_to_be_compared].append(stdev(len_similar_word))
                else:
                    dic["std_len_similar_words_research_" + column_to_be_compared].append(-1)
            else:
                dic["max_len_similar_words_research_" + column_to_be_compared].append(0)
                dic["mean_len_similar_words_research_" + column_to_be_compared].append(0)
                dic["std_len_similar_words_research_" + column_to_be_compared].append(0)

            dic["mean_Levenshtein_distance_search_" + column_to_be_compared].append(mean(dist_Levenshtein))
            dic["mean_Levenshtein_distance_search_N1_" + column_to_be_compared].append(mean(dist_Levenshtein_N1))
            dic["mean_Levenshtein_distance_search_N2_" + column_to_be_compared].append(mean(dist_Levenshtein_N2))   
            dic["min_Levenshtein_distance_search_" + column_to_be_compared].append(min(dist_Levenshtein))
            dic["min_Levenshtein_distance_search_N1_" + column_to_be_compared].append(min(dist_Levenshtein_N1))
            dic["min_Levenshtein_distance_search_N2_" + column_to_be_compared].append(min(dist_Levenshtein_N2))

            dic["mean_Jaro_distance_search_" + column_to_be_compared].append(mean(dist_Jaro))
            dic["mean_Jaro_distance_search_N1_" + column_to_be_compared].append(mean(dist_Jaro_N1))
            dic["mean_Jaro_distance_search_N2_" + column_to_be_compared].append(mean(dist_Jaro_N2))
            dic["min_Jaro_distance_search_" + column_to_be_compared].append(min(dist_Jaro))
            dic["min_Jaro_distance_search_N1_" + column_to_be_compared].append(min(dist_Jaro_N1))
            dic["min_Jaro_distance_search_N2_" + column_to_be_compared].append(min(dist_Jaro_N2))

            if len(dist_Levenshtein) > 1:
                dic["std_Levenshtein_distance_search_" + column_to_be_compared].append(stdev(dist_Levenshtein))
                dic["std_Levenshtein_distance_search_N1_" + column_to_be_compared].append(stdev(dist_Levenshtein_N1))
                dic["std_Levenshtein_distance_search_N2_" + column_to_be_compared].append(stdev(dist_Levenshtein_N2))

                dic["std_Jaro_distance_search_" + column_to_be_compared].append(stdev(dist_Jaro))
                dic["std_Jaro_distance_search_N1_" + column_to_be_compared].append(stdev(dist_Jaro_N1))
                dic["std_Jaro_distance_search_N2_" + column_to_be_compared].append(stdev(dist_Jaro_N2))
            else :
                dic["std_Levenshtein_distance_search_" + column_to_be_compared].append(-1)
                dic["std_Levenshtein_distance_search_N1_" + column_to_be_compared].append(-1)
                dic["std_Levenshtein_distance_search_N2_" + column_to_be_compared].append(-1)

                dic["std_Jaro_distance_search_" + column_to_be_compared].append(-1)
                dic["std_Jaro_distance_search_N1_" + column_to_be_compared].append(-1)
                dic["std_Jaro_distance_search_N2_" + column_to_be_compared].append(-1)

            if best_distance != []:
                dic["closest_distance_match_" + column_to_be_compared].append(min(best_distance))
                dic["closest_distance_match_N1_" + column_to_be_compared].append(min(best_distance_N1))
                dic["closest_distance_match_N2_" + column_to_be_compared].append(min(best_distance_N2))
                if len(best_distance) > 1:
                    dic["closest_distance_match_dev_" + column_to_be_compared].append(stdev(best_distance))
                    dic["closest_distance_match_dev_N1_" + column_to_be_compared].append(stdev(best_distance_N1))
                    dic["closest_distance_match_dev_N2_" + column_to_be_compared].append(stdev(best_distance_N2))
                else:
                    dic["closest_distance_match_dev_" + column_to_be_compared].append(-1)
                    dic["closest_distance_match_dev_N1_" + column_to_be_compared].append(-1)
                    dic["closest_distance_match_dev_N2_" + column_to_be_compared].append(-1)
            else:
                dic["closest_distance_match_" + column_to_be_compared].append(-2)
                dic["closest_distance_match_N1_" + column_to_be_compared].append(-2)
                dic["closest_distance_match_N2_" + column_to_be_compared].append(-2)
                dic["closest_distance_match_dev_" + column_to_be_compared].append(-2)
                dic["closest_distance_match_dev_N1_" + column_to_be_compared].append(-2)
                dic["closest_distance_match_dev_N2_" + column_to_be_compared].append(-2)

    # Ajout des features dans df
    dic_ = pd.DataFrame(dic)
    dic_.to_csv(column_to_be_compared + '.csv', index=False)
    for col in dic_.columns:
        df[col] = dic_[col]
    return df


# New features
df_all["nbWords_search_terms"] = df_all.research_words.apply(lambda x: len(x))
df_all["total_search_terms_length"] = df_all.search_term.apply(lambda x: len(x))

# New features (with brand)
brand_nb_same_words = []
brand_total_length_same_words = []
brand_percentage_length_same_words = [] # matching percentage

# Distances between research and brand
mean_distance_brand_L = []
min_distance_brand_L = []
mean_distance_brand_J = []
min_distance_brand_J = []
mean_distance_brand_LN = []
min_distance_brand_LN = []
mean_distance_brand_JN = []
min_distance_brand_JN = []

for i in range(df_all.shape[0]): 
    # Brand words
    dist_brand_Le = []
    dist_brand_Jaro = []
    dist_brand_Le_N = []
    dist_brand_Jaro_N = []
    brand_same = []
    
    for word in df_all['research_words'].iloc[i]:
        if df_all["brand"].iloc[i] == "null":
            dist_brand_Le.append(-1)
            dist_brand_Jaro.append(-1)
            dist_brand_Le_N.append(-1)
            dist_brand_Jaro_N.append(-1)
        else:
            for s_word in df_all['brand_words'].iloc[i]:
                dL = abs(edit_distance(word, s_word))
                dJ = abs(jaro_similarity(word, s_word))
                dist_brand_Le.append(dL)
                dist_brand_Jaro.append(dJ)
                dist_brand_Le_N.append(abs((max(len(s_word), len(word))-dL)/max(len(s_word), len(word))))
                dist_brand_Le_N.append(abs((min(len(s_word), len(word))-dL)/min(len(s_word), len(word))))
                dist_brand_Jaro_N.append(abs((max(len(s_word), len(word))-dJ)/max(len(s_word), len(word))))
                dist_brand_Jaro_N.append(abs((min(len(s_word), len(word))-dJ)/min(len(s_word), len(word))))
                if word == s_word:
                    brand_same.append(word)

    # Création de features
    # Marque - brand
    brand_nb_same_words.append(len(brand_same))
    brand_total_length_same_words.append(sum(map(lambda x:len(x), brand_same)))                                         
    brand_percentage_length_same_words.append((sum(map(lambda x:len(x), brand_same))/sum(map(lambda x:len(x), df_all.iloc[i]["brand_words"])))) # matching percentage
    mean_distance_brand_L.append(mean(dist_brand_Le))
    min_distance_brand_L.append(min(dist_brand_Le))
    mean_distance_brand_J.append(mean(dist_brand_Jaro))
    min_distance_brand_J.append(min(dist_brand_Jaro))
    mean_distance_brand_LN.append(mean(dist_brand_Le_N))
    min_distance_brand_LN.append(min(dist_brand_Le_N))
    mean_distance_brand_JN.append(mean(dist_brand_Jaro_N))
    min_distance_brand_JN.append(min(dist_brand_Jaro_N))

# Adding features
df_all["brand_nb_same_words"] = brand_nb_same_words
df_all["brand_total_length_same_words"] = brand_total_length_same_words
df_all["brand_percentage_length_same_words"] = brand_percentage_length_same_words
df_all["mean_distance_brand_L"] = mean_distance_brand_L
df_all["min_distance_brand_L"] = min_distance_brand_L
df_all["mean_distance_brand_J"] = mean_distance_brand_J
df_all["min_distance_brand_J"] = min_distance_brand_J
df_all["mean_distance_brand_LN"] = mean_distance_brand_LN
df_all["min_distance_brand_LN"] = min_distance_brand_LN
df_all["mean_distance_brand_JN"] = mean_distance_brand_JN
df_all["min_distance_brand_JN"] = min_distance_brand_JN


# Ajout de features avec la fonction
df_all = compare_research_words(df_all, column_to_be_compared='product_description_words', original_text='product_description')
df_all = compare_research_words(df_all, column_to_be_compared='product_words', original_text='product_title')
df_all = compare_research_words(df_all, column_to_be_compared="list_attributes_words", original_text="list_attributes")

# Suppression des colonnes inutiles
df_model = df_all.drop(['product_title', 'search_term', 'product_description', 'brand', 'product_words', 'research_words',
                        'brand_words', 'product_description_words', 'list_attributes_words', 'list_attributes'], axis=1)
features = df_model.columns
print(features)
df_model.to_csv('dataframe.csv', index=False)