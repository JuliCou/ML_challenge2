# Imports

import pandas as pd
import numpy as np
from statistics import mean, stdev
from math import sqrt
import re

import nltk
# nltk.download('punkt')
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
stemmer = PorterStemmer()
# def nlp_prep(col):
#     punc = [',', ';', ')', '(', " "]

#     words = []
#     for i in range(len(col)):
#         token = sent_tokenize(col[i])
#         list_words = []
#         for j in range(len(token)):
#             tok = word_tokenize(token[j].lower())
#             # Filter punctuation
#             for elt in punc:
#                 tok = list(filter(lambda a: a != elt, tok))
#             list_words += [stemmer.stem(wd) for wd in tok]
#         words.append(list_words)
#     return words
def nlp_prep(sentence):
    punc = [',', ';', ')', '(', " "]

    token = sent_tokenize(sentence)
    list_words = []
    for j in range(len(token)):
        tok = word_tokenize(token[j].lower())
        # Filter punctuation
        for elt in punc:
            tok = list(filter(lambda a: a != elt, tok))
        list_words += [stemmer.stem(wd) for wd in tok]

    return list_words


def str_stem(s): 
    if isinstance(s, str):
        s = re.sub(r"([0-9]+)( *)/( *)([0-9])", r"\1/\4", s)
        s = re.sub(r"([0-9]+)( *)\.( *)([0-9])", r"\1.\4", s) # supprime les space entre 2 chiffres : 21 . 0 devient 21.0
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1inches ", s) # Uniformise les distances inch en [0-9+]in.
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1feet ", s) # Uniformise les distances feet en [0-9+]ft.
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1pounds ", s) # same pour poids
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1square feet ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cubic feet ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gallons ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1ounces ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1centimeters ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1milimeters ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1degrees ", s)
        s = re.sub(r"([0-9]+)( *)(v|volts|volt)\.?", r"\1volts ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1watts ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amps ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1hours ", s)
        s = re.sub(r"([a-zA-Z])\.", r"\1", s)
        car = ["[", "]", "?", "(", ")", ",", ".", ";", "/", "-", "\\"]  # "." utilisé pour les chiffres
        for char in car:
            s = s.replace(char, " ")
        s = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1gallons.perminute ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1gallons.perhour ", s)
        s = re.sub(r"(#+)([0-9a-zA-Z]+)", "", s)
        s = re.sub(r"([0-9]+)( *)(hours|qt.|amps|watts|volts|degrees|milimeters|centimeters|ounces|gallons|pounds|feet|inches)( *)x( *)([0-9]+)( *)(hours|qt.|amps|watts|volts|degrees|milimeters|centimeters|ounces|gallons|pounds|feet|inches)", r"\1x\6 \3 \8 ", s)
        s = re.sub(r"([0-9]+)( *)x( *)([0-9]+)", r"\1x\4 ", s)
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
        s = s.replace("\\x8a"," ")
        # Remove text between paranthesis/brackets)
        s = s.replace("[ ]?[[(].+?[])]"," ")
        # remove sizes
        s = s.replace("size: .+$","")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b","")
        return " ".join([re.sub('[^A-Za-z0-9-./]', ' ', word) for word in s.lower().split()])
    else:
        return ""


# Fonction pour nettoyer les données du dataset attributes
def attr_treatment(s):
    if isinstance(s, str):
        s = re.sub(r"([Bb]ullet[0-9]+)", "", s)
        s = re.sub(r"([Mm][Ff][Gg] [Bb]rand [Nn]ame)", "", s)
        s = re.sub(r"([Mm]aterial)", "", s)
        return " ".join([re.sub('[^A-Za-z0-9-./]', ' ', word) for word in s.lower().split()])
    else:
        return ""


# Load Dictionnary used for spell checking
# txt file from correcting_sear_terms.py
corpus_file_name = 'spellChecker/spell_check_dictionary.txt'
dic_words = {}
term_index = 0  # column of the term in the dictionary text file
count_index = 1  # column of the term frequency in the dictionary text file
with open(corpus_file_name, "r", encoding=None) as infile:
    for line in infile:
        line_parts = line.rstrip().split(" ")
        if len(line_parts) >= 2:
            key = line_parts[term_index]
            try:
                count = int(line_parts[count_index])
                if count <= 0 or count >= 2 ** 64:
                    count = None
            except ValueError:
                count = None
            if count is not None:
                key = nlp_prep(key)
                for k in key:
                    if k in dic_words:
                        dic_words[k] += count
                    else:
                        dic_words[k] = count


print("STEP 1: Merging des différents dataset")
# Load corrected terms
correction = pd.read_csv("corrected_search_terms.csv", header=0, sep=",")
corrections = correction[["id", "corrected_search_terms", "nbre_corr_search"]]
df_all = pd.merge(df, corrections, how="left", on='id')

# df_brand = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_all = pd.merge(df_all, desc, how='left', on='product_uid')
# df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

print("STEP 2: Agglomération des attributs")
list_attributes = []
for id in df_all.product_uid.unique():
    # New id corresponding to product_uid
    attr = attributes[attributes.product_uid == id]
    attr = attr[attr.value != "no"]
    attr = attr[attr.value != "No"]
    l = list(attr.name + ' ' + attr.value)
    l = [l[i] if type(l[i]) == 'str' else str(l[i]) for i in range(len(l))]
    li = attr_treatment(' '.join(l))
    list_attributes.append(li)

df_attr = pd.DataFrame({"product_uid": df_all.product_uid.unique(), "list_attributes":list_attributes})
df_all = pd.merge(df_all, df_attr, how='left', on="product_uid")

print("STEP 3: Agglomération des données")
print("STEP 3.1: sur product title - création de title_id")
product_title = list(df_all.product_title.unique())
new_product_id = {product_title[i] : i for i in range(len(product_title))}

title_id = []
for i in range(df_all.shape[0]):
    title_id.append(new_product_id[df_all.product_title.iloc[i]])

df_all["title_id"] = title_id

print("STEP 3.2: sur product_description - création de product_description_id")
product_description = list(df_all.product_description.unique())
new_product_id = {product_description[i] : i for i in range(len(product_description))}

product_description_id = []
for i in range(df_all.shape[0]):
    product_description_id.append(new_product_id[df_all.product_description.iloc[i]])

df_all["product_description_id"] = product_description_id

print("STEP 3.3: sur list_attributes - création de list_attributes_id")
list_attributes = list(df_all.list_attributes.unique())
new_product_id = {list_attributes[i] : i for i in range(len(list_attributes))}

list_attributes_id = []
for i in range(df_all.shape[0]):
    list_attributes_id.append(new_product_id[df_all.list_attributes.iloc[i]])

df_all["list_attributes_id"] = list_attributes_id

print("STEP 3.4: sur search_term - création de search_term_id")
search_term = list(df_all.search_term.unique())
new_product_id = {search_term[i] : i for i in range(len(search_term))}

search_term_id = []
for i in range(df_all.shape[0]):
    search_term_id.append(new_product_id[df_all.search_term.iloc[i]])

df_all["search_term_id"] = search_term_id

# Tokenize words and stem
print("STEP 4: Nettoyage de toutes les chaines de caractères")
df_all['product_title'] = df_all['product_title'].apply(str_stem)
print("STEP 4: 1/4")
df_all['search_term'] = df_all['search_term'].apply(str_stem)
df_all['corrected_search_terms'] = df_all['corrected_search_terms'].apply(str_stem)
print("STEP 4: 2/4")
df_all['product_description'] = df_all['product_description'].apply(str_stem)
print("STEP 4: 3/4")
df_all["list_attributes"] = df_all["list_attributes"].apply(str_stem)

print("STEP 5: Tokenizing + stemmer")
df_all["product_words"] = df_all['product_title'].apply(nlp_prep)
print("STEP 5: 1/4")
df_all["research_words"] = df_all['search_term'].apply(nlp_prep)
df_all["corrected_research_words"] = df_all["corrected_search_terms"].apply(nlp_prep)
print("STEP 5: 2/4")
df_all["product_description_words"] = df_all['product_description'].apply(nlp_prep)
print("STEP 5: 3/4")
df_all["list_attributes_words"] = df_all["list_attributes"].apply(nlp_prep)

print("STEP 6: Stop words")
stop_words = pd.read_excel('stopWords.xlsx', header=None, encoding="ISO-8859-1")
stop_words = list(stop_words[0])
def filter_stop_words(vec, stop_words=stop_words):
    for word in stop_words:
        vec = list(filter(lambda a: a != word, vec))
    return vec

df_all["product_description_words"] = df_all["product_description_words"].apply(filter_stop_words)
df_all["list_attributes_words"] = df_all["list_attributes_words"].apply(filter_stop_words)
df_all["product_words"] = df_all["product_words"].apply(filter_stop_words)

# Création de quelques features
print("STEP 7: Création de features...")
identifiant = df_all.product_uid.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.product_uid==id]))

nbre_caracteristiques = []
for id in identifiant: 
    nbre_caracteristiques.append(len(attributes[attributes.product_uid==id]))

brand = []
longueur_brand = []
nbre_produits_same_brand = []
for id in identifiant:
    sous_ens = attributes[attributes.product_uid == id]
    brandID = sous_ens[sous_ens.name == "MFG Brand Name"]
    nbreBrand = brandID.shape[0]
    if nbreBrand != 0:
        brandID_ = str(brandID.iloc[0].value).lower()
        if brandID_ == "nan" and brandID_ == "none":
            brandID_ = ""
            nbre_produits_same_brand.append(0)
        else:
            sous_ens = attributes[attributes.value == brandID.iloc[0].value]
            nbre_produits_same_brand.append(len(sous_ens.product_uid.unique()))
    else:
        brandID_ = ""
        nbre_produits_same_brand.append(0)
    brand.append(brandID_)  
    longueur_brand.append(len(brandID_))

nbre_mots_brand = [len(x.split(" ")) for x in brand]
longueur_brand = [abs(a - b + 1) for a, b in zip(longueur_brand, nbre_mots_brand)]

# Ajout des features au dataset
df_ = pd.DataFrame(data={"product_uid" : identifiant,
                         "nbre_requetes": nbre_requetes,
                         "nbre_caracteristiques": nbre_caracteristiques,
                         "longueur_brand": longueur_brand,
                         "nbre_mots_brand": nbre_mots_brand,
                         "brand" : brand,
                         "nbre_produits_same_brand": nbre_produits_same_brand})
df_all = pd.merge(df_all, df_, how='left', on='product_uid')

identifiant = df_all.title_id.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.title_id==id]))
df_ = pd.DataFrame(data={"title_id" : identifiant,
                         "nbre_requetes_title": nbre_requetes})
df_all = pd.merge(df_all, df_, how='left', on='title_id')

identifiant = df_all.product_description_id.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.product_description_id==id]))

df_ = pd.DataFrame(data={"product_description_id" : identifiant,
                         "nbre_requetes_product_description_id": nbre_requetes})
df_all = pd.merge(df_all, df_, how='left', on='product_description_id')

identifiant = df_all.list_attributes_id.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.list_attributes_id==id]))

df_ = pd.DataFrame(data={"list_attributes_id" : identifiant,
                         "nbre_requetes_list_attributes_id": nbre_requetes})
df_all = pd.merge(df_all, df_, how='left', on='list_attributes_id')

identifiant = df_all.search_term_id.unique()
nbre_requetes = []
for id in identifiant:
    nbre_requetes.append(len(df_all[df_all.search_term_id==id]))

df_ = pd.DataFrame(data={"search_term_id" : identifiant,
                         "nbre_requetes_search_term_id": nbre_requetes})
df_all = pd.merge(df_all, df_, how='left', on='search_term_id')

print("STEP 8: Tokenizing Stemmer Brand")
df_all['brand'] = df_all['brand'].apply(str_stem)
df_all["brand_words"] = df_all['brand'].apply(nlp_prep)

print("STEP 9: Ecriture fichiers données texte dataframe_NLP.csv")
df_traitement_nlp = df_all[['id', 'product_title', 'search_term', "corrected_search_terms",
                        'product_description', 'brand', 'product_words', 'research_words',
                        'brand_words', 'product_description_words', 'list_attributes_words', 
                        'list_attributes', "list_attributes_id", "corrected_research_words",
                        "product_description_id", "title_id"]]
df_traitement_nlp.to_csv('dataframe_NLP.csv', index=False)

# Fonction pour créer des features en 
# Comparant avec les mots de recherche
def compare_research_words(df, column_to_be_compared, original_text):
    dic = {}
    df_ = df.copy(deep=True)
    dic["id"] = df["id"]  # for merge with df
    # New empty feature lists
    dic["nbWords_" + column_to_be_compared] = df[column_to_be_compared].apply(lambda x: len(x))
    dic["total_length_char_" + column_to_be_compared] = df[column_to_be_compared].apply(lambda x: len(x))
    dic["nb_same_words_research_" + column_to_be_compared] = []
    dic["nb_similar_words_research_" + column_to_be_compared] = []
    dic["max_len_same_words_research_" + column_to_be_compared] = []
    dic["mean_len_same_words_research_" + column_to_be_compared] = []
    dic["std_len_same_words_research_" + column_to_be_compared] = []
    dic["max_freq_same_words_research_" + column_to_be_compared] = []
    dic["mean_freq_same_words_research_" + column_to_be_compared] = []
    dic["min_freq_same_words_research_" + column_to_be_compared] = []
    dic["max_len_similar_words_research_" + column_to_be_compared] = []
    dic["mean_len_similar_words_research_" + column_to_be_compared] = []
    dic["std_len_similar_words_research_" + column_to_be_compared] = []
    dic["max_freq_similar_words_research_" + column_to_be_compared] = []
    dic["mean_freq_similar_words_research_" + column_to_be_compared] = []
    dic["min_freq_similar_words_research_" + column_to_be_compared] = []
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
    for i in range(df_.shape[0]):
        nb = len(df_[column_to_be_compared].iloc[i])
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
            dic["max_freq_same_words_research_" + column_to_be_compared].append(None)
            dic["mean_freq_same_words_research_" + column_to_be_compared].append(None)
            dic["min_freq_same_words_research_" + column_to_be_compared].append(None)
            dic["max_freq_similar_words_research_" + column_to_be_compared].append(None)
            dic["mean_freq_similar_words_research_" + column_to_be_compared].append(None)
            dic["min_freq_similar_words_research_" + column_to_be_compared].append(None)
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
            freq_same_word = []
            len_similar_word =[]
            freq_similar_word = []
            nbre_corr = df_["nbre_corr_search"].iloc[i]
            research_words = df_["corrected_research_words"].iloc[i]
            if nbre_corr != 0:
                for word_ in df_["research_words"].iloc[i]:
                    if word_ not in research_words:
                        research_words.append(word_)
            for word in research_words:
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
                        if s_word in dic_words:
                            freq_same_word.append(dic_words[s_word])
                        else:
                            freq_same_word.append(0)
                    if dis_L_normalised2[-1] > 0.7:
                        len_similar_word.append(len(word))
                        if s_word in dic_words:
                            freq_similar_word.append(dic_words[s_word])  # Fréquence du mots de la description
                        else:
                            freq_similar_word.append(0)
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
                dic["max_freq_same_words_research_" + column_to_be_compared].append(max(freq_same_word))
                dic["mean_freq_same_words_research_" + column_to_be_compared].append(mean(freq_same_word))
                dic["min_freq_same_words_research_" + column_to_be_compared].append(min(freq_same_word))
                if len(len_same_word) > 1:
                    dic["std_len_same_words_research_" + column_to_be_compared].append(stdev(len_same_word))
                else:
                    dic["std_len_same_words_research_" + column_to_be_compared].append(-1)
            else:
                dic["max_len_same_words_research_" + column_to_be_compared].append(0)
                dic["mean_len_same_words_research_" + column_to_be_compared].append(0)
                dic["std_len_same_words_research_" + column_to_be_compared].append(0)
                dic["max_freq_same_words_research_" + column_to_be_compared].append(0)
                dic["mean_freq_same_words_research_" + column_to_be_compared].append(0)
                dic["min_freq_same_words_research_" + column_to_be_compared].append(0)
            if len_similar_word != []:
                dic["max_len_similar_words_research_" + column_to_be_compared].append(max(len_similar_word))
                dic["mean_len_similar_words_research_" + column_to_be_compared].append(mean(len_similar_word))
                dic["max_freq_similar_words_research_" + column_to_be_compared].append(max(freq_similar_word))
                dic["mean_freq_similar_words_research_" + column_to_be_compared].append(mean(freq_similar_word))
                dic["min_freq_similar_words_research_" + column_to_be_compared].append(min(freq_similar_word))
                if len(len_similar_word) > 1:
                    dic["std_len_similar_words_research_" + column_to_be_compared].append(stdev(len_similar_word))
                else:
                    dic["std_len_similar_words_research_" + column_to_be_compared].append(-1)
            else:
                dic["max_len_similar_words_research_" + column_to_be_compared].append(0)
                dic["mean_len_similar_words_research_" + column_to_be_compared].append(0)
                dic["std_len_similar_words_research_" + column_to_be_compared].append(0)
                dic["max_freq_similar_words_research_" + column_to_be_compared].append(0)
                dic["mean_freq_similar_words_research_" + column_to_be_compared].append(0)
                dic["min_freq_similar_words_research_" + column_to_be_compared].append(0)
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
        df_[col] = dic_[col]
    return df_


# New features
print("STEP 10: Ajout de features sur la recherche faite")
df_all["nbWords_search_terms"] = df_all.research_words.apply(lambda x: len(x))
df_all["total_search_terms_length"] = df_all.search_term.apply(lambda x: len(x))


# New features (with brand)
print("STEP 11: Features avec brand - si une marque apparait dans la recherche")
def compare_with_brand(df):
    dic = {}
    dic["id"] = df["id"]  # for merge with df
    # New empty feature lists
    dic["brand_nb_same_words"] = []
    dic["brand_total_length_same_words"] = []
    dic["brand_percentage_length_same_words"] = []
    # Distances between research and brand
    dic["mean_distance_brand_L"] = []
    dic["min_distance_brand_L"] = []
    dic["mean_distance_brand_J"] = []
    dic["min_distance_brand_J"] = []
    dic["mean_distance_brand_LN"] = []
    dic["min_distance_brand_LN"] = []
    dic["mean_distance_brand_JN"] = []
    dic["min_distance_brand_JN"] = []
    for i in range(df_all.shape[0]): 
        # Brand words
        dist_brand_Le = []
        dist_brand_Jaro = []
        dist_brand_Le_N = []
        dist_brand_Jaro_N = []
        brand_same = []
        nbre_corr = df_all["nbre_corr_search"].iloc[i]
        research_words = df_all["corrected_research_words"].iloc[i]
        if nbre_corr != 0:
            for word_ in df_all["research_words"].iloc[i]:
                if word_ not in research_words:
                    research_words.append(word_)
        for word in research_words:
            if df["brand"].iloc[i] == "":
                dist_brand_Le.append(-1)
                dist_brand_Jaro.append(-1)
                dist_brand_Le_N.append(-1)
                dist_brand_Jaro_N.append(-1)
            else:
                for s_word in df['brand_words'].iloc[i]:
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
        dic["brand_nb_same_words"].append(len(brand_same))
        dic["brand_total_length_same_words"].append(sum(map(lambda x:len(x), brand_same)))
        if df_all.iloc[i]["brand_words"] != []:                                
            dic["brand_percentage_length_same_words"].append((sum(map(lambda x:len(x), brand_same))/sum(map(lambda x:len(x), df.iloc[i]["brand_words"])))) # matching percentage
        else:
            dic["brand_percentage_length_same_words"].append(0)
        dic["mean_distance_brand_L"].append(mean(dist_brand_Le))
        dic["min_distance_brand_L"].append(min(dist_brand_Le))
        dic["mean_distance_brand_J"].append(mean(dist_brand_Jaro))
        dic["min_distance_brand_J"].append(min(dist_brand_Jaro))
        dic["mean_distance_brand_LN"].append(mean(dist_brand_Le_N))
        dic["min_distance_brand_LN"].append(min(dist_brand_Le_N))
        dic["mean_distance_brand_JN"].append(mean(dist_brand_Jaro_N))
        dic["min_distance_brand_JN"].append(min(dist_brand_Jaro_N))
    # Ajout des features dans df
    dic_ = pd.DataFrame(dic)
    for col in dic_.columns:
        df[col] = dic_[col]
    return df


df_all = compare_with_brand(df_all)

# Ajout de features avec la fonction
print("STEP 12: Ajout de features avec la fonction compare_research_word")
df_all = compare_research_words(df_all, column_to_be_compared='product_description_words', original_text='product_description')
print("STEP 12: 1/3")
df_all = compare_research_words(df_all, column_to_be_compared='product_words', original_text='product_title')
print("STEP 12: 2/3")
df_all = compare_research_words(df_all, column_to_be_compared="list_attributes_words", original_text="list_attributes")
# dic = compare_research_words(df_all, column_to_be_compared="list_attributes_words", original_text="list_attributes")
# Suppression des colonnes inutiles
print("STEP 13: Ecriture / Sauvegarde des données extraites")
df_model = df_all.drop(['product_uid','product_title', 'search_term', "corrected_search_terms", 'product_description', 'brand', 'product_words', 'research_words',
                        'brand_words', 'product_description_words', 'list_attributes_words', 'list_attributes'], axis=1)
features = df_model.columns
print(features)
df_model.to_csv('dataframe.csv', index=False)
