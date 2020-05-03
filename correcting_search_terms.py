# Imports

import pandas as pd
import numpy as np
from collections import Counter
import re

from spellCorrector import SymSpell
from itertools import chain
from nltk.metrics.distance import jaro_similarity

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


# Fonction pour nettoyer les données du dataset attributes
def str_cleaning(s): 
    if isinstance(s, str):
        # Uniformisation pour le systeme unitaire
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", "inches ", s) # Uniformise les distances inch en [0-9+]in.
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", "feet ", s) # Uniformise les distances feet en [0-9+]ft.
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", "pounds ", s) # same pour poids
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", "square feet ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", "cubic feet ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gallons ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 ounces ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 centimeters ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 milimeters ", s)
        s = re.sub(r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 degrees ", s)
        s = re.sub(r"([0-9]+)( *)(v.|volts|volt)", r"\1 volts ", s)
        s = re.sub(r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watts ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amps ", s)
        s = re.sub(r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. ", s)
        s = re.sub(r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hours ", s)
        s = re.sub(r"([a-zA-Z])\.", r"\1", s)
        car = ["[", "]", "?", "(", ")", ",", ".", ";", "/", "-", "\\"]  # "." utilisé pour les chiffres
        for char in car:
            s = s.replace(char, " ")
        s = re.sub(r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gallons.perminute ", s)
        s = re.sub(r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gallons.perhour ", s)
        s = re.sub(r"(#+)([0-9a-zA-Z]+)", "", s)
        # Suppression de tous les chiffres
        s = re.sub(r"([0-9])( *)/( *)([0-9])", "", s)
        s = re.sub(r"([0-9]+)( *)x( *)([0-9]+)", "", s)
        s = re.sub(r"([0-9])", "", s)
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", "", s) 
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
        s = s.replace("[ ]?[[(].+?[])]","")
        # remove sizes
        s = s.replace("size: .+$","")
        s = s.replace("size [0-9]+[.]?[0-9]+\\b","")
        return s.lower()
    else:
        return ""


def str_cleaning_research(s): 
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
        
        return s.lower()
    else:
        return ""


def attr_treatment(s):
    if isinstance(s, str):
        s = re.sub(r"([0-9]+)", "", s)
        s = re.sub(r"([Bb]ullet)([0-9]+)", "", s)
        s = re.sub(r"([Mm][Ff][Gg] [Bb]rand [Nn]ame)", "", s)
        s = re.sub(r"([Mm]aterial)", "", s)
        return s
    else:
        return s


print("STEP 1: Création dataset complet")
df_all = pd.merge(df, desc, how='right', on='product_uid')

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

df_attr = pd.DataFrame({"product_uid": df_all.product_uid.unique(), "list_attributes": list_attributes})
df_all = pd.merge(df_all, df_attr, how='left', on="product_uid")

print("STEP 2: Correction termes de recherche")
corpus = []
df_all["list_attributes"] = df_all["list_attributes"].apply(lambda x : str_cleaning(x))
df_all["product_description"] = df_all["product_description"].apply(lambda x : str_cleaning(x))
df_all["product_title"] = df_all["product_title"].apply(lambda x : str_cleaning(x))
df_all["search_term"] = df_all["search_term"].apply(lambda x : str_cleaning_research(x))

corpus.extend(list(chain.from_iterable(df_all["list_attributes"].apply(lambda x: x.split(' ')))))
corpus.extend(list(chain.from_iterable(df_all["product_description"].apply(lambda x: x.split(' ')))))
corpus.extend(list(chain.from_iterable(df_all["product_title"].apply(lambda x: x.split(' ')))))

print("build symspell")
cor = Counter(corpus)
corpus_dir = 'spellChecker/'
corpus_file_name = 'spell_check_dictionary.txt'
symspell = SymSpell(verbose=10)
symspell.build_vocab(dictionary=cor, file_dir=corpus_dir, file_name=corpus_file_name)
symspell.load_vocab(corpus_file_path=corpus_dir+corpus_file_name)


print('corrections')

def return_most_probable_corr(x):
    y = symspell.corrections(x)
    if len(y) == 1:
        return y[0]["word"], y[0]["distance"]
    if len(y) > 1:
        print("longueur sup à 1 pour ", x, ' obtention de y : ', y)
        word = y[0]["word"]
        proba = y[0]["distance"]
        for i in range(1, len(y)):
            if y[i]["distance"] < proba:
                word = y[i]["word"]
                proba = y[i]["distance"]
            if y[i]["distance"] == proba:
                search_term = re.sub("( )+", "", x)
                correction1 = re.sub("( )+", "", word)
                correction2 = re.sub("( )+", "", y[i]["word"])
                if jaro_similarity(correction1, search_term) < jaro_similarity(correction2, search_term):
                    word = y[i]["word"]
                    proba = y[i]["distance"]
            
        return word, proba


def return_most_probable_corrs(x):
    y = ""
    distance = 0
    for elt in x:
        if elt == "":
            continue
        elif re.sub(r"([0-9])", "", elt) != elt:
            y += elt + " "
        else:
            y_elt = symspell.best_correction(elt)
            y += y_elt[0]["word"] + " "
            distance += y_elt[0]["distance"]  
    return y, distance

df_all["search_terms"] = df_all["search_term"].apply(lambda x: x.split(" "))
resultats = df_all["search_terms"].apply(lambda x: return_most_probable_corrs(x))
df_all["corrected_search_terms"] = [elt[0] for elt in resultats]
df_all["nbre_corr_search"] = [elt[1] for elt in resultats]

# df_all[["corrected_search_terms", "nbre_corr_search"]] = df_all["search_term"].apply(lambda x: return_most_probable_corr(x))
df_all[['id', "search_term", "corrected_search_terms", "nbre_corr_search"]].to_csv("corrected_search_terms.csv", index=False)




