# Imports
import pandas as pd
import numpy as np
from statistics import mean, stdev
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from scipy.sparse import csr_matrix

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# Import data
dataframe = pd.read_csv('dataframe_NLP.csv', header=0, sep=",", encoding="ISO-8859-1")
dataframe["product_words"] = dataframe["product_words"].apply(lambda x: eval(x))
dataframe["research_words"] = dataframe["research_words"].apply(lambda x: eval(x))
dataframe["brand_words"] = dataframe["brand_words"].apply(lambda x: eval(x))
dataframe["product_description_words"] = dataframe["product_description_words"].apply(lambda x: eval(x))
dataframe["list_attributes_words"] = dataframe["list_attributes_words"].apply(lambda x: eval(x))
dataframe["corrected_research_words"] = dataframe["corrected_research_words"].apply(lambda x: eval(x))

# Create corpus from dataframe
corpus = list(dataframe["product_words"])
piv_corpus = len(corpus)
corpus += list(dataframe["product_description_words"])
corpus += list(dataframe["list_attributes_words"])
corpus += list(dataframe["brand_words"])
corpus += list(dataframe["corrected_research_words"])

def remove_figures(vec):
    new_vec = []
    for elt in vec:
        if re.sub(r"([0-9]+)", "f", elt) != "f":
            new_vec.append(elt)
    return new_vec

corpus = list(map(remove_figures, corpus))

def join_list(lt):
    return " ".join(lt)

corpus = list(map(join_list, corpus))

# TF-IDF
vec = TfidfVectorizer()
vec1 = vec.fit_transform(corpus)

# Attribution aux variables
title = vec1[0:piv_corpus]
description = vec1[piv_corpus:2*piv_corpus]
attributes = vec1[2*piv_corpus:3*piv_corpus]
brand = vec1[3*piv_corpus:4*piv_corpus]
search = vec1[4*piv_corpus:5*piv_corpus]

# Nouvelles features
produit_vec1 = []
produit_vec2 = []
produit_vec3 = []
produit_vec4 = []

car1_title = [] # min diff de 0
car2_title = [] # max
car3_title = [] # mean without 0
car4_title = [] # nbre diff de 0
car5_title = [] # somme

car1_description = [] # min diff de 0
car2_description = [] # max
car3_description = [] # mean without 0
car4_description = [] # nbre diff de 0
car5_description = [] # somme

car1_attributes = [] # min diff de 0
car2_attributes = [] # max
car3_attributes = [] # mean without 0
car4_attributes = [] # nbre diff de 0
car5_attributes = [] # somme

car1_brand = [] # min diff de 0
car2_brand = [] # max
car3_brand = [] # mean without 0
car4_brand = [] # nbre diff de 0
car5_brand = [] # somme

car1_search = [] # min diff de 0
car2_search = [] # max
car3_search = [] # mean without 0
car4_search = [] # nbre diff de 0
car5_search = [] # somme

for i in range(piv_corpus):
    elt = list(title[i].nonzero()[1])
    elt = title[i].toarray()[0][elt]
    if elt != []:
        car1_title.append(min(elt)) # min diff de 0
        car2_title.append(max(elt)) # max
        car3_title.append(mean(elt)) # mean without 0
        car4_title.append(len(elt)) # nbre diff de 0
        car5_title.append(sum(elt)) # somme
    else:
        car1_title.append(0) # min diff de 0
        car2_title.append(0) # max
        car3_title.append(0) # mean without 0
        car4_title.append(0) # nbre diff de 0
        car5_title.append(0) # somme
    #
    elt = list(description[i].nonzero()[1])
    elt = description[i].toarray()[0][elt]
    if elt != []:
        car1_description.append(min(elt)) # min diff de 0
        car2_description.append(max(elt)) # max
        car3_description.append(mean(elt)) # mean without 0
        car4_description.append(len(elt)) # nbre diff de 0
        car5_description.append(sum(elt)) # somme
    else:
        car1_description.append(0) # min diff de 0
        car2_description.append(0) # max
        car3_description.append(0) # mean without 0
        car4_description.append(0) # nbre diff de 0
        car5_description.append(0) # somme        
    #
    elt = list(attributes[i].nonzero()[1])
    elt = attributes[i].toarray()[0][elt]
    if elt != []:
        car1_attributes.append(min(elt)) # min diff de 0
        car2_attributes.append(max(elt)) # max
        car3_attributes.append(mean(elt)) # mean without 0
        car4_attributes.append(len(elt)) # nbre diff de 0
        car5_attributes.append(sum(elt)) # somme
    else:
        car1_attributes.append(0) # min diff de 0
        car2_attributes.append(0) # max
        car3_attributes.append(0) # mean without 0
        car4_attributes.append(0) # nbre diff de 0
        car5_attributes.append(0) # somme
    #
    elt = list(brand[i].nonzero()[1])
    elt = brand[i].toarray()[0][elt]
    if elt != []:
        car1_brand.append(min(elt)) # min diff de 0
        car2_brand.append(max(elt)) # max
        car3_brand.append(mean(elt)) # mean without 0
        car4_brand.append(len(elt)) # nbre diff de 0
        car5_brand.append(sum(elt)) # somme
    else:
        car1_brand.append(0) # min diff de 0
        car2_brand.append(0) # max
        car3_brand.append(0) # mean without 0
        car4_brand.append(0) # nbre diff de 0
        car5_brand.append(0) # somme
    #
    elt = list(search[i].nonzero()[1])
    elt = search[i].toarray()[0][elt]
    if elt != []:
        car1_search.append(min(elt)) # min diff de 0
        car2_search.append(max(elt)) # max
        car3_search.append(mean(elt)) # mean without 0
        car4_search.append(len(elt)) # nbre diff de 0
        car5_search.append(sum(elt)) # somme
    else:
        car1_search.append(0) # min diff de 0
        car2_search.append(0) # max
        car3_search.append(0) # mean without 0
        car4_search.append(0) # nbre diff de 0
        car5_search.append(0) # somme
    #
    produit_vec1.append(np.vdot(title[i].toarray()[0], search[i].toarray()[0]))
    produit_vec2.append(np.vdot(description[i].toarray()[0], search[i].toarray()[0]))
    produit_vec3.append(np.vdot(attributes[i].toarray()[0], search[i].toarray()[0]))
    produit_vec4.append(np.vdot(brand[i].toarray()[0], search[i].toarray()[0]))

# Avec .idf_
# Vect with 1 and 0
# Every non nul values set to 1
keep = vec1.nonzero()
n_keep = len(keep[0])

relu_vec1 = csr_matrix((np.ones(n_keep), (keep[0], keep[1])), shape=vec1.shape)
relu_title = relu_vec1[0:piv_corpus]
relu_description = relu_vec1[piv_corpus:2*piv_corpus]
relu_attributes = relu_vec1[2*piv_corpus:3*piv_corpus]
relu_brand = relu_vec1[3*piv_corpus:4*piv_corpus]
relu_search = relu_vec1[4*piv_corpus:5*piv_corpus]

# More features with .idf_
vec = vec.idf_

produit_vec5 = []
produit_vec6 = []
produit_vec7 = []
produit_vec8 = []

produit_vec9 = []
produit_vec10 = []
produit_vec11 = []
produit_vec12 = []

car1_rv_search = [] # min diff de 0
car2_rv_search = [] # max
car3_rv_search = [] # mean without 0
car4_rv_search = [] # nbre diff de 0
car5_rv_search = [] # somme

car1_rv_title = [] # min diff de 0
car2_rv_title = [] # max
car3_rv_title = [] # mean without 0
car4_rv_title = [] # nbre diff de 0
car5_rv_title = [] # somme

car1_rv_description = [] # min diff de 0
car2_rv_description = [] # max
car3_rv_description = [] # mean without 0
car4_rv_description = [] # nbre diff de 0
car5_rv_description = [] # somme

car1_rv_attributes = [] # min diff de 0
car2_rv_attributes = [] # max
car3_rv_attributes = [] # mean without 0
car4_rv_attributes = [] # nbre diff de 0
car5_rv_attributes = [] # somme

car1_rv_brand = [] # min diff de 0
car2_rv_brand = [] # max
car3_rv_brand = [] # mean without 0
car4_rv_brand = [] # nbre diff de 0
car5_rv_brand = [] # somme

for i in range(piv_corpus):
    r_title = relu_title[i].toarray()[0]
    r_description = relu_description[i].toarray()[0]
    r_attributes = relu_attributes[i].toarray()[0]
    r_brand = relu_brand[i].toarray()[0]
    r_search = relu_search[i].toarray()[0]
    produit_vec5.append(np.vdot(r_title, r_search))
    produit_vec6.append(np.vdot(r_description, r_search))
    produit_vec7.append(np.vdot(r_attributes, r_search))
    produit_vec8.append(np.vdot(r_brand, r_search))
    #
    r_search = relu_search[i].toarray()[0]
    #
    a = relu_search[i].toarray()[0]*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    if elt != []:
        car1_rv_search.append(min(elt)) # min diff de 0
        car2_rv_search.append(max(elt)) # max
        car3_rv_search.append(mean(elt)) # mean without 0
        car4_rv_search.append(len(elt)) # nbre diff de 0
        car5_rv_search.append(sum(elt)) # somme
    else:
        car1_rv_search.append(0) # min diff de 0
        car2_rv_search.append(0) # max
        car3_rv_search.append(0) # mean without 0
        car4_rv_search.append(0) # nbre diff de 0
        car5_rv_search.append(0) # somme
    #
    a = relu_title[i].toarray()[0]*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    if elt != []:
        car1_rv_title.append(min(elt)) # min diff de 0
        car2_rv_title.append(max(elt)) # max
        car3_rv_title.append(mean(elt)) # mean without 0
        car4_rv_title.append(len(elt)) # nbre diff de 0
        car5_rv_title.append(sum(elt)) # somme
    else:
        car1_rv_title.append(0) # min diff de 0
        car2_rv_title.append(0) # max
        car3_rv_title.append(0) # mean without 0
        car4_rv_title.append(0) # nbre diff de 0
        car5_rv_title.append(0) # somme
    produit_vec9.append(np.vdot(a, r_search))
    #
    a = relu_description[i].toarray()[0]*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    if elt != []:
        car1_rv_description.append(min(elt)) # min diff de 0
        car2_rv_description.append(max(elt)) # max
        car3_rv_description.append(mean(elt)) # mean without 0
        car4_rv_description.append(len(elt)) # nbre diff de 0
        car5_rv_description.append(sum(elt)) # somme
    else:
        car1_rv_description.append(0) # min diff de 0
        car2_rv_description.append(0) # max
        car3_rv_description.append(0) # mean without 0
        car4_rv_description.append(0) # nbre diff de 0
        car5_rv_description.append(0) # somme
    produit_vec10.append(np.vdot(a, r_search))
    #
    a = relu_attributes[i].toarray()[0]*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    if elt != []:
        car1_rv_attributes.append(min(elt)) # min diff de 0
        car2_rv_attributes.append(max(elt)) # max
        car3_rv_attributes.append(mean(elt)) # mean without 0
        car4_rv_attributes.append(len(elt)) # nbre diff de 0
        car5_rv_attributes.append(sum(elt)) # somme
    else:
        car1_rv_attributes.append(0) # min diff de 0
        car2_rv_attributes.append(0) # max
        car3_rv_attributes.append(0) # mean without 0
        car4_rv_attributes.append(0) # nbre diff de 0
        car5_rv_attributes.append(0) # somme
    produit_vec11.append(np.vdot(a, r_search))
    #
    a = relu_brand[i].toarray()[0]*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    if elt != []:
        car1_rv_brand.append(min(elt)) # min diff de 0
        car2_rv_brand.append(max(elt)) # max
        car3_rv_brand.append(mean(elt)) # mean without 0
        car4_rv_brand.append(len(elt)) # nbre diff de 0
        car5_rv_brand.append(sum(elt)) # somme
    else:
        car1_rv_brand.append(0) # min diff de 0
        car2_rv_brand.append(0) # max
        car3_rv_brand.append(0) # mean without 0
        car4_rv_brand.append(0) # nbre diff de 0
        car5_rv_brand.append(0) # somme
    produit_vec12.append(np.vdot(a, r_search))

# New dataframe with features
df_model = pd.DataFrame({"id":dataframe.id, "f1":produit_vec1})
df_model["f2"] = produit_vec2
df_model["f3"] = produit_vec3
df_model["f4"] = produit_vec4
df_model["f5"] = produit_vec5
df_model["f6"] = produit_vec6
df_model["f7"] = produit_vec7
df_model["f8"] = produit_vec8
df_model["f9"] = produit_vec9
df_model["f10"] = produit_vec10
df_model["f11"] = produit_vec11
df_model["f12"] = produit_vec12
df_model["car1_title"] = car1_title
df_model["car2_title"] = car2_title
df_model["car3_title"] = car3_title
df_model["car4_title"] = car4_title
df_model["car5_title"] = car5_title
df_model["car1_description"] = car1_description
df_model["car2_description"] = car2_description
df_model["car3_description"] = car3_description
df_model["car4_description"] = car4_description
df_model["car5_description"] = car5_description
df_model["car1_attributes"] = car1_attributes
df_model["car2_attributes"] = car2_attributes
df_model["car3_attributes"] = car3_attributes
df_model["car4_attributes"] = car4_attributes
df_model["car5_attributes"] = car5_attributes
df_model["car1_brand"] = car1_brand
df_model["car2_brand"] = car2_brand
df_model["car3_brand"] = car3_brand
df_model["car4_brand"] = car4_brand
df_model["car5_brand"] = car5_brand
df_model["car1_search"] = car1_search
df_model["car2_search"] = car2_search
df_model["car3_search"] = car3_search
df_model["car4_search"] = car4_search
df_model["car5_search"] = car5_search
df_model["car1_rv_search"] = car1_rv_search
df_model["car2_rv_search"] = car2_rv_search
df_model["car3_rv_search"] = car3_rv_search
df_model["car4_rv_search"] = car4_rv_search
df_model["car5_rv_search"] = car5_rv_search
df_model["car1_rv_title"] = car1_rv_title
df_model["car2_rv_title"] = car2_rv_title
df_model["car3_rv_title"] = car3_rv_title
df_model["car4_rv_title"] = car4_rv_title
df_model["car5_rv_title"] = car5_rv_title
df_model["car1_rv_description"] = car1_rv_description
df_model["car2_rv_description"] = car2_rv_description
df_model["car3_rv_description"] = car3_rv_description
df_model["car4_rv_description"] = car4_rv_description
df_model["car5_rv_description"] = car5_rv_description
df_model["car1_rv_attributes"] = car1_rv_attributes
df_model["car2_rv_attributes"] = car2_rv_attributes
df_model["car3_rv_attributes"] = car3_rv_attributes
df_model["car4_rv_attributes"] = car4_rv_attributes
df_model["car5_rv_attributes"] = car5_rv_attributes
df_model["car1_rv_brand"] = car1_rv_brand
df_model["car2_rv_brand"] = car2_rv_brand
df_model["car3_rv_brand"] = car3_rv_brand
df_model["car4_rv_brand"] = car4_rv_brand
df_model["car5_rv_brand"] = car5_rv_brand

df_model.to_csv("TF_IDF_features.csv", index=False)