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

# TF-IDF N-gram
# vec = TfidfVectorizer()
vec = TfidfVectorizer(ngram_range=(1, 3))
vec1 = vec.fit_transform(corpus)
vec1.shape
vec = vec.idf_

title = vec1[0:piv_corpus]
description = vec1[piv_corpus:2*piv_corpus]
attributes = vec1[2*piv_corpus:3*piv_corpus]
brand = vec1[3*piv_corpus:4*piv_corpus]
search = vec1[4*piv_corpus:5*piv_corpus]

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


def ajout_caracterisques_vec(vect_analyse, vect_features):
    if vect_analyse != []:
        vect_features.append(min(vect_analyse)) # min diff de 0
        vect_features.append(max(vect_analyse)) # max
        vect_features.append(mean(vect_analyse)) # mean without 0
        vect_features.append(len(vect_analyse)) # nbre diff de 0
        vect_features.append(sum(vect_analyse)) # somme
    else:
        vect_features.append(0) # min diff de 0
        vect_features.append(0) # max
        vect_features.append(0) # mean without 0
        vect_features.append(0) # nbre diff de 0
        vect_features.append(0) # somme
    return vect_features

# Featuring
def get_features(title_i, description_i, attributes_i, brand_i, search_i,
                 relu_title_i, relu_description_i, relu_attributes_i, relu_brand_i, relu_search_i):
    features = []
    #
    elt = list(title_i.nonzero()[1])
    elt = title_i.toarray()[0][elt]
    features = ajout_caracterisques_vec(elt, features)
    #
    elt = list(description_i.nonzero()[1])
    elt = description_i.toarray()[0][elt]
    features = ajout_caracterisques_vec(elt, features) 
    #
    elt = list(attributes_i.nonzero()[1])
    elt = attributes_i.toarray()[0][elt]
    features = ajout_caracterisques_vec(elt, features)
    #
    elt = list(brand_i.nonzero()[1])
    elt = brand_i.toarray()[0][elt]
    features = ajout_caracterisques_vec(elt, features)
    #
    elt = list(search_i.nonzero()[1])
    elt = search_i.toarray()[0][elt]
    features = ajout_caracterisques_vec(elt, features)
    #
    # Produit vectoriel avec termes de recherche
    #
    features.append(np.vdot(title_i.toarray()[0], search_i.toarray()[0]))
    features.append(np.vdot(description_i.toarray()[0], search_i.toarray()[0]))
    features.append(np.vdot(attributes_i.toarray()[0], search_i.toarray()[0]))
    features.append(np.vdot(brand_i.toarray()[0], search_i.toarray()[0]))
    # Produit vectoriel avec Title
    features.append(np.vdot(description_i.toarray()[0], title_i.toarray()[0]))
    features.append(np.vdot(attributes_i.toarray()[0], title_i.toarray()[0]))
    features.append(np.vdot(brand_i.toarray()[0], title_i.toarray()[0]))
    # Produit vectoriel avec description
    features.append(np.vdot(attributes_i.toarray()[0], description_i.toarray()[0]))
    features.append(np.vdot(brand_i.toarray()[0], description_i.toarray()[0]))
    # Produit vectoriel avec attributes_i
    features.append(np.vdot(brand_i.toarray()[0], attributes_i.toarray()[0]))
    #
    r_title = relu_title_i.toarray()[0]
    r_description = relu_description_i.toarray()[0]
    r_attributes = relu_attributes_i.toarray()[0]
    r_brand = relu_brand_i.toarray()[0]
    r_search = relu_search_i.toarray()[0]
    # Features search
    features.append(np.vdot(r_title, r_search))
    features.append(np.vdot(r_description, r_search))
    features.append(np.vdot(r_attributes, r_search))
    features.append(np.vdot(r_brand, r_search))
    # Features title
    features.append(np.vdot(r_description, r_title))
    features.append(np.vdot(r_attributes, r_title))
    features.append(np.vdot(r_brand, r_title))
    # Features description
    features.append(np.vdot(r_attributes, r_description))
    features.append(np.vdot(r_brand, r_description))
    # Features description
    features.append(np.vdot(r_brand, r_attributes))
    #
    a = r_search*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    features = ajout_caracterisques_vec(elt, features)
    #
    a = r_title*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    features = ajout_caracterisques_vec(elt, features)
    features.append(np.vdot(a, r_search))
    #
    a = r_description*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    features = ajout_caracterisques_vec(elt, features)
    features.append(np.vdot(a, r_search))
    #
    a = r_attributes*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    features = ajout_caracterisques_vec(elt, features)
    features.append(np.vdot(a, r_search))
    #
    a = r_brand*vec
    elt = list(a.nonzero()[0])
    elt = a[elt]
    features = ajout_caracterisques_vec(elt, features)
    features.append(np.vdot(a, r_search))
    #
    return features


resultats = list(map(get_features, title, description, attributes, brand, search, relu_title, relu_description, relu_attributes, relu_brand, relu_search))
columns_name = ["car1_title", "car2_title", "car3_title", "car4_title", "car5_title",
                "car1_description", "car2_description", "car3_description", "car4_description", "car5_description",
                "car1_attributes", "car2_attributes", "car3_attributes", "car4_attributes", "car5_attributes",
                "car1_brand", "car2_brand", "car3_brand", "car4_brand", "car5_brand",
                "car1_search", "car2_search", "car3_search", "car4_search", "car5_search",
                "pv1_search", "pv2_search", "pv3_search", "pv4_search",
                "pv1_title", "pv2_title", "pv3_title",
                "pv1_description", "pv2_description", "pv1_attributes",
                "pv1_r_search", "pv2_r_search", "pv3_r_search", "pv4_r_search",
                "pv1_r_title", "pv2_r_title", "pv3_r_title",
                "pv1_r_description", "pv2_r_description", "pv1_r_attributes",
                "car1_rv_search", "car2_rv_search", "car3_rv_search", "car4_rv_search", "car5_rv_search",
                "car1_rv_title", "car2_rv_title", "car3_rv_title", "car4_rv_title", "car5_rv_title", "pv_r_title_search", 
                "car1_rv_description", "car2_rv_description", "car3_rv_description", "car4_rv_description", "car5_rv_description", "pv_r_desc_search", 
                "car1_rv_attributes", "car2_rv_attributes", "car3_rv_attributes", "car4_rv_attributes", "car5_rv_attributes", "pv_r_attr_search",
                "car1_rv_brand", "car2_rv_brand", "car3_rv_brand", "car4_rv_brand", "car5_rv_brand", "pv_r_brd_search"]
dic = {}
for nb, feat in enumerate(columns_name):
    dic[feat] = [x[nb] for x in resultats]

df_model = pd.DataFrame(dic)
df_model.to_csv("TF_IDF_features_new.csv", index=False)
