from __future__ import print_function
import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk.collocations 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.manifold import TSNE


# import LDA_trained files
base_dir = '/Users/osuhyeon/NLTK_court_document/1 LDA/'
LDA_trained_files = [
    'adidas_LDA_nparray', 
    'JennyYoo_LDA_nparray',
    'ChristianLouboutin_LDA_nparray',
    'PUMA_LDA_nparray',
    'AtelierLuxuryGroup_LDA_nparray',
    'Versace_LDA_nparray',
    'LouisVuitton_LDA_nparray'
    ]

# import LDA_trained perplexity files
LDA_trained_perplexity_files = [
    'adidas_LDAprint_Perplexity', 
    'JennyYoo_LDAprint_Perplexity',
    'ChristianLouboutin_LDAprint_Perplexity',
    'PUMA_LDAprint_Perplexity',
    'AtelierLuxuryGroup_LDAprint_Perplexity',
    'Versace_LDAprint_Perplexity',
    'LouisVuitton_LDAprint_Perplexity'
    ]


# import individual LDA_trained numpy
n_compoents = 11
target = LDA_trained_files[6]
npy_file = str(base_dir + target + '.npy')
npy_train_data = np.load(npy_file)
print("type(npy_train_data):", type(npy_train_data))
print("np.shape(npy_train_data):", np.shape(npy_train_data))


# import individual LDA_trained perplexity value
npy_perplexity_file = base_dir + LDA_trained_perplexity_files[6] + '.txt'
npy_train_perplexity = open(npy_perplexity_file, 'r', encoding = "UTF-8").read()
npy_train_perplexity_value = re.sub(r'\W', '', npy_train_perplexity)


# import file TSNE.py
# import class TSNE()
from TSNE import TSNE
tsne_operator = TSNE()
tsne_y = tsne_operator.TSNE(
    n_components = 3, 
    lda_trained_data = npy_train_data,
    lda_train_perplexity = npy_train_perplexity_value
    )


# re.sub(pattern, repl, string, count, flags)
save_dir = "/Users/osuhyeon/NLTK_court_document/0 test_results/"
name = re.sub("_LDA_nparray", ".png", str(target))
save_name = save_dir + name

# file TSNE_vis_3d.py
# class tsne_vis()
from TSNE_vis import tsne_vis_3d
tsne_vis_operator = tsne_vis_3d()
plot3d = tsne_vis_operator.Axes3D_tsne_scatterplot(
    tsne_data = tsne_y, 
    lda_data = npy_train_data,
    save_name = save_name
    )
