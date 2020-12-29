from __future__ import print_function
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk.collocations 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.manifold import TSNE


# Corpus
directory_path = '/Users/osuhyeon/NLTK_court_document/3 TXTDATA/'
Case_file_dict = {
    "Case1_file_path" : [
        'Adidas Complaint 2015-09-14',  
        'Adidas Opinion and Order 2016-02-12', 
        'Adidas Opinion and Order 2016-04-16',
        'Adidas Opinion and Order 2017-08-03',
        'Adidas Court Opinion 2018-03-10'],
    "Case2_file_path" : [
        'JennyYoo Complaint 2018-10-26', 
        'JennyYoo Discovery Order 2019-12-16'],
    "Case3_file_path" : [
        'ChristianLouboutin Complaint 2011-04-07',
        'ChristianLouboutin Court of Appeals for the Second Circuit 2013-03-08',
        'ChristianLouboutin Court Opinion 2011-08-10',
        'ChristianLouboutin Court Opinion 2012-09-05',
        'ChristianLouboutin Order of Dismissal 2012-12-27'],
    "Case4_file_path" : [
        'PUMA Complaint 2017-03-31',
        'PUMA Memorandum in opposition to notice of motion and motion 2017-05-22',
        'PUMA Minutes Order denying plaintiffs motion 2017-06-02', 
        'PUMA Motion for motion 2017-04-11'],
    "Case5_file_path" : [
        'AtelierLuxuryGroup AmendedComplaint 2020-8-24',
        'AtelierLuxuryGroup Complaint 2020-01-22'],
    "Case6_file_path" : [
        'Versace Complaint 2019-11-25'],
    "Case7_file_path" : [
        'LouisVuitton Judge Opinion 2006-06-30',
        'LouisVuitton_Opinion and Order_2007-12-14',
        'LouisVuitton_Opinion_and_Order_2008-04-24',
        'LouisVuitton_Opinion_and_Order_2008-05-30']}

# If all cases had to be analyzed, 
# Use codes below:
# Case_file_dict.values()


# Pre-filtering before CounterVectorize
exclude = ['-', 'et', 'al', 'also', 'B.V.', 'LLC','L.L.C', 'LLP', 'INC', 'Inc','S.r.l', 'S.a.S', 'called', 'would', 'have', 'See Id', 'Id']
exclude.extend([x.lower() for x in exclude])
exclude.extend([x.upper() for x in exclude])

exclude_sub = [re.sub(r"\W", "", x) for x in exclude]
exclude.extend(exclude_sub)
exclude.extend([x.lower() for x in exclude_sub])
exclude.extend([x.upper() for x in exclude_sub])



parties = []
parties_attorneys = "/Users/osuhyeon/NLTK_court_document/3 TXTDATA/[Praties_Attorneys].txt"
parties_file_read = open(parties_attorneys, "r").readlines()

for party in parties_file_read:
    p = re.sub(r"\s", " ", party)
    if p == " ":
        del(p)
    else:
        parties.append(p)

refined_parties = []
for party in parties: 
    party_token = nltk.tokenize.word_tokenize(party)
    newparty_list = [x for x in party_token if x not in exclude]
    newparty_list2 = " ".join(newparty_list)
    refined_parties.append(newparty_list2)

str_parties = " ".join(refined_parties)
tokenized_party_name = nltk.tokenize.word_tokenize(str_parties)

# Complement Exclude_list 2nd
exclude.extend(tokenized_party_name)


# sklearn_ LatentDirichletAllocation()
# input data = after-CounterVectorized NUMPY DATA

# skleran_ Countervectorizer()
# input data = readlines LIST DATA 
# 1 item = 1 sentence

# If ‘file’, the sequence items must have a ‘read’ method (file-like object) that is called to fetch the bytes in memory.
# Otherwise the input is expected to be a sequence of items that can be of type string or byte.
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


# Dictionary
def Corpus(Case_file_path, corpus, word_filter):
    path_list = []
    for file in Case_file_path:
        each_file = directory_path + file + '.txt' 
        path_list.append(each_file)
    
    for file in path_list:
        # each_file_content = list[1 sentence = 1 item]
        
        sentences = open(file, 'r').readlines()

        untokenized_sentences = []
        for sentence in sentences:
            if sentence == "\n":
                del(sentence)
            else:
                newsentence = re.sub(r"[0-9|*\W]+", " ", sentence)
                untokenized_sentences.append(newsentence)

        tokenized = [nltk.tokenize.word_tokenize(untokenized_sentences[i]) for i in range(len(untokenized_sentences))]
        str_sentence = []
        # Access for individual sentence.
        # Becasuse lemmatizing is available for individual word.
        for sentence in tokenized:
            lemmatized_sentence = [WordNetLemmatizer().lemmatize(word, 'v') for word in sentence]
            filtered_sentence = [x for x in lemmatized_sentence if x not in word_filter and len(x) > 2]
            each_str_sentence = ' '.join(filtered_sentence)
            str_sentence.append(each_str_sentence)

        corpus.extend(str_sentence)
        # Be Cautious!
        # corpus.append() : to make dictionary type assorting by each files        
    return corpus


print("Input value referring which case to proceed _")
i = input()
case_list = "Case%d_file_path" % int(i)
a_Case_file_path_list = Case_file_dict[case_list]


corpus = []
Case_corpus = Corpus(
    Case_file_path= a_Case_file_path_list, 
    corpus = corpus, 
    word_filter = exclude)



# Word numbers in a topic according to the Minimum Perplexity
ngram_range = (1,3)
n_features = 1000


# Choose var_n_components
var_n_components_1_20 = [i for i in range(1,21,1)]
stopwords = nltk.corpus.stopwords.words('english')

import Choosing_n_components
perplexity_operator = Choosing_n_components.Choosing_n_components()
lda_train_perplexities = perplexity_operator.Choosing_n_components(
    ngram_range = ngram_range,
    n_features= n_features,
    var_n_components= var_n_components_1_20, 
    train_data= corpus,
    stop_words = stopwords)

lda_train_dict_1_20 = dict(zip(var_n_components_1_20, lda_train_perplexities))
print("lda_train_dict_1_20:", lda_train_dict_1_20)

x_keys = [i for i in lda_train_dict_1_20.keys()]
y_vals = [i for i in lda_train_dict_1_20.values()]
plt.plot(x_keys, y_vals)
plt.xticks(x_keys)

plt.xlabel(xlabel = "Number of Topics")
plt.ylabel(ylabel = "Perplexity")

save_path = "/Users/osuhyeon/NLTK_court_document/(2,3) LDA Perplexities/"
plt.savefig(save_path + "Case%d.png" % int(i))
with open(save_path + "Case%d.txt" % int(i), 'w') as f:
    f.write(str(lda_train_dict_1_20))

print("All processes are completed! Check the folder : \n %s" % save_path)