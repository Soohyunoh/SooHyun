import os
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk.collocations 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 


# Corpus
directory_path = '/Users/osuhyeon/NLTK_court_document/TXTDATA/'
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
        'AtelierLuxuryGroup AmendedComplaint 2020-08-24',
        'AtelierLuxuryGroup Complaint 2020-01-22'],
    "Case6_file_path" : [
        'Versace Complaint 2019-11-25'],
    "Case7_file_path" : [
        'LouisVuitton Judge Opinion 2006-06-30',
        'LouisVuitton_Opinion and Order 2007-12-14',
        'LouisVuitton_Opinion_and_Order 2008-04-24',
        'LouisVuitton_Opinion_and_Order 2008-05-30']}

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
parties_attorneys = "/Users/osuhyeon/NLTK_court_document/TXTDATA/[Praties_Attorneys].txt"
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

        # Access for individual sentence.
        # Becasuse lemmatizing is available for individual word.
        for sentence in tokenized:
            lemmatized_sentence = [WordNetLemmatizer().lemmatize(word, 'v') for word in sentence]
            filtered_sentence = [x for x in lemmatized_sentence if x not in word_filter and len(x) > 2]
            
            each_str_sentence = ' '.join(filtered_sentence)
            str_sentence = "".join(each_str_sentence)
            corpus.append(str_sentence)
            
        print("Making corpus by adding a file: ", file)
        # Be Cautious!
        # corpus = 1 item conntected to 1 sentece for 1 document 
    return corpus


# CountVectorizer processer
def Proceeding_TFcv(n_features, ngram_range, split_ratio, data, stopwords):
    ngram_tf = CountVectorizer(
        stop_words = stopwords, 
        ngram_range = ngram_range, 
        max_features = n_features) 

    train_data_amount = int(len(data) * split_ratio)
    ngram_tf_train = ngram_tf.fit_transform(data[:train_data_amount])
    # ngram_tf_test = ngram_tf.transform(data[int(len(data) * split_ratio):])
        
    print("ngram_tf_train_fit_transformed:", type(ngram_tf_train))
    print("ngram_tf_train_fit_transformed:", np.shape(ngram_tf_train))

    return ngram_tf, ngram_tf_train

# LDA processor
def Proceeding_LDA(n_component, ngram_tf_train):
    print("Fitting LDA models with tf features,")
    print(" n_components = %d" % n_component)

    lda = LatentDirichletAllocation(
        n_components = n_component, 
        learning_method = 'online', 
        random_state = 0,
        # doc_topic_prior = 1.0,
        # topic_word_prior = 1.0
        )
    lda.fit(ngram_tf_train)
    lda_train = lda.fit_transform(ngram_tf_train)
    lda_train_perplexity = lda.perplexity(ngram_tf_train)
    # To use ngram_tf_text:
    # lda_test = lda.fit(ngram_tf_test)

    print("lda_train:", type(lda_train), np.shape(ngram_tf_train))
    print("lda_train_perplexity:", lda_train_perplexity)
    
    return lda, lda_train, lda_train_perplexity

# LDA Numpy Result to String Convertor
def lda_print(CountVectorizer, lda, n_features):
    # sklearn.decomposition.LatentDirichletAllocation.components_ : array, [n_topics, n_features]
    # components_[i, j] represents word j in topic i.
    # lda.components_ : after fitted by train data
    # lda.components_ = {topic index : [word1, word2, ...word j...]}
    # word j = [topic1 weight, topic2 weight, ..., topic i weight]
    # words.argsort()[:-(n_features+1):-1] : calling all words in the topic i

    # Premised that "CountVectorizer" is fitted by the train data
    each_ngram_word = CountVectorizer.get_feature_names()
    topic_words_dict = {}

    # Premised that "lda" is fitted by the CountVectorized train data
    for topic_idx, words in enumerate(lda.components_):
        topic = [each_ngram_word[item] for item in words.argsort()[:-(n_features+1):-1]]
        topic_words_dict[topic_idx] = topic
        # len(topic) indicates size of the topic (how many words are included in the topic) 

    return topic_words_dict


# Start with an Input Range
print("Input value referring which case to proceed _")
print("Start Value _ (start from 1)")
inputstr1 = input()
print("End Value _ (end by 8)")
inputstr2 = input()
inputrange = [i for i in range(int(inputstr1), int(inputstr2))]

print("Start with a range_ ")
print(inputrange, type(inputrange))

# Start Loop
for i in inputrange:
    # Ingredients for Corpus
    print("Please wait. Start to access to Case%d _ " % i)
    case_list = "Case%d_file_path" % int(i)
    a_Case_file_path_list = Case_file_dict[case_list]

    # Corpus
    corpus = []
    Case_corpus = Corpus(
        Case_file_path= a_Case_file_path_list, 
        corpus = corpus, 
        word_filter = exclude)

    # Ingredients for LDA
    # Perplexity
    saved_path = "/Users/osuhyeon/NLTK_court_document/(Result) LDA Perplexities/"
    perplexity_str = open(saved_path + "Case%d.txt" % int(i), 'r').read()
    perplexity_re = re.sub("{|}", "", perplexity_str)
    perplexity_dict = dict(item.split(":") for item in perplexity_re.split(","))

    # Topic numbers according to the Minimum Perplexity
    for key,value in perplexity_dict.items():
        if value == min(perplexity_dict.values()):
            n_component = int(key)

    # Word numbers in a topic according to the Minimum Perplexity
    ngram_range = (1,3)
    n_features = 100


    # CountVectorize
    stopwords = nltk.corpus.stopwords.words('english')
    ngram_tf, ngram_tf_train = Proceeding_TFcv(
        n_features= n_features, 
        ngram_range = ngram_range,
        split_ratio = 1.0, 
        data = Case_corpus, 
        stopwords = stopwords)

    # Latent Dirichlet Allocation
    lda, lda_train, lda_train_perplexity = Proceeding_LDA(
        n_component = n_component, 
        ngram_tf_train = ngram_tf_train)

    # save numpy Latent Dirichlet Allocation result
    saveLDAnumpy = "/Users/osuhyeon/NLTK_court_document/(Result) LDA Numpy/" 
    np.save(saveLDAnumpy + "Case%d" % int(i), lda_train)
    # To open numpy file, Use code below:
    # numpyfile = np.load(filepath + ".npy")

    # LDA Result to text dictionary
    # No more need of lda_trained_data 
    topic_words_dict = lda_print(
        CountVectorizer = ngram_tf, 
        lda = lda, 
        n_features = n_features)

    # save text dictionary
    saveTextDictionary = "/Users/osuhyeon/NLTK_court_document/(Result) LDA Text Dictionary/"
    saveTextDictionaryName = saveTextDictionary + "Case%d" % int(i)
    with open(saveTextDictionaryName, 'w', encoding="UTF8") as f:
        f.write(str(topic_words_dict))

    # save pprinted text dictionary
    target_list = open(saveTextDictionaryName, 'r').read()
    target_str = re.sub("{|}", "", target_list)
    target_dict = dict(item.split(": [")for item in target_str.split("], "))

    pprint_file = open(saveTextDictionaryName + "_pprint", 'w', encoding="UTF8")
    LDA_train_data_summary = "LDA train data:" + str(type(lda_train)) + str(np.shape(ngram_tf_train)) + "\nLDA train perplexity:" + str(lda_train_perplexity)
    pprint_file.write(LDA_train_data_summary)

    for topic,words in target_dict.items():
        # type(words) : str
        # type(words_in_a_topic) : list
        words_in_a_topic = words.split("', '")
        # topic_size per topic = n_features 
        topic_size = len(words_in_a_topic)
        topic_summary = topic + ", size(%d)" % topic_size
        pprint_file.write("\n %s" % topic_summary)
        pprint_file.write("\n [%s] \n" % words)

print("All processes are completed! Check the folder : \n %s" % saveLDAnumpy + "\n" + saveTextDictionary)





# Be CAUTIOUS! 
# ann = ANN. = Announcement

# Adidas 
"""
        linew2 = [re.sub('plaintiff', 'adidas', li2) for li2 in linew]
        linew3 = [re.sub('defendant', "SKECHERS", li3) for li3 in linew2]
"""

# Jenny Yoo
"""
        linew2 = [re.sub('plaintiff|jy', 'JennyYoo', li2) for li2 in linew]
        linew3 = [re.sub('defendant|db', "David'sBridal", li3) for li3 in linew2]
"""

# Christian Louboutin
"""
        linew2 = [re.sub('plaintiff', 'LouisVuitton', li2) for li2 in linew]
        linew3 = [re.sub('defendant|ysl', "YvesSaintLaurent", li3) for li3 in linew2]
"""

# PUMA
"""
        linew2 = [re.sub('plaintiff', 'PUMA', li2) for li2 in linew]
        linew3 = [re.sub('defendant|', "FOREVER21", li3) for li3 in linew2]
"""

# Atherlier Luxury Group
"""
        linew2 = [re.sub('plaintiff|atelier luxury group', 'Amiri', li2) for li2 in linew]
        linew3 = [re.sub('defendant|zara usa', "ZARA", li3) for li3 in linew2]
"""

# Versace
"""
        linew2 = [re.sub('plaintiff', 'Versace', li2) for li2 in linew]
        linew3 = [re.sub('defendant', "FashionNova", li3) for li3 in linew2]
"""

# Louis Vuitton
"""
        linew2 = [re.sub('plaintiff|Appellant|louis vuitton malletier', 'LouisVuitton', li2) for li2 in linew]
        linew3 = [re.sub('defendant', "Dooney&Bourke", li3) for li3 in linew2]
"""
