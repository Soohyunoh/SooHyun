# Choosing n_components by perplextiy

class Choosing_n_components():

    def Choosing_n_components(self, n_features, var_n_components, ngram_range, train_data, stop_words):
        import nltk
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation 

        # split_ratio = int(len(train_data)*0.7)
        split_ratio = int(len(train_data)*1)
        perplexities = []

        for i in var_n_components:
            print("\n Start LDA iteration with var_n_components")
            n_components = i         
            ngram_tf = CountVectorizer(
                stop_words = stop_words, 
                ngram_range= ngram_range, 
                max_features = n_features) 
            ngram_tf_train = ngram_tf.fit_transform(train_data[:split_ratio])
            # ngram_tf_test = ngram_tf.transform(train_data[split_ratio:])
        
            print("ngram_tf_train_fit_transformed:", type(ngram_tf_train), "np.shape:", np.shape(ngram_tf_train))
            print("Fitting LDA models with tf features,", 
                    " n_components = %d, n_features = %d" % (n_components, n_features))

            lda = LatentDirichletAllocation(
                n_components = n_components, 
                learning_method = 'online', 
                random_state = 0)

            lda.fit(ngram_tf_train)
            lda_train = lda.fit_transform(ngram_tf_train)
            print("lda_train_data:", np.shape(lda_train))
            # lda_test = lda.transform(ngram_tf_test)
            # print("lda_test:", type(lda_test), "np.shape:", np.shape(ngram_tf_test))
            
            lda_train_perplexity = lda.perplexity(ngram_tf_train)
            perplexities.append(lda_train_perplexity)
            print("lda_train_perplexity:", lda_train_perplexity)

        return perplexities
