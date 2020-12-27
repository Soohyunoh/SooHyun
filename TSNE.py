class TSNE:
    # TSNE
    # n_component = number of dimensions to compress
    # Consider n_iter following Iteration 50 ~ 1000 results!

    def TSNE(self, n_components, lda_train_perplexity, lda_trained_data):
        from sklearn.manifold import TSNE
        import numpy as np

        tsne_model = TSNE(
            n_components = 3, 
            init = 'random', 
            perplexity = int(lda_train_perplexity),
            verbose = 2, 
            random_state = 0,
            n_iter = 280)
        
        tsne_y = tsne_model.fit_transform(lda_trained_data)
        print("tsne_y:", np.shape(tsne_y))
        return tsne_y