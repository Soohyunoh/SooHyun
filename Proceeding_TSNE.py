from __future__ import print_function
import nltk
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE

# tsne_operator
def tsne_operator(n_components, lda_train_perplexity, lda_trained_data):
    tsne_model = TSNE(
        n_components = 3, 
        init = 'random', 
        perplexity = lda_train_perplexity,
        verbose = 2, 
        random_state = 0,
        n_iter = 500)
        
    tsne_trained_data = tsne_model.fit_transform(lda_trained_data)
    print("tsne_trained_data:", np.shape(tsne_trained_data))
    
    return tsne_trained_data


# tsne_scatterplot_3d
def tsne_scatterplot_3d(tsne_data, lda_data):
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    z = tsne_data[:, 2]
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_zlim(np.min(z),np.max(z))
    
    # lda_data : ROW=word, COLUMN=topic
    # c = np.array, column vector
    c = lda_data.argmax(axis=1)
    maxsize = [] 
    for word in lda_data:
        maxsize.append(max(word))
    size_matrix = np.array(maxsize) * int(150)

    img = ax.scatter(
        x,y,z, 
        c=c, cmap = plt.hot(), alpha=0.8, 
        s=size_matrix,
        edgecolors = '#000000',
        linewidth = 0.1
        )
    
    plt.xlabel('')
    plt.ylabel('')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    
    # Legend = colorbar
    divisionamount = max(c) - min(c) + int(1)
    division = np.linspace(start=min(c), stop=max(c), num=divisionamount, endpoint=True)
    fig.colorbar(
        img, 
        ax = ax, 
        ticks = division,
        shrink = 0.5, orientation = 'vertical')
    
    # Legend = color cell
    # legend = ax.legend(
    #     *img.legend_elements(),
    #     loc = "upper left",
    #     title = "Topics",
    #     frameon = False,
    #     markerscale = 1.5)
    # ax.add_artist(colorbar)

    # No meaning on tsne axis individually.
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(x, " ")
    plt.yticks(y, " ")
    # plt.legend(loc="right") 

    return plt



# import LDA_trained files.
saveLDAnumpy = "/Users/osuhyeon/NLTK_court_document/(Result) LDA Numpy/" 
saveLDAperplexity = "/Users/osuhyeon/NLTK_court_document/(Result) LDA Perplexities/"

# Start with an Input Range
print("Import LDA_trained files with Case index _ ")
print("Start Value _ (start from 1)")
inputstr1 = input()
print("End Value _ (end by 8)")
inputstr2 = input()
inputrange = [i for i in range(int(inputstr1), int(inputstr2))]

print("Start with a range_ ")
print(inputrange, type(inputrange))

# Start Loop
for i in inputrange:
    # import LDA_trained numpy.
    npyfilepath = saveLDAnumpy + "Case%d.npy" % int(i)
    lda_train = np.load(npyfilepath)

    # import LDA_trained perplexity according to the LDA_trained numpy above.
    perplexityfilepath = saveLDAperplexity + "Case%d.txt" % int(i)
    perplexity_str = open(perplexityfilepath, 'r').read()
    perplexity_re = re.sub("{|}", "", perplexity_str)
    perplexity_dict = dict(item.split(": ") for item in perplexity_re.split(", "))
    for key,value in perplexity_dict.items():
        if value == min(perplexity_dict.values()):
            n_components = int(key)
            min_perplexity = float(value)

    print("For Case%d _ " % int(i))

    print("type(npy_train_data): ", type(lda_train))
    print("np.shape(npy_train_data): ", np.shape(lda_train))

    print("number of topics: ", n_components)
    print("min_perplexity: ", min_perplexity)


    # tsne_operator
    tsne_trained_data = tsne_operator(
        n_components = 3, 
        lda_trained_data = lda_train,
        lda_train_perplexity = min_perplexity)

    # post-tsne_operator:
    # np.save(tsne_data)
    tsne_savedir = "/Users/osuhyeon/NLTK_court_document/(Result) TSNE/"
    tsne_savename = tsne_savedir + "Case%d.npy" % int(i)

    np.save(tsne_savename, tsne_trained_data)

    print("Is %s saved? " % tsne_savename, os.path.exists(tsne_savename))
    if os.path.exists(tsne_savename) == True:
        # pre-tsne_scattorplot_3d:
        # np.load(tsne_data)
        tsne_trained_data = np.load(tsne_savename)
        # tsne_scattorplot_3d
        plot3d = tsne_scatterplot_3d(
            tsne_data = tsne_trained_data, 
            lda_data = lda_train)
        # post-scattorplot_3d:
        tsne_plt_savename = tsne_savedir + "Case%d_3d.png" % int(i)
        plot3d.savefig(tsne_plt_savename)
        print("Is %s saved?" % tsne_plt_savename, os.path.exists(tsne_plt_savename))
    else:
        print("No tsne numpy saved!")


print("Complete all process!")
print("Check file directories:")
print(tsne_savedir)
