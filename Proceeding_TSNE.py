from __future__ import print_function
import nltk
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE


# import LDA_trained files.
saveLDAnumpy = "/Users/osuhyeon/NLTK_court_document/(2,3) LDA Numpy/" 
saveLDAperplexity = "/Users/osuhyeon/NLTK_court_document/(2,3) LDA Perplexities/"
print("Import LDA_trained files with Case index _ ")
i = input()

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

print("type(npy_train_data): ", type(lda_train))
print("np.shape(npy_train_data): ", np.shape(lda_train))
print("number of topics: ", n_components)
print("min_perplexity: ", min_perplexity )


# tsne_operator
def tsne_operator(n_components, lda_train_perplexity, lda_trained_data):
    tsne_model = TSNE(
        n_components = 3, 
        init = 'random', 
        perplexity = lda_train_perplexity,
        verbose = 2, 
        random_state = 0,
        n_iter = 280)
        
    tsne_trained_data = tsne_model.fit_transform(lda_trained_data)
    print("tsne_trained_data:", np.shape(tsne_trained_data))
    
    return tsne_trained_data


# tsne_scatterplot_3d
def tsne_scatterplot_3d(tsne_data, lda_data, case_index):
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    z = tsne_data[:, 2]
    # lda_data : ROW=word, COLUMN=topic
    # c = np.array, column vector
    ax.set_xlim((min(x),max(x)))
    ax.set_ylim((min(y),max(y)))
    ax.set_zlim((min(z),min(z)))

    c = lda_data.argmax(axis=1)
    listsize = [] 
    for word in lda_data:
        max_val = max(word)
        listsize.append(max_val)
    size_matrix = np.array(listsize) * int(150)

    img = ax.scatter(
        x,y,z, 
        c=c, cmap = plt.hot(), alpha=0.8, 
        s=size_matrix,
        edgecolors = '#000000'
        )
    
    plt.xlabel('')
    plt.ylabel('')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    
    # Legend = colorbar
    amount = max(c) - min(c) + int(1)
    division = np.linspace(start = min(c), stop = max(c), num = amount, endpoint=True)
    fig.colorbar(
        img, 
        ax=ax, 
        ticks = division,
        shrink = 0.5, orientation = 'vertical')
    # colorbar.set_label('Case%d Topics' % int(case_index), fontsize = 9)
    # plt.colorbar()
    
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



# tsne_operator
tsne_trained_data = tsne_operator(
    n_components = 3, 
    lda_trained_data = lda_train,
    lda_train_perplexity = min_perplexity)

tsne_savedir = "/Users/osuhyeon/NLTK_court_document/(2,3) TSNE/"
tsne_savename = tsne_savedir + "Case%d.npy" % int(i)
np.save(tsne_savename, tsne_trained_data)
print("Is %s saved? /n" % tsne_savename)
print(os.path.exists(tsne_savename))



tsne_trained_data = np.load(tsne_savename)

# tsne_scattorplot_3d
tsne_plt_savename = tsne_savedir + "Case%d_3d.png" % int(i)

plot3d = tsne_scatterplot_3d(
    tsne_data = tsne_trained_data, 
    lda_data = lda_train,
    case_index = i)

plot3d.savefig(tsne_plt_savename)
print("Is %s saved?" % tsne_plt_savename)
print(os.path.exists(tsne_plt_savename))


print("Complete all process!")
print("Check file directories:")
print(tsne_savedir)

plot3d.show(plot3d)