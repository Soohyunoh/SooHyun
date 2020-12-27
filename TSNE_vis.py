class tsne_vis:
    def pandas_tsne_scatterplot(self, tsne_data, lda_data):

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # pandas DataFrame by "color"
        # pandas DataFrame(only 2D array available!)
        # 2D axis=0 : among each column vectors
        # 2D axis=1 : among each row vectors

        # a = np.arrange(6).reshape(2,3) + 10
        # a
        # >> array([
        #   [10, 11, 12],
        #   [13, 14, 15]
        # ])
        # np.argmax(a, axis=0)
        # >> array([1, 1, 1])
        # np.argmax(a, axis=1)
        # >> array([2, 2])

        df = pd.DataFrame(tsne_data)
        df['topic'] = lda_data.argmax(axis=1)
        df.columns = ['TSNE1', 'TSNE2', 'topic'] 

        # Draw 3d scatter plot
        # topic_idx : topic number
        # words[0] == topic_idx
        # words[1] == [TSNE1, TSNE2] 
        # color_list = [plt.cm.get_cmap('afmhot')[a] for a in range(len(set(df['topic'])))]
        color_list = [
            "#ff0000", # 0 red
            "#ffbf00", # 1 yellow
            "#fa8072", # 2 salmon
            "#ff00ff", # 3 magenta
            "#ccccff", # 4 light purple
            "#800080", # 5 purple 
            "#00ffff", # 6 aqua
            "#0000ff", # 7 blue
            "#dfff00", # 8 lime
            "#00ff00", # 9 lime green
            "#000000", # 10 black
            ]

        for topic_idx, words in enumerate(df.groupby('topic')):
            plt.scatter(
                words[1]['TSNE1'], 
                words[1]['TSNE2'], 
                alpha=0.8, 
                s= 2,
                color=color_list[topic_idx], 
                label='topic {}'.format(int(words[0]))
            )

        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.legend(loc="upper right", title="Topics")

        plt.title('TSNE plot of Latent Dirichlet Allocation')
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)

        return plt.show()

    
    def pandas_tsne_scatterplot_closer(self, tsne_data, lda_data, x_closer_lim, y_closer_lim):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(tsne_data)
        df['topic'] = lda_data.argmax(axis=1)
        df.columns = ['TSNE1', 'TSNE2', 'topic'] 

        color_list = [
            "#ff0000", # 0 red
            "#ffbf00", # 1 yellow
            "#fa8072", # 2 salmon
            "#ff00ff", # 3 magenta
            "#ccccff", # 4 light purple
            "#800080", # 5 purple 
            "#00ffff", # 6 aqua
            "#0000ff", # 7 blue
            "#dfff00", # 8 lime
            "#00ff00", # 9 lime green
            "#000000", # 10 black
            ]

        for topic_idx, words in enumerate(df.groupby('topic')):
            plt.scatter(
                words[1]['TSNE1'], 
                words[1]['TSNE2'], 
                alpha=0.5, 
                s= 0.1,
                color=color_list[topic_idx], 
                label='topic {}'.format(int(words[0]))
            )

        plt.xlabel('TSNE1')
        plt.ylabel('TSNE2')
        plt.legend(loc="upper right", title="Topics")

        plt.title('Closer Look of TSNE plot of Latent Dirichlet Allocation')
        plt.xlim(-x_closer_lim, x_closer_lim)
        plt.ylim(-y_closer_lim, y_closer_lim)

        return plt.show()



class tsne_vis_3d:
    def Axes3D_tsne_scatterplot(self, tsne_data, lda_data, save_name):
        
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        x = tsne_data[:, 0]
        y = tsne_data[:, 1]
        z = tsne_data[:, 2]
        c = lda_data.argmax(axis=1)

        list_name_variables = ['x', 'y', 'z', 'c']

        img = ax.scatter(x,y,z, s=10, c=c, cmap = plt.hot(), alpha=0.7, linewidths=1)
        cbar = fig.colorbar(img, shrink=0.5, aspect=15)
        cbar.ax.get_yaxis().labelpad = 15 
        cbar.ax.set_ylabel("Topics", rotation = 270)

        ax.set_xlabel(list_name_variables[0])
        ax.set_ylabel(list_name_variables[1])
        ax.set_zlabel(list_name_variables[2])

        #plt.legend(loc="upper right")
        plt.title('TSNE plot of Latent Dirichlet Allocation')

        return plt.savefig(save_name)