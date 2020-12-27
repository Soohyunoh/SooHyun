class tsne_vis_3d:
    def pandas_tsne_scatterplot(self, tsne_data, lda_data):
        
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = tsne_data[:, 0]
        y = tsne_data[:, 1]
        z = tsne_data[:, 2]
        c = lda_data.argmax(axis=1)

        img = ax.scatter(x,y,z, c=c, cmap = plt.hot())
        fig.colorbar(img)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.zlabel('z')

        plt.legend(loc="upper right", title="Topics")
        plt.title('TSNE plot of Latent Dirichlet Allocation')
        plt.show()