# Example of tSNEkhorn on COIL dataset

# %%
from sklearn.manifold import TSNE  # comparison with scikit-learn implementation
import matplotlib.pyplot as plt
from snekhorn.utils import COIL_dataset
import torch
from snekhorn.utils import PCA
from snekhorn import SNEkhorn
from snekhorn.dimension_reduction import SNE
from sklearn import metrics
# %% Load the COIL dataset
X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
pca = PCA(n_components=50)
# we make a preprocessing with PCA in dim 50
X_process = pca.fit_transform(X_coil)

# %%
perp = 30
tsnekhorn = SNEkhorn(perp=perp,
                     student_kernel=True,  # use the tSNEkhorn version
                     init='pca',  # initialize the embedding with PCA
                     lr=0.5,  # learning rate for minimizing the KL divergence
                     max_iter=800,  # maximum number of iteration of the descent algorithm
                     # when facing unstability and Nan square_parametrization=True helps
                     square_parametrization=False,
                     lr_sea=1.0,  # learning rate for computing the symmetric entropic affinity
                     max_iter_sea=1800,  # maximum iterations for computing the symmetric entropic affinity
                     max_iter_sinkhorn=10,  # number of Sinkhorn iterations
                     tolog=True
                     )
# %% fitting SNEkhorn
tsnekhorn.fit(X_process)
# %% Comparing with our implementation of tSNE/SNE
tsne = SNE(perp=perp, student_kernel=True, tolog=True,
           lr=0.5, max_iter=800, init='pca')
tsne.fit(X_process)
# %% Comparing with scikit-learn implementation
sklearn_sne = TSNE(perplexity=perp)
sklearn_sne.fit(X_process.numpy())
sklearn_embedding = torch.from_numpy(sklearn_sne.embedding_)

# %% Plot the corresponding embeddings and the silhouette scores
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
scatter_kwargs_coil = {'s': 5, 'alpha': 0.8,
                       'c': Y_coil, 'cmap': plt.get_cmap('tab20')}


params = {'text.usetex': True}
        #   'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

plt.rc('font', family='Times New Roman')
i = 0
for emb, name in zip([tsne.embedding_, sklearn_embedding, tsnekhorn.embedding_], ['tSNE (vanilla)', 'tSNE (scikit-learn)', 'tSNEkhorn']):
    score = float(metrics.silhouette_score(emb, Y_coil))
    axs[i].scatter(emb[:, 0], emb[:, 1], **scatter_kwargs_coil)
    axs[i].set_title('{0} \n (score = {1:.2f})'.format(
        name, score), font='Times New Roman', fontsize=30)
    i += 1
plt.tight_layout()
plt.show()
