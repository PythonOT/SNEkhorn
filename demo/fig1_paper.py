import matplotlib.pyplot as plt
import torch
from snekhorn.utils import COIL_dataset
from snekhorn.utils import PCA
from snekhorn import SNEkhorn
from snekhorn.dimension_reduction import SNE
from snekhorn.utils import entropy

X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
pca = PCA(n_components=50)
# we make a preprocessing with PCA in dim 50
X_process = pca.fit_transform(X_coil)
scatter_kwargs_coil = {'s': 5, 'alpha': 0.8, 'c': Y_coil, 'cmap': plt.get_cmap('tab20')}

# SNEkhorn
perp = 30
tsnekhorn = SNEkhorn(perp=perp,
                     student_kernel=True,  # use the tSNEkhorn version
                     init='pca',  # initialize the embedding with PCA
                     lr=0.5,  # learning rate for minimizing the KL divergence
                     max_iter=800,  # maximum number of iteration of the descent algorithm
                     # when facing unstability and Nan square_parametrization=True helps
                     square_parametrization=False,
                     lr_sea=1e-1,# learning rate for computing the symmetric entropic affinity
                     max_iter_sea=20000,  # maximum iterations for computing the symmetric entropic affinity
                     max_iter_sinkhorn=10,  # number of Sinkhorn iterations
                     tolog=True,
                     optimizer='Adam'
                     )
# %% fitting SNEkhorn
tsnekhorn.fit(X_process)

# t-SNE
tsne = SNE(perp=perp, student_kernel=True, tolog=True,
           lr=0.5, max_iter=800, init='pca')
tsne.fit(X_process)

embed_sne = tsne.embedding_
embed_sym = tsnekhorn.embedding_

H_sne = entropy(tsne.affinity_data.data, log=False, ax=-1)
Perp_sne = torch.exp(H_sne - 1)

H_sym = entropy(tsnekhorn.affinity_data.data, log=False, ax=-1)
Perp_sym = torch.exp(H_sym - 1)

fig, axs = plt.subplots(2, 2, figsize=(10,6), gridspec_kw = {'height_ratios':[3,1]})

params = {'text.usetex': True}
        #   'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)
plt.rc('font', family='Times New Roman')

axs[0,0].scatter(embed_sne[:,0], embed_sne[:,1], **scatter_kwargs_coil)
axs[0,0].set_title(f't-SNE', font='Times New Roman', fontsize=25)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[0,1].scatter(embed_sym[:,0], embed_sym[:,1], **scatter_kwargs_coil)
axs[0,1].set_title(f't-SNEkhorn', font='Times New Roman', fontsize=25)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])

scatter_kwargs_coil_perm = scatter_kwargs_coil.copy()
scatter_kwargs_coil_perm['c'] = [i//(n//20)+1 for i in range(n)]
perm = torch.argsort(Y_coil)
axs[1,0].set_ylabel(r'$e^{\mathrm{H}(\mathbf{P}_{i:})-1}$', fontsize=20)

axs[1,0].set_xlabel('Sample '+r'$i$', font='Times New Roman', fontsize=20)
axs[1,0].scatter(torch.arange(n), Perp_sne[perm], **scatter_kwargs_coil_perm)
axs[1,0].set_title(r'$\overline{\mathbf{P}^{\mathrm{e}}}$', fontsize=20)
axs[1,0].set_yticks([30,120,260])

axs[1,1].set_xlabel('Sample '+r'$i$', font='Times New Roman', fontsize=20)
axs[1,1].scatter(torch.arange(n), Perp_sym[perm], **scatter_kwargs_coil_perm)
axs[1,1].set_title(r'$\mathbf{P}^{\mathrm{se}}$', fontsize=20)
axs[1,1].set_ylim(28,32)

plt.show()