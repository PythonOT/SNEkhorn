# Example of SNEkhorn on COIL dataset

#%%
import matplotlib.pyplot as plt
from snekhorn.utils import COIL_dataset
import torch
from snekhorn.utils import PCA
from snekhorn import SNEkhorn
from snekhorn.dimension_reduction import SNE
from sklearn import metrics
#%%
X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
pca = PCA(n_components=50)
X_process = pca.fit_transform(X_coil)

# %%
perp = 30
tsnekhorn = SNEkhorn(perp=perp, 
                    student_kernel=True, 
                    lr=0.1, 
                    max_iter=200, 
                    square_parametrization=False,
                    lr_sea=1.0, 
                    max_iter_sea=1200,
                    tolog=True,
                    )
#%%
tsnekhorn.fit(X_process)
#%%
tsne = SNE(perp=perp, student_kernel=True, tolog=True, lr=0.1, max_iter=500)
tsne.fit(X_process)
#%%
from sklearn.manifold import TSNE
sklearn_sne = TSNE(perplexity=perp)
sklearn_sne.fit(X_process.numpy())
sklearn_embedding = torch.from_numpy(sklearn_sne.embedding_)
#%%
from snekhorn.utils import entropy
H_sne = entropy(tsne.PX_, log=False, ax=-1)
Perp_sne = torch.exp(H_sne - 1)

H_sym = entropy(tsnekhorn.PX_, log=False, ax=-1)
Perp_sym = torch.exp(H_sym - 1)
# %%
fig, axs = plt.subplots(1, 3, figsize=(10,4))
scatter_kwargs_coil = {'s': 5, 'alpha': 0.8, 'c': Y_coil, 'cmap': plt.get_cmap('tab20')}


params = {'text.usetex': True, 
          'text.latex.preamble': [r'\usepackage{cmbright}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

plt.rc('font', family='Times New Roman')
i=0
for emb, name in zip([tsne.embedding_, sklearn_embedding, tsnekhorn.embedding_],['tSNE', 'tSNE (scikit-learn)', 'tSNEkhorn']):

    axs[i].scatter(emb[:,0], emb[:,1], **scatter_kwargs_coil)
    axs[i].set_title(f'{0} (score:{1:.2f})'.format(name, float(metrics.silhouette_score(emb, Y_coil))), font='Times New Roman', fontsize=25)
    #axs[0,0].set_xticks([-5,5])
    i+=1
plt.show()
#%%



scatter_kwargs_coil_perm = scatter_kwargs_coil.copy()
scatter_kwargs_coil_perm['c'] = [i//(n//20)+1 for i in range(n)]
perm = torch.argsort(Y_coil)

axs[1,0].set_ylabel('Perplexities', fontsize=20)
axs[1,0].set_xlabel('Sample '+r'$i$', font='Times New Roman', fontsize=20)

axs[1,0].scatter(torch.arange(n), Perp_sne[perm], **scatter_kwargs_coil_perm)
axs[1,0].set_title(r'$\overline{\mathbf{P}^{\mathrm{e}}}$', fontsize=20)
axs[1,0].set_yticks([30,120,260])
axs[1,1].scatter(torch.arange(n), Perp_sym[perm], **scatter_kwargs_coil_perm)
axs[1,1].set_title(r'$\mathbf{P}^{\mathrm{se}}$', fontsize=20)
axs[1,1].set_ylim(29,31)

axs[1,1].set_xlabel('Sample '+r'$i$', font='Times New Roman', fontsize=20)

plt.show()
# %%
