
# %%
from sklearn.manifold import TSNE  # comparison with scikit-learn implementation
import matplotlib.pyplot as plt
from snekhorn.utils import COIL_dataset
import torch
from snekhorn.utils import PCA
from snekhorn.affinities import SymmetricEntropicAffinity, NanError
from snekhorn.dimension_reduction import DRWrapper
from sklearn import metrics

torch.manual_seed(16)
# %% Load the COIL dataset
X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
pca = PCA(n_components=50)
# we make a preprocessing with PCA in dim 50
X_process = pca.fit_transform(X_coil)
# %%
perps_test = [10, 30, 50, 100, 200, 400, 500, 800, 1000]
all_affinities = {}
max_iter_aff = 3000
# precompute affinities, try two learning rates
for i, perp in enumerate(perps_test):
    try:
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=1.0, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_process)
    except NanError as e:
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=0.5, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_process)
    all_affinities[perp] = PX
    print('Perp = {} done'.format(perp))
# %% Save the affinities
# import pickle
# with open('../data/saved/symmetric_entropic_affinities_COIL.pkl', 'wb') as f:
#     pickle.dump(all_affinities, f)
# %%
lr = 1.0
max_iter = 1500
max_iter_sinkhorn = 20
tsnekhorn = DRWrapper(affinity_data='precomputed',
                      affinity_embedding='student_bistochastic',
                      params_affinity_embedding={
                          'max_iter': max_iter_sinkhorn},
                      lr=lr,
                      max_iter=max_iter)

# we also look at a simpler tSNEkhorn version where the embedding affinity
# is a student kernel normalized by its total sum
simpletsnekhorn = DRWrapper(affinity_data='precomputed',
                            affinity_embedding='student',
                            lr=lr,
                            max_iter=max_iter)
res = {}
for model, name in zip([None], ['tSNEkhorn', 'Simple tSNEkhorn', 'tSNE (sklearn)']):
    for i, perp in enumerate(perps_test):
        if name == 'tSNE (sklearn)':
            tsne_sklearn = TSNE(perplexity=perp)
            tsne_sklearn.fit(X_process.numpy())
            emb = torch.from_numpy(tsne_sklearn.embedding_)
        else:
            PX_ = all_affinities[perp]
            model.fit(PX_)
            emb = model.embedding_
        score = float(metrics.silhouette_score(emb, Y_coil))
        if name not in res.keys():
            res[name] = [score]
        else:
            res[name].append(score)
    print('!!!! Model {} done... !!!!'.format(name))

# %%
cmap = plt.cm.get_cmap('tab10')
fs = 15
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i, name in enumerate(['Simple tSNEkhorn', 'tSNEkhorn', 'tSNE (sklearn)']):
    ax.plot(perps_test, res[name], marker='o', c=cmap(i), lw=2, label=name)
ax.legend(fontsize=fs-2)
ax.grid()
ax.set_xscale('log')
ax.set_xlabel('Perplexity', fontsize=fs)
ax.set_ylabel('Silhouette score', fontsize=fs)
ax.set_title('Silhouette scores on COIL dataset', fontsize=fs+2)
# %%
