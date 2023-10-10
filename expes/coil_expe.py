
# %%
# comparison with scikit-learn implementation
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from snekhorn.utils import COIL_dataset
import torch
from snekhorn.utils import PCA
from snekhorn.affinities import SymmetricEntropicAffinity, NanError
from snekhorn.dimension_reduction import DRWrapper
from sklearn import metrics
from sklearn.manifold import trustworthiness

torch.manual_seed(16)
# %% Load the COIL dataset
X_coil, Y_coil = COIL_dataset('../data/coil-20-proc')
perp = 30
n = X_coil.shape[0]
pca = PCA(n_components=50)
# we make a preprocessing with PCA in dim 50
X_process = pca.fit_transform(X_coil)
# %%
perps_test = [5, 10, 30, 50, 100, 200, 400, 500, 800, 1000]
all_affinities = {}
max_iter_aff = 4500
# precompute affinities, try two learning rates
for i, perp in enumerate(perps_test):
    try:
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=1.0, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_process)
    except NanError as e:
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=0.2, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_process)
    all_affinities[perp] = PX
    print('Perp = {} done'.format(perp))
# %% Save the affinities
with open('../data/saved/symmetric_entropic_affinities_COIL.pkl', 'wb') as f:
    pickle.dump(all_affinities, f)
# %%
lr = 1.0
max_iter = 1500
max_iter_sinkhorn = 5
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
silhouette_scores = {}
trustworthiness_scores = {}
for model, name in zip([simpletsnekhorn, tsnekhorn, None],
                       ['Simple tSNEkhorn', 'tSNEkhorn', 'tSNE (sklearn)']):
    for i, perp in enumerate(perps_test):
        if name == 'tSNE (sklearn)':
            tsne_sklearn = TSNE(perplexity=perp)
            tsne_sklearn.fit(X_process.numpy())
            emb = torch.from_numpy(tsne_sklearn.embedding_)
        else:
            PX_ = all_affinities[perp]
            model.fit(PX_)
            emb = model.embedding_
        # calculate scores and store them
        silhouette = float(metrics.silhouette_score(emb, Y_coil))
        trust_score = float(trustworthiness(X_process, emb))
        for score, dic_score in zip([silhouette, trust_score],
                                    [silhouette_scores, trustworthiness_scores]):
            if name not in dic_score.keys():
                dic_score[name] = [score]
            else:
                dic_score[name].append(score)

    print('!!!! Model {} done... !!!!'.format(name))

# %%
cmap = plt.cm.get_cmap('tab10')
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
j = 0
for name_score, dic_score in zip(['Silhouette', 'Trustworthiness'],
                                 [silhouette_scores, trustworthiness_scores]):
    for i, name in enumerate(['Simple tSNEkhorn', 'tSNEkhorn', 'tSNE (sklearn)']):
        ax[j].plot(perps_test, dic_score[name],
                   marker='o', c=cmap(i), lw=3, label=name)
    ax[j].legend(fontsize=fs-2)
    ax[j].grid()
    ax[j].tick_params(axis='both', which='major', labelsize=13)
    ax[j].set_xscale('log')
    ax[j].set_xlabel('Perplexity', fontsize=fs)
    ax[j].set_ylabel('{} score'.format(name_score), fontsize=fs)
    j += 1
plt.suptitle(
    'Silhouette/ Trustworthiness scores on COIL', fontsize=fs)
plt.tight_layout()
plt.savefig('../data/saved/coil_expe.pdf')
# %%
