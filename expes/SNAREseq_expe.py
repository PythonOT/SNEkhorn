# %%
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
from snekhorn.dimension_reduction import SNE as OurSNE
import numpy as np
torch.manual_seed(16)

X_snare = np.genfromtxt("../data/SCOT/SNAREseq_1.txt", delimiter="\t")
Y_snare = np.genfromtxt("../data/SCOT/SNAREseq_label.txt", delimiter="\t")

X_snare = torch.from_numpy(X_snare)
X_snare = (X_snare-X_snare.mean()) / X_snare.std()
perps_test = [5, 10, 30, 50, 100, 200, 400, 500, 800, 1000]

#%% if precomputed
# file = open('../data/saved/symmetric_entropic_affinities_SNAREseq.pkl', 'rb')
# all_affinities = pickle.load(file)

# %%
all_affinities = {}
max_iter_aff = 5000
# precompute affinities, try two learning rates
for i, perp in enumerate(perps_test):
    try:
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=1e-1, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_snare)
    except NanError as e:
        print('NanError we try a different lr')
        symmetric_entropic_affinity = SymmetricEntropicAffinity(
            perp=perp, lr=3e-2, max_iter=max_iter_aff, square_parametrization=False)
        PX = symmetric_entropic_affinity.compute_affinity(X_snare)
    all_affinities[perp] = PX
    print('Perp = {} done'.format(perp))
# %% Save the affinities
# with open('../data/saved/symmetric_entropic_affinities_SNAREseq.pkl', 'wb') as f:
#     pickle.dump(all_affinities, f)
# %%
lr = 1e-1 # learning rate for all methods, tune it for better results
max_iter = 3000
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
for model, name in zip([simpletsnekhorn, tsnekhorn, None, None],
                       ['Simple tSNEkhorn', 'tSNEkhorn', 'tSNE (sklearn)', 'tSNE (vanilla)']):
    for i, perp in enumerate(perps_test):
        if name == 'tSNE (sklearn)':
            tsne_sklearn = TSNE(perplexity=perp, learning_rate=lr) #we choose the same learning rate for fair comparison
            tsne_sklearn.fit(X_snare.numpy())
            emb = torch.from_numpy(tsne_sklearn.embedding_)
        elif name == 'tSNE (vanilla)': #we also compare with our vanilla implementation of tSNE
            our_tsne = OurSNE(perp=perp, lr=lr, student_kernel=True,
                              max_iter_ea=1000, max_iter=max_iter, verbose=True)
            our_tsne.fit(X_snare)
            emb = our_tsne.embedding_
        else:
            PX_ = all_affinities[perp]
            model.fit(PX_)
            emb = model.embedding_
        # calculate scores and store them
        silhouette = float(metrics.silhouette_score(emb, Y_snare))
        trust_score = float(trustworthiness(X_snare, emb))
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
    for i, name in enumerate(['Simple tSNEkhorn', 'tSNEkhorn', 'tSNE (sklearn)', 'tSNE (vanilla)']):
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
    'Silhouette/ Trustworthiness scores on SNAREseq', fontsize=fs)
plt.tight_layout()
plt.savefig('../data/saved/SNAREseq_expe.pdf')
# %%
