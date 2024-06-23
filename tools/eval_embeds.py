import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pandas as pd

eps =1e-9
def compute_affinity_matrix(embeddings,kernel = None):
    l2_norms = torch.norm(embeddings, dim=1)
    embeddings_normalized = embeddings / l2_norms[:, None]
    cosine_similarities = embeddings_normalized@embeddings_normalized.T 
    affinity = 1- (cosine_similarities + 1.0) / 2.0
    if kernel:
        sigma = torch.sort(affinity**2,1)[0].max() 
        sigma *=0.0005
        affinity = torch.exp(-(affinity**2)/sigma)
    return affinity
metainfo = pd.read_csv('/dsi/gannot-lab1/LibriSpeech_mls_french/metainfo.csv')
male_names = set(metainfo['SPEAKER'][metainfo['GENDER'] == 'M'])
female_names = set(metainfo['SPEAKER'][metainfo['GENDER'] == 'F'])

# embds = [] #REGULAR EMBEDS
# with open('/home/workspace/yoavellinson/unsupervised_learning/SPICE/outputs/mel_embeds.pickle','rb') as f:
#     loaded_embeds = pickle.load(f)
# start = 0
# indecies = {}
# for l in loaded_embeds.keys():
#     e = torch.stack(loaded_embeds[l])
#     embds.append(e)
#     stop = start + e.shape[0]
#     indecies[l] = np.arange(start,stop,1)
#     start = stop

male_embds = []
female_embds = []
with open('/home/workspace/yoavellinson/unsupervised_learning/SPICE/outputs/mel_embeds.pickle','rb') as f:
    loaded_embeds = pickle.load(f)

for l in loaded_embeds.keys():
    e = torch.stack(loaded_embeds[l])
    if int(l) in male_names:
        male_embds.append(e)
    else:
        female_embds.append(e)

print(f'Males: {len(male_embds)}, Females: {len(female_embds)}')
male_embds = torch.cat(male_embds,dim=0)
female_embds = torch.cat(female_embds,dim=0)
male_idx = male_embds.shape[0]
embds = torch.cat((male_embds,female_embds),dim=0)
aff_mat = compute_affinity_matrix(embds,kernel=True)
shw = plt.imshow(aff_mat,vmin=0, vmax=1, cmap='turbo')
bat = plt.colorbar(shw)
plt.xticks([int(male_idx/2),male_idx,int(male_idx * 1.5)],['males','','females'])
plt.yticks([int(male_idx/2),male_idx,int(male_idx * 1.5)],['males','','females'])
plt.savefig('/home/workspace/yoavellinson/unsupervised_learning/SPICE/outputs/mel_embeds_kernel_gender_turbo.png')
