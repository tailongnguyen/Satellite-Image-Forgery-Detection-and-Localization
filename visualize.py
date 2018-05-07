from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

# p_ims = os.listdir("pristine_patches/")
# p_ims = np.asarray(p_ims)
print("Loading data ...")
p_3d = np.load("pristines.npy")
f_3d = np.load("forged.npy")
print("Decomposing ...")

# pristines = np.load('pristines.npy')
# forged = np.load('forged.npy')
# pca = PCA(n_components=3)
# pca.fit(pristines)
# p_3d = pca.transform(pristines)
# f_3d = pca.transform(forged)

# encodings = np.concatenate((p_3d, f_3d), axis = 0)
# encodings = np.load('encodings.npy')
# mean_p = np.mean(p_3d, axis = 0)
# dist = np.sqrt(np.sum((f_3d - mean_p)**2, axis= 1))
# dist_p = np.sqrt(np.sum((p_3d - mean_p)**2, axis=1))
# p_ims = pristines[dist_p < np.min(dist, 0) + 10]
# print(p_ims.shape)
# np.save("embeddings", np.concatenate((p_ims, forged), 0))

# for im in p_ims_rm:
#     os.remove("pristine_patches/{}".format(im))

# fp_3d = p_3d[dist_p < np.min(dist, 0) + 10]
# print(fp_3d.shape)
out = open("logs/labels.tsv", "w")
out.write('Name\tFrequency\n')

for i in range(p_3d.shape[0]):
    out.write('pristine\t{}\n'.format(p_3d.shape[0]))
    
for i in range(f_3d.shape[0]):
    out.write('forged\t{}\n'.format(f_3d.shape[0]))
print(np.concatenate((p_3d, f_3d), 0).shape)
np.save("embeddings", np.concatenate((p_3d, f_3d), 0))

# np.save('encodings', encodings)

# mean_p = np.mean(p_3d, axis= 0)

# dist = np.sqrt(np.sum((f_3d - mean_p)**2, axis= 1))
# max_dist = np.max(dist)

# indices = f_ims[dist > max_dist * 0.2]
# for idx in indices:
#     os.rename("forged_patches/" + idx, "candidates/" +idx)
# candidates = f_3d[dist > max_dist * 0.2]
# print("{} candidates".format(candidates.shape[0]))

# print("Plotting ...")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(p_3d[:, 0], p_3d[:, 1], p_3d[:, 2], c = 'C0')
# ax.scatter(f_3d[:, 0], f_3d[:, 1], f_3d[:, 2], c = 'C1')
# # ax.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], c = 'C1')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
