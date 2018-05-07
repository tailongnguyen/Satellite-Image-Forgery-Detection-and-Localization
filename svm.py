import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.decomposition import PCA

clf = SVC(gamma=1/2048, probability= True)
X = np.load('embeddings.npy')
pca = PCA(n_components=3)
pca.fit(X)
X_3d = pca.transform(X)

print(X.shape)
y = np.ones((X.shape[0], 1))
y[-284:] = 0 
clf.fit(X_3d, y)
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_3d, y)
print(result)
