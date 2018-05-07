from adversarial_autoencoder import *
from skimage.io import imread
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pickle
import cv2 

aae = AdversarialAutoencoder()
aae.adversarial_autoencoder.load_weights("adversarial_ae.h5")

encodings = []
counter = {"forged_patches": 0, "pristine_patches": 0}
for fol in ['forged_patches', 'pristine_patches']:
    test_imgs = os.listdir(os.path.join('data', fol))
    stage = len(test_imgs) // 100
    cur_stage = 0
    for i, idx in enumerate(test_imgs):
        if (i+1) % 100 == 0:
            cur_stage += 1

        print("[" + "="*cur_stage + " "*(stage - cur_stage) + "]",
            end='\r', flush=True)
        im = cv2.imread(os.path.join('data', fol, idx))
        if im.shape != (64, 64, 3):
            continue
        im = im.astype(np.float32)/255.
        encoding = aae.adversarial_autoencoder.layers[1].predict(
            np.expand_dims(im, 0))
        encoding = encoding.reshape(2048, 1)
        encodings.append(encoding)
        counter[fol] += 1

encodings = np.squeeze(np.asarray(encodings))
np.save("encodings", encodings)
print(
    "{} forged patches and {} pristine patches".format(counter['forged_patches'], counter['pristine_patches'])
)
clf = SVC(gamma=1/2048, probability=True)
# pca = PCA(n_components=3)
# pca.fit(encodings)
# X_3d = pca.transform(encodings)


y = np.ones((encodings.shape[0], 1))
y[:counter['forged_patches']] = 0
y[counter['pristine_patches']:] = 1

clf.fit(encodings, y)
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(encodings, y)
print(result)
