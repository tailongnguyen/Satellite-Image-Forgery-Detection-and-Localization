from adversarial_autoencoder import *
import cv2
import os
import matplotlib.pyplot as plt

# aae = AdversarialAutoencoder()
# aae.adversarial_autoencoder.load_weights("adversarial_ae.h5")

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            left_x = x if x + \
                windowSize[1] < image.shape[1] else image.shape[1] - windowSize[1]
            right_x = x + windowSize[1] if x + \
                windowSize[1] < image.shape[1] else image.shape[1]
            left_y = y if y + \
                windowSize[0] < image.shape[0] else image.shape[0] - windowSize[0]
            right_y = y + windowSize[0] if y + \
                windowSize[0] < image.shape[0] else image.shape[0]
            yield (left_x, left_y, image[left_y: right_y, left_x: right_x])


imgs = os.listdir("test_data")
im = cv2.imread("test_data/" + np.random.choice(imgs))
mask = np.zeros((640, 640)).astype(np.float32)

for (x, y, patch) in sliding_window(im, 20, (64, 64)):
    # patch = patch.astype(np.float32)/255.0
    # encoding = aae.adversarial_autoencoder.layers[1].predict(
    #     np.expand_dims(patch, 0))
    # encoding.resize((1, 2048))
    # en_3d = pca.transform(encoding)
    # probs = svm_clf.predict_proba(en_3d)
    probs = np.random.uniform(0, 1)
    mask[y:y+64, x:x+64] = np.maximum(mask[y:y+64, x:x+64],
                                      np.ones((64, 64), dtype=np.float32) * probs)

plt.figure()
plt.subplot(121)
plt.title("Input")
plt.imshow(im)
plt.subplot(122)
plt.imshow(mask, cmap='gray')
plt.savefig("softmask")
# plt.show()
