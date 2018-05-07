import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import glob


LOG_DIR = 'logs/'
encodings = np.load("embeddings.npy")
print(encodings.shape)

embedding_var = tf.Variable(encodings, name='embeddings')


init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    saver = tf.train.Saver()
    saver.save(session, "logs/model.ckpt", 1)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()
# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join('labels.tsv')
# Link this tensor to its metadata file (e.g. labels).
# embedding.metadata_path = os.path.join('augment_labels.tsv')

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
os.system("tensorboard --logdir=logs/")
