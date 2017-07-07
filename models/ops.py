import numpy as np
import tensorflow as tf

def bow(dict_size, sentence):
    bow = np.zeros( (dict_size, ), dtype = np.float32 )
    for w in sentence:
        bow[w] += 1
    return bow


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

