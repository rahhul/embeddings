# python3

import os
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

print(os.getcwd())

SHUFFLE_BUFFER_SIZE = 30000
BATCH_SIZE = 128


# get dataset
def load_dataset(name):
    """Loads dataset using tfds
    and returns a dataset
    """
    dataset, info = tfds.load(name=name,
                              with_info=True,
                              data_dir='data/external')
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE,
                                          reshuffle_each_iteration=False)

    return train_dataset


# tokenize text
def tokenize_text(dataset):
    """Tokenize text and save a vocabulary file
    """
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary = set() # removes duplicates
    for _, reviews in dataset.enumerate():
        review_text = reviews['data']
        reviews_tokens = tokenizer.tokenize(review_text.get('review_body').numpy())
        # add to vocabulary set
        vocabulary.update(reviews_tokens)

    return vocabulary

encoder = tfds.features.text.TokenTextEncoder.load_from_file('vocab')
# encoder.load_from_file("vocab")

# function to encode review_body text
def encode(text_tensor, label_tensor):
    """Encodes dataset with the encoder.
    """
    # encode text
    encoded_text = encoder.encode(text_tensor.numpy())
    label = tf.where(label_tensor > 3, 1, 0)
    return encoded_text, label


def encode_map_fn(tensor):
    
    text = tensor['data'].get('review_body')
    label = tensor['data'].get('star_rating')
    
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int32))
    # set shapes for eager
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label



def split_dataset(dataset, test_size):
    """Split dataset into 
    train and test sets"""
    train_data = dataset.skip(test_size).shuffle(SHUFFLE_BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE)
    
    test_data = dataset.take(test_size)
    test_data = test_data.padded_batch(BATCH_SIZE)
    
    return train_data, test_data



