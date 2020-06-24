# python3

import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

SHUFFLE_BUFFER_SIZE = 30000

# get dataset
def load_dataset(name):
    """Loads dataset using tfds
    and returns a dataset
    """
    dataset, info = tfds.load(name=name,
                              with_info=True,
                              data_dir='./data/external')
    train_dataset = dataset['train']
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE,
                                          reshuffle_each_iteration=False)
    
    return train_dataset
    
      

def foo(x):
    if x is 'foo':
        return 'bar'
    else :
        return 'baz'
    
        
