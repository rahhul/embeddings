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
    

# tokenize text
def tokenize_text(dataset):
    """Tokenize text and save a vocabulary file
    """
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary = set()
    for _, reviews in dataset.enumerate():
        review_text = reviews['data']
        reviews_tokens = tokenizer.tokenize(review_text.get('review_body').numpy())
        # add to vocabulary set
        vocabulary.update(reviews_tokens)
    
    # encode vocabulary
    encoder = tfds.features.text.TokenTextEncoder(vocabulary)
    # encoder.save_to_file('vocab')
    print("Saved vocabulary file.")    
        
    return len(vocabulary)

def foo(x):
    if x is 'foo':
        return 'bar'
    else :
        return 'baz'
    
        
data_name = 'amazon_us_reviews/Mobile_Electronics_v1_00'

sample_dataset = load_dataset(name=data_name)
sample_vocab = tokenize_text(sample_dataset)
print(sample_vocab)