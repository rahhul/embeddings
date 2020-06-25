import os
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import load_dataset, tokenize_text, encode_map_fn, split_dataset
import datetime

print(os.getcwd())

data_name = 'amazon_us_reviews/Mobile_Electronics_v1_00'

train_dataset = load_dataset(name=data_name)
vocabulary = tokenize_text(train_dataset)
print(len(vocabulary))
vocab_size = len(vocabulary) + 1
print(f"Vocabulary size: {vocab_size}")

# tokenize text
encoder = tfds.features.text.TokenTextEncoder(vocabulary)
encoder.save_to_file('vocab')
print("Saved vocabulary file.")

# apply encoding to dataset
encoded_dataset = train_dataset.map(encode_map_fn)
train_data, test_data = split_dataset(encoded_dataset, test_size=10000)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 128))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
for units in [64,64]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

print(model.summary())

logdir = os.path.join(
    "/training/logs", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='/training/sentiment_analysis.hdf5', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data, epochs=2, validation_data=test_data,
                    callbacks=[tensorboard_callback, checkpointer])

model.save('/training/final_model.hdf5')
