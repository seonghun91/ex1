

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

seed = 1234
tf.random.set_seed(seed)
batch_size = "no"
version = "ver1"

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 512
    max_length = 300
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json') as f:
      datas = json.load(f)

    for data in datas:
      sentences.append(data['headline'])
      labels.append(data['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(2048, activation='relu'),
    Dense(2048, activation='relu'),
    Dense(2048, activation='relu'),
    Dense(2048, activation='relu'),
    Dense(1, activation='sigmoid')
    ])


    # model = tf.keras.Sequential([
    #     # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                save_weights_only=True, 
                                save_best_only=True, 
                                monitor='val_loss',
                                verbose=1)
    
    epochs=300

    history = model.fit(train_padded, train_labels, 
                    validation_data=(validation_padded, validation_labels),
                    callbacks=[checkpoint],
                    epochs=epochs)
    
    model.load_weights(checkpoint_path)
    valid_loss = min(history.history['val_loss'])
    return model, valid_loss


# Note that you'll need to save your model as a .h5 like thissssssssssssssssssssssssssssssssssssssss
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model, valid_loss = solution_model()
    # model.save("TF4-sarcasm.h5")
    # sssssssssssssssssssssssssssssssssssssssssssssssssszzzzzzzzmodel.save("./TF4-sarcasm-"+str(round(valid_loss,5))+"-seed-"+str(seed)+"-"+version+".h5")
