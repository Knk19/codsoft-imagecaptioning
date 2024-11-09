import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
import numpy as np  # Import numpy to avoid NameError

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Use np.expand_dims after importing numpy
    x = tf.keras.applications.vgg16.preprocess_input(x)
    features = model.predict(x)
    return features

# Assume captions_data is a preprocessed dataset of (image_features, caption) pairs
def create_captioning_model(vocab_size, max_caption_length):
    image_input = Input(shape=(4096,))
    image_embedding = Dense(256, activation='relu')(image_input)
    
    caption_input = Input(shape=(max_caption_length,))
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)
    
    decoder = tf.keras.layers.add([image_embedding, caption_lstm])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    
    model = Model(inputs=[image_input, caption_input], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Example usage
vocab_size = 5000  # Example vocabulary size
max_caption_length = 20  # Example max length of captions

captioning_model = create_captioning_model(vocab_size, max_caption_length)
captioning_model.summary()
