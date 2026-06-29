import tensorflow as tf
from tensorflow import keras

def crear_modelo_compilado(img_size=(64, 64)):
    """
    Construye y compila la arquitectura CNN original.
    """
    model = keras.Sequential()

    
    model.add(keras.layers.Input(shape=(img_size[0], img_size[1], 3)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.5))

    
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.5))

    
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Dropout(0.5))

   
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    
    return model