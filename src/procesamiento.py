import tensorflow as tf
from pathlib import Path


IMG_SIZE = (64, 64)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_and_prep_data(ruta_base, split, shuffle=False):
    """
    Carga, redimensiona y normaliza las imágenes aplicando caché en memoria.
    Equivalente a tus funciones load_ds() y prep() originales.
    """
    ruta_completa = str(Path(ruta_base) / split)
    
    # 1. Carga del directorio
    ds = tf.keras.utils.image_dataset_from_directory(
        ruta_completa,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode='binary'
    )
    
    # 2. Normalización (1.0 / 255)
    normalizer = tf.keras.layers.Rescaling(1.0 / 255)
    
    # 3. Optimización (map, cache y prefetch)
    ds = ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)