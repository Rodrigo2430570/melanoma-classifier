import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
from src.procesamiento import load_and_prep_data
from src.modelo import crear_modelo_compilado

def ejecutar_entrenamiento():
    ruta_datos = "data/Data_Melanoma_Cancer"
    nombre_modelo = "20260621_melanoma_cnn_classifier.keras"

    print("Preparando flujos de datos...")
    train_ds = load_and_prep_data(ruta_datos, "train", shuffle=True)
    val_ds = load_and_prep_data(ruta_datos, "validation", shuffle=False)

    print("Instanciando arquitectura de la CNN...")
    model = crear_modelo_compilado()

    
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

   
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Clasificador_Melanoma")
    mlflow.tensorflow.autolog() 

    with mlflow.start_run():
        print("Iniciando fase de entrenamiento (10 épocas máximo)...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[callback]
        )

        print(f"Exportando modelo a disco: {nombre_modelo}")
        model.save(nombre_modelo)

if __name__ == "__main__":
    ejecutar_entrenamiento()