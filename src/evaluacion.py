import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns            
import mlflow  
import os  

def evaluar_modelo():
    ruta_modelo = "20260621_melanoma_cnn_classifier.keras"
    ruta_test = "data/Data_Melanoma_Cancer/test"
    
    print("Cargando el modelo...")
    model = tf.keras.models.load_model(ruta_modelo)

    print("Cargando datos de prueba (64x64)...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        ruta_test,
        image_size=(64, 64), 
        batch_size=32,
        shuffle=False, 
        label_mode='binary'
    )
    
    
    class_names = test_ds.class_names 

    normalizer = tf.keras.layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE
    
    test_model_ds = test_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=AUTOTUNE)
    test_model_ds = test_model_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Extrayendo etiquetas reales...")
    y_parts = []
    for _, labels in test_ds:
        y_parts.append(labels.numpy())
    y_true = np.concatenate(y_parts, axis=0)
    y_true = np.asarray(y_true).reshape(-1).astype(int)

    print("Calculando Loss y Accuracy...")
    loss, accuracy = model.evaluate(test_model_ds, verbose=0)

    print("Generando predicciones...")
    preds = model.predict(test_model_ds, verbose=0)
    preds = np.asarray(preds).reshape(-1)
    y_pred = (preds >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    metricas = {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "matriz_confusion": {
            "verdaderos_negativos": int(cm[0][0]),
            "falsos_positivos": int(cm[0][1]),
            "falsos_negativos": int(cm[1][0]),
            "verdaderos_positivos": int(cm[1][1])
        }
    }

    print("Guardando métricas en metricas.json...")
    with open("metricas.json", "w") as f:
        json.dump(metricas, f, indent=4)
        
    print("Registrando métricas de Test en MLflow...")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Clasificador_Melanoma")
    
    with mlflow.start_run(run_name="Evaluacion_Test_Final"):
        mlflow.log_metric("test_accuracy", metricas["accuracy"])
        mlflow.log_metric("test_loss", metricas["loss"])
        
        mlflow.log_param("falsos_negativos", metricas["matriz_confusion"]["falsos_negativos"])
        mlflow.log_param("verdaderos_positivos", metricas["matriz_confusion"]["verdaderos_positivos"])
        
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicha")
        plt.ylabel("Real")
        plt.title("Matriz de confusión-Predicción de imágenes de tipos de melanoma")
        
        imagen_temporal = "matriz_confusion_temp.png"
        plt.savefig(imagen_temporal)
        plt.close() 
        
        mlflow.log_artifact(imagen_temporal)
        
        if os.path.exists(imagen_temporal):
            os.remove(imagen_temporal)
            
    print("¡Validación completada con éxito!")

if __name__ == "__main__":
    evaluar_modelo()