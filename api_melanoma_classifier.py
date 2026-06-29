import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from src.entrenamiento import ejecutar_entrenamiento

app = Flask(__name__)
app.config["DEBUG"] = True

#Cargamos el modelo previamente entrenado
MODEL_PATH = "20260621_melanoma_cnn_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado")

# Endpoint principal para servir la interfaz web
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Endpoint principal para recibir la imagen y devolver el diagnóstico.
@app.route('/predict', methods=['POST'])
def predict():

    # Verificamos que la petición contenga un archivo con la llave 'image'
    if 'image' not in request.files:
        return jsonify({"error": "No se seleccionó ninguna imagen"}), 400
        
    file = request.files['image']
    

    try:
        # Guardamos la imagen temporalmente en la computadora
        temp_path = "temp_image.jpg"
        file.save(temp_path)
        
        # Cargamos la imagen desde la ruta donde se guardó y redimensionamos a 64x64 píxeles
        img = tf.keras.utils.load_img(temp_path, target_size=(64, 64))
        
        # Borramos la imagen temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        #Convertir a matriz numérica y normalizar
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        prob_maligno = float(prediction[0][0])
        prob_benigno = 1.0 - prob_maligno
        
        #Asignar el diagnóstico final
        es_maligno = prob_maligno >= 0.5
        clase_predicha = "Maligno" if es_maligno else "Benigno"
        
        # Determinar cuál es la probabilidad y multiplicarla por 100 para mostrarla en porcentaje
        prob_mostrar = (prob_maligno if es_maligno else prob_benigno) * 100
        
        #Devolver el resultado exitoso
        return jsonify({
            "prediccion": clase_predicha,
            "probabilidad": f"{prob_mostrar:.2f}%",
            "detalles": {
                "probabilidad_benigno": f"{prob_benigno * 100:.2f}%",
                "probabilidad_maligno": f"{prob_maligno * 100:.2f}%"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint de fit el modelo
@app.route('/fit', methods=['POST'])
def fit():
    try:
        # Ejecutar el entrenamiento
        ejecutar_entrenamiento()
        
        return jsonify({"mensaje": "Entrenamiento completado exitosamente"}), 200
    except Exception as e:
        return jsonify({"error": f"Error durante el entrenamiento: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()