# Melanoma Classifier

Este proyecto es una API construida con Flask y TensorFlow/Keras para clasificar imágenes de lesiones en la piel como **Malignas** o **Benignas** (melanoma). También proporciona una interfaz web simple para cargar la imagen y recibir el diagnóstico.

## Requisitos

- Python 3.11
- Se recomienda usar un entorno virtual

## Instalación

1. Clona este repositorio o descarga los archivos.
2. Crea un entorno virtual:
   ```bash
   python -m venv env
   ```
3. Activa el entorno virtual:
   - En Windows:
     ```bash
     env\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source env/bin/activate
     ```
4. Instala las dependencias necesarias. Ejecuta:
   ```bash
   pip install -r requirements.txt
   ```
5. Descarga el dataset de imágenes usando DVC:
   ```bash
   dvc pull
   ```
   *(Nota: Asegúrate de configurar primero tu almacenamiento remoto de DVC con Google Drive).*

## Uso

1. Para iniciar el servidor de desarrollo de Flask, ejecuta:
   ```bash
   python api_melanoma_classifier.py
   ```
2. La API estará disponible localmente en `http://127.0.0.1:5000/`.

## Endpoints

### 1. Interfaz Web (GET `/`)
Muestra la página principal con el formulario para subir la imagen.

### 2. Predicción (POST `/predict`)
Recibe una imagen y devuelve el diagnóstico.
- **Form-Data**:
  - `image`: Archivo de imagen (JPG, PNG, etc.)
- **Respuesta Exitosa (Ejemplo)**:
  ```json
  {
      "prediccion": "Maligno",
      "probabilidad": "85.20%",
      "detalles": {
          "probabilidad_benigno": "14.80%",
          "probabilidad_maligno": "85.20%"
      }
  }
  ```

### 3. Entrenamiento (POST `/fit`)
Ejecuta la rutina de entrenamiento del modelo.
- **Respuesta Exitosa**:
  ```json
  {
      "mensaje": "Entrenamiento completado exitosamente"
  }
  ```

## Estructura del Proyecto

- `api_melanoma_classifier.py`: Archivo principal que define la API de Flask y sus rutas.
- `20260621_melanoma_cnn_classifier.keras`: Modelo pre-entrenado de TensorFlow/Keras.
- `src/`: Contiene el código fuente adicional.
- `static/`: Archivos estáticos como CSS, JavaScript.
- `templates/`: Plantillas HTML (como `index.html`).
- `data/`: Directorio donde se encuentra el dataset de imágenes ISIC con subcarpetas /test, /train y /validation, cada una con subcarpetas benigna y maligna.
