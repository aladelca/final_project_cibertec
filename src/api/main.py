import sys
from pathlib import Path
import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import joblib
import pandas as pd

# Obtener la ruta absoluta del directorio raíz del proyecto
ROOT_PATH = Path(__file__).parent.parent.parent
SRC_PATH = ROOT_PATH / "src"

# Agregar el directorio raíz al path de Python
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# Importar módulos después de configurar el path
from src.pipeline.model import Model
from src.pipeline.preprocessing import MyPreprocessor

app = FastAPI(
    title="API de Predicción de Precios",
    description="API para predecir precios de productos usando descripción e imagen",
    version="1.0.0"
)

# Cargar modelo y preprocessor al iniciar la aplicación
MODEL_PATH = SRC_PATH / "models" / "modelo.joblib"
PREPROCESSOR_PATH = SRC_PATH / "models" / "preprocessor.joblib"

try:
    print(f"Intentando cargar modelo desde: {MODEL_PATH}")
    print(f"Intentando cargar preprocessor desde: {PREPROCESSOR_PATH}")
    
    # Verificar que los archivos existan
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo del preprocessor en: {PREPROCESSOR_PATH}")
    
    model = joblib.load(str(MODEL_PATH))
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    print("Modelo y preprocessor cargados exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo o preprocessor: {str(e)}")
    model = None
    preprocessor = None

class PredictionResponse(BaseModel):
    prediccion: list
    precio_predicho: list

def process_image_file(file: UploadFile) -> np.ndarray:
    """
    Procesa un archivo de imagen subido y lo convierte a array numpy.
    
    Args:
        file: Archivo de imagen subido
        
    Returns:
        np.ndarray: Array de la imagen
    """
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")
    finally:
        file.file.close()

@app.get("/")
async def root():
    """Endpoint de prueba para verificar que la API está funcionando."""
    return {"message": "API de Predicción de Precios funcionando"}

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    descripcion: str = Form(...),
    imagen: UploadFile = File(...)
):
    """
    Endpoint para realizar predicciones usando descripción e imagen.
    
    Args:
        descripcion: Descripción del producto
        imagen: Archivo de imagen del producto
        
    Returns:
        Dict con las predicciones
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo o preprocessor no cargados correctamente"
        )
    
    try:
        # Procesar la imagen
        img_array = process_image_file(imagen)
        
        # Crear DataFrame con los datos
        data = pd.DataFrame({
            'descripcion': [descripcion],
            preprocessor.img_column_name: [img_array]
        })
        
        # Preprocesar datos
        processed_data = preprocessor.transform(data)
        
        # Realizar predicción
        predictions = model.predict(processed_data)
        
        return {
            "prediccion": predictions.tolist(),
            "precio_predicho": predictions.tolist()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )

@app.post("/predict/batch", response_model=list[PredictionResponse])
async def make_batch_prediction(
    descripciones: list[str] = Form(...),
    imagenes: list[UploadFile] = File(...)
):
    """
    Endpoint para realizar predicciones en lote.
    
    Args:
        descripciones: Lista de descripciones de productos
        imagenes: Lista de archivos de imagen
        
    Returns:
        Lista de predicciones
    """
    if len(descripciones) != len(imagenes):
        raise HTTPException(
            status_code=400,
            detail="El número de descripciones debe coincidir con el número de imágenes"
        )
    
    predictions = []
    for descripcion, imagen in zip(descripciones, imagenes):
        try:
            # Procesar la imagen
            img_array = process_image_file(imagen)
            
            # Crear DataFrame con los datos
            data = pd.DataFrame({
                'descripcion': [descripcion],
                'imagen': [img_array]
            })
            
            # Preprocesar datos
            processed_data = preprocessor.transform(data)
            
            # Realizar predicción
            prediction = model.predict(processed_data)
            predictions.append({
                "prediccion": prediction.tolist(),
                "precio_predicho": prediction.tolist()
            })
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al procesar el item {len(predictions) + 1}: {str(e)}"
            )
    
    return predictions

def start():
    """Función para iniciar el servidor."""
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    start() 