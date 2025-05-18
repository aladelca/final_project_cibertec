import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from pipeline.model import Model
from pipeline.preprocessing import MyPreprocessor

def load_object(path: str) -> Model:
    """Carga un objeto guardado con joblib."""
    return joblib.load(path)

def load_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen y la convierte a array numpy.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        np.ndarray: Array de la imagen en formato BGR
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return img
    except Exception as e:
        raise Exception(f"Error al cargar la imagen: {str(e)}")

def process_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Procesa los datos de entrada (descripción e imagen) y los convierte a DataFrame.
    
    Args:
        input_data: Diccionario con descripción e imagen
        
    Returns:
        pd.DataFrame: DataFrame con los datos procesados
    """
    try:
        # Cargar y procesar la imagen
        img_array = load_image(input_data['imagen'])
        
        # Crear DataFrame con los datos
        data = pd.DataFrame({
            'descripcion': [input_data['descripcion']],
            'imagen': [img_array]
        })
        
        return data
    except Exception as e:
        raise Exception(f"Error al procesar los datos de entrada: {str(e)}")

def preprocess_data(data: pd.DataFrame, preprocessor_path: str) -> pd.DataFrame:
    """
    Preprocesa los datos usando el preprocessor guardado.
    
    Args:
        data: DataFrame con los datos a preprocesar
        preprocessor_path: Ruta al preprocessor guardado
        
    Returns:
        pd.DataFrame: Datos preprocesados
    """
    try:
        preprocessor = load_object(preprocessor_path)
        return preprocessor.transform(data)
    except Exception as e:
        raise Exception(f"Error en el preprocesamiento: {str(e)}")

def predict(model: Model, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza predicciones con el modelo.
    
    Args:
        model: Modelo entrenado
        data: DataFrame con los datos preprocesados
        
    Returns:
        Dict con las predicciones
    """
    try:
        predictions = model.predict(data)
        return {
            'prediccion': predictions.tolist(),
            'precio_predicho': model.predict(data).tolist()
        }
    except Exception as e:
        raise Exception(f"Error al realizar predicciones: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Realizar predicciones con el modelo')
    parser.add_argument('--input', type=str, required=True, help='Ruta al archivo JSON de entrada')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo guardado')
    parser.add_argument('--preprocessor', type=str, required=True, help='Ruta al preprocessor guardado')
    parser.add_argument('--output', type=str, help='Ruta para guardar las predicciones (opcional)')
    
    args = parser.parse_args()
    
    try:
        # Cargar datos de entrada
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Procesar datos
        data = process_input(input_data)
        
        # Preprocesar datos
        processed_data = preprocess_data(data, args.preprocessor)
        
        # Cargar modelo y hacer predicciones
        model = load_object(args.model)
        predictions = predict(model, processed_data)
        
        # Guardar o mostrar resultados
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            print(f"Predicciones guardadas en: {args.output}")
        else:
            print(json.dumps(predictions, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

    
