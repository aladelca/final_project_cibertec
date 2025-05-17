from PIL import Image
import numpy as np
import pandas as pd
import torch
from fashion_clip.fashion_clip import FashionCLIP
import unicodedata
import nltk
nltk.download('stopwords')
nltk.download('punkt')  # para el tokenizador
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class MyPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, img_column_name: str, price_column_name: str):
        self.stop_es = set(stopwords.words('spanish'))
        self.img_column_name = img_column_name
        self.price_column_name = price_column_name
        self.vectorizer = TfidfVectorizer()
        return None
    def _get_array(self,df: pd.DataFrame, column_name: str) -> np.ndarray:
        """
        Convierte una columna de imágenes en un array numpy.

        Args:
            df: DataFrame con la columna de imágenes
            column_name: Nombre de la columna que contiene las imágenes
        Returns:
            Array numpy con las imágenes
        """
        return np.array([Image.fromarray(arr.astype('uint8')) for arr in df[column_name]])
    def _get_embeddings(self,df: pd.DataFrame, column_name: str) -> np.ndarray:
        """
        Obtiene los embeddings de las imágenes.

        Args:
            df: DataFrame con la columna de imágenes
            column_name: Nombre de la columna que contiene las imágenes
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        images = self._get_array(df, column_name)
        fclip = FashionCLIP("fashion-clip")
        fclip.model.to(device)
        fclip.model.eval()
        # Extrae embeddings en batches
        embeddings = fclip.encode_images(images, batch_size=32)
        df["embeddings"] = list(embeddings)
        return df

    def _filter_rows_wo_price(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra las filas que no tienen precio.
        """
        return df[df["precio_oferta"]!=" "]

    def _clean_price(self,df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df[column_name + "_limpio"] = df[column_name].str.replace("S/ ", "").astype(float)
        return df

    def _quitar_tildes(self,texto: str) -> str:
        # Descompone los caracteres en base + marcas diacríticas (NFKD)
        texto_norm = unicodedata.normalize("NFKD", texto)
        # Filtra y elimina las marcas diacríticas
        return "".join(c for c in texto_norm if not unicodedata.combining(c))
    def _quitar_stopwords(self,texto: str) -> list[str]:
        tokens = word_tokenize(texto, language='spanish')
        tokens_filtrados = [
            tok for tok in tokens
            if tok.lower() not in self.stop_es
            and tok not in string.punctuation
        ]
        return tokens_filtrados
    def fit(self, df: pd.DataFrame, y=None):
        df = self._get_embeddings(df, self.img_column_name)
        # Considerar solamente en el fit
        df = self._filter_rows_wo_price(df)
        df = self._clean_price(df, self.price_column_name)
        df["descripcion_limpio"] = df["descripcion"].apply(self._quitar_tildes)
        df["descripcion_limpio"] = df["descripcion_limpio"].apply(self._quitar_stopwords)
        df["descripcion_limpio"] = df["descripcion_limpio"].apply(self._tokenizar)
        self.vectorizer.fit(df["descripcion_limpio"])
        descripcion_vectorizada = self.vectorizer.transform(df["descripcion_limpio"]).toarray()
        final_array = np.concatenate([np.stack(df['embeddings'].values), descripcion_vectorizada], axis=1)
        return final_array, df[self.price_column_name + "_limpio"]
    
    def fit_transform(self, df: pd.DataFrame, y=None):
        x, y = self.fit(df)
        return x,y

    def transform(self, df: pd.DataFrame):
        df = self._get_embeddings(df, self.img_column_name)
        df["descripcion_limpio"] = df["descripcion"].apply(self._quitar_tildes)
        df["descripcion_limpio"] = df["descripcion_limpio"].apply(self._quitar_stopwords)
        df["descripcion_limpio"] = df["descripcion_limpio"].apply(self._tokenizar)
        descripcion_vectorizada = self.vectorizer.transform(df["descripcion_limpio"]).toarray()
        final_array = np.concatenate([np.stack(df['embeddings'].values), descripcion_vectorizada], axis=1)
        return final_array




