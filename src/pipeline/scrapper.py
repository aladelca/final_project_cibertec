import logging
from typing import List, Tuple
import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Scrapper:
    """
    Clase para realizar web scraping de productos de Ripley.
    
    Attributes:
        driver: Instancia del driver de Selenium configurado
    """
    
    def __init__(self, driver: uc.Chrome, base_url: str):
        """
        Inicializa el Scrapper con un driver de Selenium.
        
        Args:
            driver: Instancia de undetected_chromedriver
        """
        self.driver = driver
        self.driver.get(base_url)
        self.driver.maximize_window()
        logger.info("Scrapper initialized with Chrome driver")

    def _find_element_safe(self, xpath: str, timeout: int = 3) -> str:
        """
        Busca un elemento de forma segura, retornando cadena vacía si no se encuentra.
        
        Args:
            xpath: Ruta XPath del elemento a buscar
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            str: Texto del elemento o cadena vacía si no se encuentra
        """
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            return element.text
        except (TimeoutException, NoSuchElementException) as e:
            logger.debug(f"Element not found at xpath {xpath}: {str(e)}")
            return ""

    def _find_image_safe(self, xpaths: List[str], timeout: int = 3) -> str:
        """
        Busca una imagen entre múltiples posibles XPaths.
        
        Args:
            xpaths: Lista de XPaths donde buscar la imagen
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            str: URL de la imagen o cadena vacía si no se encuentra
        """
        for xpath in xpaths:
            try:
                element = WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                return element.get_attribute("src")
            except (TimeoutException, NoSuchElementException):
                continue
        logger.debug("No image found in any of the provided xpaths")
        return ""

    def _scroll_page(self, start: int = 0, step: int = 800, max_scrolls: int = 30):
        """
        Realiza scroll en la página de forma gradual.
        
        Args:
            start: Posición inicial del scroll
            step: Incremento por cada scroll
            max_scrolls: Número máximo de scrolls a realizar
        """
        current = start
        for i in range(max_scrolls):
            try:
                ActionChains(self.driver).scroll_by_amount(current, step).perform()
                current += step
                time.sleep(0.2)  # Reducido a 0.2 segundos
            except Exception as e:
                logger.error(f"Error durante el scroll: {str(e)}")
                break

    def _wait_for_page_load(self, timeout: int = 10):
        """
        Espera a que la página cargue completamente.
        
        Args:
            timeout: Tiempo máximo de espera en segundos
        """
        try:
            self.wait.until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            logger.info("Página cargada completamente")
        except TimeoutException:
            logger.warning("Timeout esperando la carga de la página")

    def scrap_ripley(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Realiza el scraping de productos de Ripley.
        
        Returns:
            Tuple con listas de descripciones, precios regulares, 
            precios oferta e imágenes
        """
        logger.info("Starting Ripley scraping process")
        
        descripciones = []
        precios_regular = []
        precios_oferta = []
        imagenes = []
        
        # Esperar a que la página cargue
        #self._wait_for_page_load()
        
        # XPaths para imágenes (ordenados por prioridad)
        image_xpaths = [
            "//div[2]/div[2]/div[1]/img",
            "//div[2]/div[2]/div[4]/img",
            "//div[2]/div[2]/div[3]/img",
            "//div[2]/div[2]/div[2]/img"
        ]
        
        # Base XPath para productos
        base_xpath = "/html/body/div[7]/div[2]/div/div[2]/section/div/div/div[{}]"
        
        # Scroll inicial más rápido
        #self._scroll_page(step=1000, max_scrolls=20)
        
        for i in range(1, 50):
            logger.info(f"Processing product {i}")
            
            # Construir XPaths completos
            product_xpath = base_xpath.format(i)
            desc_xpath = f"{product_xpath}/div/a/div[3]/div[3]"
            price_reg_xpath = f"{product_xpath}/div/a/div[3]/div[4]/div/ul/li[1]"
            price_off_xpath = f"{product_xpath}/div/a/div[3]/div[4]/div/ul/li[2]"
            img_xpaths = [f"{product_xpath}/div/a{path}" for path in image_xpaths]
            
            # Obtener datos con reintentos
            max_retries = 2
            for retry in range(max_retries):
                try:
                    desc = self._find_element_safe(desc_xpath)
                    price_reg = self._find_element_safe(price_reg_xpath)
                    price_off = self._find_element_safe(price_off_xpath)
                    img = self._find_image_safe(img_xpaths)
                    
                    descripciones.append(desc)
                    precios_regular.append(price_reg)
                    precios_oferta.append(price_off)
                    imagenes.append(img)
                    logger.info(f"Desc: {desc}")
                    logger.info(f"Price reg: {price_reg}")
                    logger.info(f"Price off: {price_off}")
                    logger.info(f"Img: {img}")
                    logger.debug(f"Producto {i} procesado exitosamente")
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        logger.error(f"Error procesando producto {i} después de {max_retries} intentos: {str(e)}")
                        descripciones.append("")
                        precios_regular.append("")
                        precios_oferta.append("")
                        imagenes.append("")
                    else:
                        time.sleep(0.5)  # Reducido a 0.5 segundos
            
            # Scroll cada 10 productos en lugar de 5
            if i % 10 == 0:
                self._scroll_page(step=800)
        
        logger.info(f"Scraping completed. Found {len(descripciones)} products")
        return descripciones, precios_regular, precios_oferta, imagenes

    def get_data_from_pagenumber(
            self,
            base_url: str,
            page_number: int,
            export_path: str
    ) -> pd.DataFrame:
        """
        Obtiene datos de una página específica.
        
        Args:
            base_url: URL base de la página
            page_number: Número de página a scrapear
            export_path: Ruta donde guardar el CSV
            
        Returns:
            DataFrame con los datos obtenidos
        """
        url = f"{base_url}&page={page_number}"
        logger.info(f"Scraping page {page_number}: {url}")
        
        self.driver.get(url)
        self.driver.maximize_window()
        self.driver.implicitly_wait(5)  # Reducido a 5 segundos
        
        # Esperar a que la página cargue completamente
        self._wait_for_page_load()
        
        data = self.scrap_ripley()
        if export_path:
            self.transform_to_df(*data).to_csv(export_path, index=False)
        return self.transform_to_df(*data)

    def transform_to_df(
            self,
            descripciones: List[str],
            precios_regular: List[str],
            precios_oferta: List[str],
            imagenes: List[str]
    ) -> pd.DataFrame:
        """
        Transforma los datos obtenidos en un DataFrame.
        
        Args:
            descripciones: Lista de descripciones de productos
            precios_regular: Lista de precios regulares
            precios_oferta: Lista de precios oferta
            imagenes: Lista de URLs de imágenes
            
        Returns:
            DataFrame con los datos organizados
        """
        return pd.DataFrame({
            "descripcion": descripciones,
            "precio_regular": precios_regular,
            "precio_oferta": precios_oferta,
            "imagen": imagenes
        })
    
