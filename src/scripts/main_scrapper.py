import argparse
import json
import logging
import sys
import pickle
from pathlib import Path
from typing import Dict, Any
import time

# Agregar el directorio src al path de Python
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from pipeline.scrapper import Scrapper
import undetected_chromedriver as uc

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_cookies(driver, cookie_path: str):
    """
    Guarda las cookies de la sesión actual.
    
    Args:
        driver: Instancia del driver de Chrome
        cookie_path: Ruta donde guardar las cookies
    """
    cookies = driver.get_cookies()
    with open(cookie_path, 'wb') as f:
        pickle.dump(cookies, f)
    logger.info(f"Cookies guardadas en: {cookie_path}")

def load_cookies(driver, cookie_path: str) -> bool:
    """
    Carga las cookies de una sesión anterior.
    
    Args:
        driver: Instancia del driver de Chrome
        cookie_path: Ruta donde están guardadas las cookies
        
    Returns:
        bool: True si las cookies se cargaron exitosamente
    """
    try:
        with open(cookie_path, 'rb') as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                logger.debug(f"Error al cargar cookie: {str(e)}")
        logger.info("Cookies cargadas exitosamente")
        return True
    except FileNotFoundError:
        logger.info("No se encontraron cookies guardadas")
        return False
    except Exception as e:
        logger.error(f"Error al cargar cookies: {str(e)}")
        return False

def create_chrome_driver():
    """
    Crea una instancia de Chrome con configuraciones para evitar detección.
    
    Returns:
        uc.Chrome: Instancia configurada del driver
    """
    options = uc.ChromeOptions()
    
    # Configuraciones para evitar detección
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--start-maximized')
    options.add_argument('--disable-notifications')
    
    # User agent realista
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
    
    # Crear el driver con las opciones
    driver = uc.Chrome(
        options=options,
        headless=False,  # Forzar modo visible
        use_subprocess=True
    )
    
    # Configuraciones adicionales
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    })
    
    # Eliminar webdriver flags
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo JSON.
    
    Args:
        config_path: Ruta al archivo de configuración JSON
        
    Returns:
        Dict con la configuración cargada
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el JSON no es válido
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuración cargada desde: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error al decodificar el archivo JSON: {config_path}")
        raise

def parse_args():
    """
    Parsea los argumentos de línea de comandos y/o archivo de configuración.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Scraper para productos de Ripley'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='URL base de la página de Ripley'
    )
    parser.add_argument(
        '--page',
        type=int,
        help='Número de página a scrapear'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Directorio de salida para los archivos CSV'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Ruta al archivo de configuración JSON'
    )
    parser.add_argument(
        '--wait-time',
        type=int,
        default=10,
        help='Tiempo de espera para cargar la página (default: 10)'
    )
    parser.add_argument(
        '--cookies-dir',
        type=str,
        default='cookies',
        help='Directorio para guardar las cookies (default: cookies)'
    )
    
    args = parser.parse_args()
    
    # Si se proporciona un archivo de configuración, cargarlo
    if args.config:
        config = load_config(args.config)
        # Actualizar args con valores del config si no están definidos en línea de comandos
        for key, value in config.items():
            if not getattr(args, key, None):
                setattr(args, key, value)
    
    # Validar argumentos requeridos
    if not args.url:
        parser.error("Se requiere la URL base (--url) o un archivo de configuración (--config)")
    
    # Establecer valores por defecto si no están definidos
    args.page = args.page or 1
    args.output = args.output or 'data'
    
    return args

def main():
    """
    Función principal que ejecuta el scraper.
    """
    args = parse_args()
    
    # Crear directorios necesarios
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cookies_dir = Path(args.cookies_dir)
    cookies_dir.mkdir(parents=True, exist_ok=True)
    cookie_path = cookies_dir / "ripley_cookies.pkl"
    
    # Configurar nombre del archivo de salida
    output_file = output_dir / f"ripley_page_{args.page}.csv"
    
    try:
        logger.info("Iniciando scraper de Ripley")
        logger.info(f"URL: {args.url}")
        logger.info(f"Página: {args.page}")
        logger.info(f"Directorio de salida: {args.output}")
        
        # Crear driver con configuraciones especiales
        driver = create_chrome_driver()
        scrapper = Scrapper(driver)
        
        # Intentar cargar cookies existentes
        if not load_cookies(driver, str(cookie_path)):
            # Si no hay cookies, visitar la página principal y esperar
            logger.info("Visitando página principal para establecer cookies...")
            driver.get("https://www.ripley.com.pe")
            driver.implicitly_wait(args.wait_time)
            
            # Esperar a que el usuario resuelva el captcha si es necesario
            input("Presiona Enter después de resolver el captcha (si aparece)...")
            
            # Guardar las cookies después de la verificación
            save_cookies(driver, str(cookie_path))
        else:
            # Si hay cookies, refrescar la página principal
            driver.get("https://www.ripley.com.pe")
            driver.implicitly_wait(args.wait_time)
        
        df = scrapper.get_data_from_pagenumber(
            base_url=args.url,
            page_number=args.page,
            export_path=str(output_file)
        )
        
        # Guardar cookies después de una sesión exitosa
        save_cookies(driver, str(cookie_path))
        
        logger.info(f"Datos guardados en: {output_file}")
        logger.info(f"Total de productos encontrados: {len(df)}")
        
    except Exception as e:
        logger.error(f"Error durante el scraping: {str(e)}")
        raise
    finally:
        driver.quit()
        logger.info("Scraper finalizado")

if __name__ == "__main__":
    main()

