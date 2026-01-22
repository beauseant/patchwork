import argparse
import os
import fitz  # PyMuPDF
import re

class PDFTextExtractor:
    """
    Clase para leer un fichero Parquet, abrir los PDFs indicados en las columnas
    de rutas, extraer su texto y guardar el resultado en nuevas columnas.
    """
    # Mensajes estandarizados
    MSG_PDF_ESCANEO = "[AVISO: PDF sin texto extraíble (posiblemente escaneado)]" 
    MSG_NO_EXISTE = "[ERROR: El fichero  PDF no existe en la ruta especificada]" 
    MSG_RUTA_INVALIDA = "[ERROR: La ruta al PDF es inválida o no se descargó]"
    MSG_PDF_FILTRADO = "[ERROR: El PDF no contiene información válida]"

    MATCHES = [
            'Consiga Adobe Reader',
            'un error de acceso',
            'una copia impresa del documento electrónico',
            'El documento no aplica a la contratación específica'
            ]

    def __init__(self):
        pass

    def __esFiltrable__ (self, texto):
        return any(x in texto for x in self.MATCHES)



    def __esTextoBasura__(self, texto, umbral_ratio=0.5):
        """
        Evalúa si un texto es "basura" de codificación basándose en 
        la proporción de letras y espacios.
        
        Args:
            texto (str): El texto a analizar.
            umbral_ratio (float): Si el ratio de "caracteres buenos" es MENOR 
                                que este umbral, se considera basura. 
                                (0.5 = 50%)
        
        Returns:
            bool: True si es basura, False si parece texto normal.
        """
        
        if not texto or len(texto) < 10:
            # Si es muy corto o está vacío, lo consideramos basura
            return True

        # 1. Contar caracteres que son letras (a-z) o espacios
        # Incluimos letras con acentos y ñ
        letras_y_espacios = re.findall(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ\s]', texto)
        
        # 2. Calcular el ratio
        ratio = len(letras_y_espacios) / len(texto)
        
        # 3. Decidir
        # Si menos del 50% del texto son letras o espacios, 
        # es muy probable que sea basura.
        return ratio < umbral_ratio

    def extraerTexto(self, pdf_path: str) -> str:
        """
        Extrae el contenido de texto de un único fichero PDF.

        Args:
            pdf_path (str): Ruta al fichero PDF.

        Returns:
            str: El texto extraído o un mensaje de error/aviso.
        """
        try:
                documento = fitz.open(pdf_path)
        except Exception as e:            
            return self.MSG_NO_EXISTE

        partes_texto = []
        for page in documento:
            partes_texto.append(page.get_text())
            
        texto_completo = "".join(partes_texto).strip()
        documento.close()

        # --- NUEVA VERIFICACIÓN ---
        if self.__esFiltrable__ (texto_completo):
            return self.MSG_PDF_FILTRADO
        else:
            if self.__esTextoBasura__ (texto_completo, umbral_ratio=0.5):
                # Si la función detecta basura, devolvemos el error
                return self.MSG_PDF_ESCANEO
            else:
                # El texto parece correcto, continuamos
                print(f"Procesando OK: {pdf_path}")
                # Aquí iría tu lógica de procesamiento normal
                return texto_completo




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae texto de los PDFs referenciados en un fichero Parquet.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "pdf_file", 
        help="Ruta al fichero PDF."
    )

    parser.add_argument(
        "txt_file", 
        help="Ruta al fichero txt."
    )

    
    args = parser.parse_args()
    
    extractor = PDFTextExtractor()

    with open(args.txt_file, "w") as f:
       f.write (extractor.extraerTexto ( args.pdf_file))