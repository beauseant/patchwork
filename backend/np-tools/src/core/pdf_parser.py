import subprocess
import logging
import json
import warnings
warnings.simplefilter("ignore")

class PDFparser:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Initialize the PDF parser with a logger."""
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger(__name__)

    def parse(self, pdf_path: str, output_dir: str = "/home/sblanco/tmp/") -> str:
        """
        Extract text from a text-based PDF using an external script.

        Parameters
        ----------
        pdf_path : str
            Path to the input PDF file.
        output_dir : str, optional
            Directory where extracted text will be stored. Default is "/home/sblanco/tmp/".

        Returns
        -------
        str
            Extracted text from the PDF.
        """

        try:
            # Suppress stdout and stderr to avoid logging messages being captured
            command = [
                "python", "src/core/pdf_extractor/processOnePDF.py",
                "--pdf_path", pdf_path,
                "--path_save", output_dir,
                "--output"
            ]
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)  # Suppress logs
            extracted_text = output.decode("utf-8").strip()

            # Try parsing JSON output if applicable
            try:
                parsed_output = json.loads(extracted_text)
                return parsed_output.get("raw_text", extracted_text)
            except json.JSONDecodeError:
                return extracted_text

        except subprocess.CalledProcessError as e:
            self._logger.error(f"Error executing processOnePDF.py: {e.output.decode('utf-8')}")
            return ""

        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            return ""