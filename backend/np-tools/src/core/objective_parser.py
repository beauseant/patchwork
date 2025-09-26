import subprocess
import logging
import json
import warnings
import os
warnings.simplefilter("ignore")

class ObjectiveParser:
    def __init__(self, logger: logging.Logger = None) -> None:
        """Initialize the Objective parser with a logger."""
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger(__name__)

    def parse(self, text:str) -> str:
        """
        Extract objectives from a given text using an external script.
        
        Parameters
        ----------
        text : str
            Text to extract objectives from.
        
        Returns
        -------
        str
            Extracted objectives from the text.
        """

        try:
            # Suppress stdout and stderr to avoid logging messages being captured
            command = [
                "python", "src/core/objective_tender_extraction/processOneObjective.py",
                "--document", text,
            ]
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL)  # Suppress logs
            extracted_objective = output.decode("utf-8").strip()

            # Try parsing JSON output if applicable
            try:
                return json.loads(extracted_objective)
            except json.JSONDecodeError:
                return extracted_objective

        except subprocess.CalledProcessError as e:
            print(e)
            self._logger.error(f"Error executing processOneObjective.py: {e.output.decode('utf-8')}")
            return ""

        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            return ""