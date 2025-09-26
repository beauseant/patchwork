import logging
import os
import time
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai.chat_models import ChatOpenAI


class CPVClassifierOpenAI:
    def __init__(self, api_key, logger=None):
        """
        Initializes the CPVClassifierOpenAI with an API key and logger.
        
        Args:
            api_key (str): OpenAI API key for authentication.
            logger (logging.Logger, optional): Custom logger. Defaults to None.
        """
        self.api_key = api_key
        self.retry_wait_time = 30  # Wait time before retrying in case of rate limit (seconds)
        self.max_retries = 3  # Maximum number of retries before failing

        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Set the API key in environment variables
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Initialize the OpenAI model
        self.model = ChatOpenAI(model_name="gpt-4o-mini",  openai_api_key=self.api_key)

    def predict_cpv_code(self, objective_text):
        """
        Predicts the CPV code from a given text using the OpenAI model.
        
        Args:
            objective_text (str): The input text for which to predict the CPV code.
        
        Returns:
            dict: A dictionary containing the predicted CPV code and the original text.
        """
        attempt = 0  # Attempt counter

        while attempt < self.max_retries:
            try:
                # Create the prompt for the model
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        SystemMessagePromptTemplate.from_template(
                            "Eres un experto en codificación CPV. Debes proporcionar únicamente códigos CPV válidos de 5 dígitos."
                        ),
                        HumanMessagePromptTemplate.from_template(
                            "Dado el siguiente texto: '{objective_text}', proporciona solo los primeros 5 dígitos del código CPV. "
                            "Debe ser un número de exactamente 5 dígitos. Solo números, sin texto adicional ni espacios.\n\n"
                            "Ejemplo:\n"
                            "Texto: 'Servicios de mantenimiento de parques.'\n"
                            "Respuesta: 77311"
                        )
                    ]
                )

                # Format the message
                prompt = prompt_template.format_messages(objective_text=objective_text)

                # Call the OpenAI model
                response = self.model(prompt)

                # Extract the generated text response
                cpv_code = response.content.strip()

                # Keep only numeric characters
                cpv_code_filtered = ''.join(filter(str.isdigit, cpv_code))

                # Adjust the length of the CPV code
                if len(cpv_code_filtered) > 5:
                    cpv_code_filtered = cpv_code_filtered[:5]
                elif len(cpv_code_filtered) == 4:
                    cpv_code_filtered += '0'

                # Validate that the final CPV code is exactly 5 digits
                if len(cpv_code_filtered) == 5:
                    self.logger.info(f"Predicted CPV code: {cpv_code_filtered}")
                    return {
                        "original_text": objective_text,
                        "cpv_predicted": cpv_code_filtered
                    }
                else:
                    self.logger.warning(f"Invalid CPV code predicted: {cpv_code_filtered}")
                    return None

            except Exception as e:
                self.logger.error(f"Error processing text '{objective_text}': {e}")

                # Handle OpenAI rate limit errors
                if "rate_limit_exceeded" in str(e):
                    attempt += 1
                    self.logger.warning(
                        f"Rate limit reached. Retrying in {self.retry_wait_time} seconds... (Attempt {attempt}/{self.max_retries})"
                    )
                    time.sleep(self.retry_wait_time)
                else:
                    return None  # If an error other than rate limit occurs, return None immediately

        self.logger.error(f"Max retries reached for text: {objective_text}")
        return None  # If max retries are reached, return None
