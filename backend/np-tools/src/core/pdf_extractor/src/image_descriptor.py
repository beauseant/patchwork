"""
This class utilizes OpenAI's API to describe and label images using the GPT-4 model. It encodes images as base64 strings and handles rate limits with exponential backoff. 

Author: Lorena Calvo-Bartolomé
Date: 04/02/2024
"""

import base64
import os
import requests
import pathlib
import backoff
import openai
from dotenv import load_dotenv


class ImageDescriptor(object):

    def __init__(self):

        # Load the API key from the .env file
        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

    def _encode_image(
            self,
            image_path: pathlib.Path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def describe_image(
        self,
        image_path: pathlib.Path
    ) -> str:
        """Generates a description of the image using OpenAI's API.

        Parameters

        """
        # Getting the base64 string
        base64_image = self._encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image? Do not use the first person to describe it, make an impersonal description."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()["choices"][0]["message"]["content"]

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def get_label_image(
        self,
        image_path: pathlib.Path
    ) -> str:
        """Generates a description of the image using OpenAI's API.

        Parameters

        """
        # Getting the base64 string
        base64_image = self._encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Give me a label to save this image. If characterized by several words, unify it with underscores. If you think the image is just a logo, then write 'logo'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response.json()["choices"][0]["message"]["content"]
