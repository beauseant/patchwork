import json
import logging
import os
from typing import Union, List
from dotenv import load_dotenv
from joblib import Memory
import requests
from ollama import Client # type: ignore
from openai import OpenAI
ollama_client = Client(
    host='http://kumo01.tsc.uc3m.es:11434',
    headers={'x-some-header': 'some-value'}
)
#memory = Memory(location='cache_kumo01_tpc11', verbose=0)
memory = Memory(location='cache_final', verbose=0)

class Prompter:
    def __init__(
        self,
        model_type: str = "ollama",
        temperature: float = 0,
        top_p: float = 0.1,
        random_seed: int = 1234,
        frequency_penalty: float = 0.0,
        path_open_ai_key: str = ".env",
        ollama_host: str = "http://127.0.0.1:11434",
        llama_cpp_host: str = "http://kumo01:11435/v1/chat/completions",
        logger=None
    ):  
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        
        self.GPT_MODELS = [
            'gpt-4o-2024-08-06',
            'gpt-4o-mini-2024-07-18',
            'chatgpt-4o-latest', 
            'gpt-4-turbo',
            'gpt-4', 
            'gpt-3.5-turbo', 
            'gpt-4o-mini', 
            'gpt-4o'
        ]
        
        self.OLLAMA_MODELS = [
            'llama3:70b-instruct',
            'llama3:70b',
            'llama3.2',
            'llama3.1:8b-instruct-q8_0',
            'llama3.1:latest',
            'qwen:32b',
            'llama3.3:70b',
            'qwen2.5:7b-instruct',
            'qwen2.5:32b',
            'qwen2.5:72b-instruct'
        ]
        
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "frequency_penalty": frequency_penalty
        }    
        
        self.model_type = model_type
        self.context = None
        self.llama_cpp_host = llama_cpp_host
        
        # Determine backend based on model_type
        if model_type in self.GPT_MODELS:
            load_dotenv(path_open_ai_key)
            self.backend = "openai"
            self._logger.info(
                f"-- -- Using OpenAI API with model: {model_type}")
        elif model_type in self.OLLAMA_MODELS:
            os.environ['OLLAMA_HOST'] = ollama_host
            self.backend = "ollama"
            self._logger.info(
                f"-- -- Using OLLAMA API with host: {ollama_host}"
            )
        elif model_type == "llama_cpp":
            self.backend = "llama_cpp"
            self._logger.info(
                f"-- -- Using llama_cpp API with host: {llama_cpp_host}"
            )
        else:
            raise ValueError("Unsupported model_type specified.")

    def _load_template(self, template_path: str) -> str:
        with open(template_path, 'r') as file:
            return file.read()

    @staticmethod
    @memory.cache
    def _cached_prompt_impl(
        template: str,
        question: str,
        model_type: str,
        backend: str,
        params: tuple,
        context=None,
        use_context: bool = False,
    ) -> dict:
        """Caching setup."""
        
        #print("Cache miss: computing results...")
        if backend == "openai":
            result, logprobs = Prompter._call_openai_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),  
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template,
                question=question,
                params=dict(params), 
            )
        else:
            raise ValueError(f"-- -- Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": dict(params),
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }


    @staticmethod
    def _call_openai_api(template, question, model_type, params):
        """Handles the OpenAI API call."""
        
        if template is not None:
             messages=[
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ]
        else:
            messages=[
                {"role": "user", "content": question},
            ]
        
        open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = open_ai_client.chat.completions.create(
            model=model_type,
            messages=messages,
            stream=False,
            temperature=params["temperature"],
            max_tokens=params.get("max_tokens", 1000),
            seed=params.get("seed", 1234),
            logprobs=True,
            top_logprobs=10,
        )
        result = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return result, logprobs


    @staticmethod
    def _call_ollama_api(template, question, model_type, params, context):
        """Handles the OLLAMA API call."""
        
        if template is not None:
            response = ollama_client.generate(
                system=template,
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        else:
            response = ollama_client.generate(
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        result = response["response"]
        logprobs = None
        context = response.get("context", None)
        return result, logprobs, context

    @staticmethod
    def _call_llama_cpp_api(template, question, params, llama_cpp_host="http://kumo01:11435/v1/chat/completions"):
        """Handles the llama_cpp API call."""
        payload = {
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 100),
            "logprobs": 1,
            "n_probs": 1,
        }
        response = requests.post(llama_cpp_host, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            result = response_data["choices"][0]["message"]["content"]
            logprobs = response_data.get("completion_probabilities", [])
        else:
            raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")

        return result, logprobs

    def prompt(
        self,
        question: str,
        system_prompt_template_path: str = None,
        use_context: bool = False,
    ) -> Union[str, List[str]]:
        """Public method to execute a prompt given a system prompt template and a question."""

        # Load the system prompt template
        system_prompt_template = None
        if system_prompt_template_path is not None:
            with open(system_prompt_template_path, "r") as file:
                system_prompt_template = file.read()

        # Ensure hashable params for caching and get cached data / execute prompt
        params_tuple = tuple(sorted(self.params.items()))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        # Update context if necessary
        if use_context:
            self.context = cached_data["inputs"]["context"]
            
        return result, logprobs