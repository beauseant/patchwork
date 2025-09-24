import re
import unicodedata

def compare(a, b):
    '''Fuzzy matching of strings to compare headers/footers in neighboring pages'''

    count = 0
    a = re.sub('\d', '@', a)
    b = re.sub('\d', '@', b)
    for x, y in zip(a, b):
        if x == y:
            count += 1
    return count / max(len(a), len(b))


def completion_to_prompt(completion):
    """
    Transform a string into input zephyr-specific input
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    """
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


def messages_to_prompt(messages):
    """
    Transform a list of chat messages into zephyr-specific input
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    """
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


def clean_string (texto):
   texto_normalizado = unicodedata.normalize('NFC', texto)
   patron = r'[^\x00-\x7fáéíóúÁÉÍÓÚñÑüÜ\u00BF\u00A1]'
   resultado = re.sub(patron, '', texto_normalizado)
   # Elimina los caracteres "#", las barras así "\", los "{}", "&", "^", "_", "-"
   resultado = re.sub(r'[¡!#\\/{}&^_-]', ' ', resultado)  
    # Reemplaza las secuencias de puntos por un solo punto
   resultado = re.sub(r'\.{2,}', '.', resultado) 
   # Esta expresión busca uno o más grupos de espacios opcionales, un punto, y espacios opcionales.
   resultado = re.sub(r'(?:\s*\.\s*)+', ' ', resultado).strip()
   return resultado


