import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from langchain.prompts import PromptTemplate  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
from langchain_community.llms import Ollama  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.schema import Document  # type: ignore
from langchain_core.runnables import RunnableLambda  # type: ignore


class MetadataExtractor:
    """
    Extrae metadatos en 3 bloques (adjudicación, solvencia, condiciones especiales)
    usando SOLO Ollama (LLM + embeddings). El host se toma de OLLAMA_HOST o default.
    """

    def __init__(
        self,
        ollama_llm: str = "llama3.1",
        ollama_embed_model: str = "mxbai-embed-large",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        retriever_k: int = 12,
    ):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://kumo01.tsc.uc3m.es:11434")
        self.ollama_llm = ollama_llm
        self.ollama_embed_model = ollama_embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        # LLM y embeddings de Ollama (ambos apuntan al mismo host/base_url)
        # num_ctx para abarcar mejor los fragmentos recuperados
        self.model = Ollama(
            model=self.ollama_llm,
            base_url=self.ollama_host,
            num_ctx=8192,
            temperature=0,
        )
        self.embeddings = OllamaEmbeddings(
            model=self.ollama_embed_model,
            base_url=self.ollama_host,
        )

        self.prompt = self._build_prompt_template()

    # ----------------------- Prompt -----------------------
    def _build_prompt_template(self) -> PromptTemplate:
        template = """
Eres un asistente experto en derecho administrativo, especializado en analizar y resumir documentos legales de contratación pública. 
Tu tarea consiste en **extraer textualmente** los criterios requeridos, según las instrucciones, y organizarlos correctamente.

### INSTRUCCIONES CLAVE

- Usa exclusivamente **texto literal** del documento. No generes contenido adicional.
- Si no hay información disponible para un criterio, responde exactamente:
  "No existe este criterio en este documento."
- No mezcles información entre secciones.
- La salida debe contener solo estos tres bloques, con los encabezados EXACTOS:

### Criterios de Adjudicación
[contenido literal extraído del documento, si lo hay]

### Criterios de Solvencia
[contenido literal extraído del documento, si lo hay]

### Condiciones Especiales de Ejecución
[contenido literal extraído del documento, si lo hay]

### DETALLES POR CATEGORÍA

1. **Criterios de Adjudicación**: subcriterios evaluables, puntuación asignada, baremación, ponderación, reglas de desempate.
2. **Criterios de Solvencia**: requisitos económicos, técnicos o profesionales que debe cumplir el licitador.
3. **Condiciones Especiales de Ejecución**: criterios sociales, medioambientales, éticos u otros que condicionan la ejecución del contrato.

### EJEMPLO DE RESPUESTA ESPERADA

### Criterios de Adjudicación
Los criterios serán evaluables automáticamente hasta 60 puntos, considerando el plazo de entrega, metodología y experiencia técnica.

### Criterios de Solvencia
El licitador deberá acreditar una solvencia económica mínima de 500.000€ y haber ejecutado dos contratos similares.

### Condiciones Especiales de Ejecución
Se exigirá el cumplimiento de cláusulas medioambientales relativas a la reducción de emisiones.


Contexto:
{context}

Pregunta:
Extrae y organiza el contenido en los tres bloques anteriores, respetando encabezados y reglas.
"""
        # Solo necesitamos "context" como variable; la pregunta está fija.
        return PromptTemplate(template=template, input_variables=["context"])

    # ----------------------- Helpers -----------------------
    @staticmethod
    def _format_content(text: str) -> str:
        """Si el texto es XML, lo convierte a texto plano; si no, lo devuelve tal cual."""
        try:
            t = (text or "").strip()
            if t.startswith("<"):
                c_et = ET.fromstring(t)
                return ET.tostring(c_et, method="text", encoding="utf-8").decode("utf-8")
            return t
        except Exception:
            return text or ""

    def _create_documents(self, text: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        # metadatos simples para depuración opcional
        return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

    @staticmethod
    def _clean_text(text: str) -> str:
        # Limpieza suave: conserva puntuación habitual en pliegos
        return re.sub(
            r"[^\w\s.,;:¡!¿?\-()/%€$°#«»“”\"'–—·]",
            " ",
            text or "",
        ).strip()

    @staticmethod
    def _divide_by_categories(texto: str) -> Dict[str, str]:
        """
        Recorta el resultado en tres claves basadas en encabezados (tolerante a tildes y espacios):
        '### Criterios de Adjudicación', '### Criterios de Solvencia',
        '### Condiciones Especiales de Ejecución'
        """
        headers_regex = {
            "criterios_adjudicacion": r"Criterios\s+de\s+Adjudicaci[oó]n",
            "criterios_solvencia": r"Criterios\s+de\s+Solvencia",
            "condiciones_especiales": r"Condiciones\s+Especiales\s+de\s+Ejecuci[oó]n",
        }

        resultado = {k: "" for k in headers_regex}
        patron = "|".join(headers_regex.values())

        # Partimos por los encabezados "### ..." tolerando espacios extra
        secciones = re.split(
            rf"(?=###\s+(?:{patron}))",
            texto or "",
            flags=re.IGNORECASE,
        )

        for seccion in secciones:
            for key, encabezado_regex in headers_regex.items():
                m = re.search(rf"###\s+{encabezado_regex}\s*",
                              seccion, flags=re.IGNORECASE)
                if m:
                    contenido = seccion[m.end():].strip()
                    resultado[key] = contenido
        return resultado

    @staticmethod
    def _format_docs_for_context(docs: List[Document]) -> str:
        # Junta los top-k chunks recuperados en un solo “Contexto”
        return "\n\n".join(d.page_content for d in docs)

    # ----------------------- Pipeline principal -----------------------
    def extract_metadata_from_text(self, text: str) -> Dict[str, str]:
        """
        Entrada: texto largo (string). Salida: dict con 3 claves:
        - criterios_adjudicacion
        - criterios_solvencia
        - condiciones_especiales
        """
        formatted = self._format_content(text)
        if not formatted:
            return {"error": "Texto vacío o inválido."}

        docs = self._create_documents(formatted)
        if not docs:
            return {"error": "El contenido quedó vacío después de procesar."}

        # Vector store con embeddings de Ollama
        try:
            vector_storage = FAISS.from_documents(docs, self.embeddings)
        except Exception as e:
            return {
                "error": (
                    f"No se pudo inicializar FAISS con embeddings de Ollama en {self.ollama_host}. "
                    f"¿Está el servidor activo? Detalle: {str(e)}"
                )
            }

        retriever = vector_storage.as_retriever(
            search_kwargs={"k": self.retriever_k})

        def build_query(_: str) -> str:
            return "criterios adjudicación solvencia condiciones especiales ejecución contratación pública"

        build_context = (
            RunnableLambda(build_query)          # -> query string
            | retriever                          # -> List[Document]
            | RunnableLambda(self._format_docs_for_context)  # -> str
            # -> dict para el PromptTemplate
            | RunnableLambda(lambda txt: {"context": txt})
        )

        chain = build_context | self.prompt | self.model | StrOutputParser()

        try:
            raw = chain.invoke(None)
            cleaned = self._clean_text(raw)
            return self._divide_by_categories(cleaned)
        except Exception as e:
            return {"error": f"Fallo al invocar LLM de Ollama en {self.ollama_host}: {str(e)}"}
