import logging
from src.core.embedder import Embedder

def main():
    
    logger = logging.getLogger(__name__)
    embedder_manager = Embedder(logger=logger)
    
    text_to_embed = "This is a sample text to be embedded."
    
    # Generate embeddings
    embeddings = embedder_manager.infer_embeddings(
        embed_from=text_to_embed,
        method="bert",
    )
    
    logger.info(f"-- -- Embeddings generated successfully: {embeddings}")

if __name__ == "__main__":
    main()