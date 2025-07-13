import torch
import logging
from collections import Counter
from llama_index.core.schema import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from typing import List, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EduFilter(TransformComponent):
    """
    A LlamaIndex transform component that classifies documents for educational value
    using the HuggingFaceTB/fineweb-edu-classifier model.

    This component logs aggregated classification results and is compatible with
    Pydantic v2 used in recent LlamaIndex versions.
    """
    # Declare fields for Pydantic validation
    model_name: str
    device: str
    tokenizer: Any
    model: Any

    def __init__(self, model_name: str = "HuggingFaceTB/fineweb-edu-classifier", device: str = "auto"):
        """
        Initializes the classifier.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            device (str): The device to run the model on ('auto', 'cpu', 'cuda').
        """
        # Resolve device
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model '{model_name}' on device: {resolved_device}")

        # Load model and tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(resolved_device)
        model.eval()  # Set the model to evaluation mode

        # Call super().__init__ to properly initialize the Pydantic model
        super().__init__(
            model_name=model_name,
            device=resolved_device,
            tokenizer=tokenizer,
            model=model
        )

    @classmethod
    def class_name(cls) -> str:
        """Get the class name, used by LlamaIndex for serialization."""
        return "EduFilter"

    def __call__(self, nodes: List[Document], **kwargs) -> List[Document]:
        """
        Processes a list of documents, filtering for high-quality educational content
        and logging aggregated scores.

        Args:
            nodes (List[Document]): A list of LlamaIndex Document objects.

        Returns:
            List[Document]: A filtered list of documents with added metadata.
        """
        if not nodes:
            return []

        high_quality_nodes = []
        score_counts = Counter()

        for node in nodes:
            text = node.get_content()
            
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=512).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Calculate scores
            logits = outputs.logits.squeeze(-1)
            score = logits.item()
            int_score = int(round(max(0, min(score, 5))))

            # Update the count for the calculated integer score
            score_counts[int_score] += 1

            # Filter for high-quality nodes and add metadata
            if int_score >= 2:
                node.metadata["fineweb_edu_score"] = score
                node.metadata["fineweb_edu_int_score"] = int_score
                high_quality_nodes.append(node)
        
        # Log the aggregated counts for all processed nodes
        logger.info(f"Processed {len(nodes)} nodes. Found {len(high_quality_nodes)} high-quality nodes.")
        logger.info(f"Aggregated score distribution (0-5): {dict(score_counts)}")

        return high_quality_nodes