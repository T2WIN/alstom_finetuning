import dspy
from dspy_signatures import AnalyzeQueryRelevance, GenerateFinalQuery

class SyntheticQuestionPipeline(dspy.Module):
    """
    A DSPy module that generates a synthetic question from a document chunk
    and a given persona.
    """
    def __init__(self):
        super().__init__()
        # These are the "optimizable" components of your pipeline.
        # dspy.Predict uses an LLM to execute the logic defined in a signature.
        self.analyzer = dspy.Predict(AnalyzeQueryRelevance)
        self.generator = dspy.Predict(GenerateFinalQuery)

    def forward(self, chunk_text, persona_description):
        """
        Defines the forward pass of the module.

        Args:
            chunk_text (str): The text content of the document chunk.
            persona_description (str): A description of the user persona.

        Returns:
            dspy.Prediction: An object containing the final generated query.
        """
        # 1. First LLM call to determine query type and format based on the signature.
        analysis = self.analyzer(
            chunk_text=chunk_text,
            persona_description=persona_description
        )
        
        # 2. Second LLM call to generate the final query, using the output from the first call.
        final_query = self.generator(
            chunk_text=chunk_text,
            persona_description=persona_description,
            query_type=analysis.query_type,
            query_format=analysis.query_format
        )

        # The final output field ('query' here) must match the key
        # used for the ground truth in your dataset (e.g., 'gold_question').
        # We rename it in the return statement for clarity during optimization.
        return dspy.Prediction(query=final_query.query)