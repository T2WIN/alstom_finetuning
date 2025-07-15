import dspy

class AnalyzeQueryRelevance(dspy.Signature):
    """Given a text chunk and a persona, determine the most relevant query type and format."""
    
    # Input Fields
    chunk_text = dspy.InputField(desc="The document chunk from which to generate a question.")
    persona_description = dspy.InputField(desc="The persona of the user who will be asking the question.")
    
    # Output Fields
    query_type = dspy.OutputField(desc="A specific type of query, e.g., 'comparison', 'clarification', 'how-to'.")
    query_format = dspy.OutputField(desc="The desired format of the final question, e.g., 'single-question', 'bullet-points'.")

class GenerateFinalQuery(dspy.Signature):
    """Given a chunk, persona, query type, and format, generate the final question."""
    
    # Input Fields
    chunk_text = dspy.InputField(desc="The document chunk.")
    persona_description = dspy.InputField(desc="The persona of the user.")
    query_type = dspy.InputField(desc="The determined query type.")
    query_format = dspy.InputField(desc="The determined query format.")
    
    # Output Field
    query = dspy.OutputField(desc="The final, well-formed question.")
