# test_dspy_pipeline.py

import pytest
from unittest.mock import MagicMock
import dspy

# Import the pipeline to be tested
from dspy_pipeline import SyntheticQuestionPipeline
from dspy_signatures import AnalyzeQueryRelevance, GenerateFinalQuery


@pytest.fixture
def configured_pipeline(mocker):
    """
    Provides an instance of SyntheticQuestionPipeline with its internal
    dspy.Predict modules (analyzer and generator) replaced by mocks.
    This version patches the dspy.Predict class before instantiation.
    """
    # Create mock instances that will become pipeline.analyzer and pipeline.generator
    # We remove `spec=dspy.Predict` to prevent mock's overly strict signature validation.
    mock_analyzer = MagicMock()
    mock_generator = MagicMock()

    # Configure what these mocks return when they are called (i.e., when the forward pass is run)
    mock_analyzer.return_value = dspy.Prediction(
        query_type='mocked_type',
        query_format='mocked_format'
    )
    mock_generator.return_value = dspy.Prediction(
        query='This is the final mocked query.'
    )

    # Patch the dspy.Predict class within the dspy_pipeline module.
    # When dspy.Predict is called in SyntheticQuestionPipeline.__init__,
    # it will return our mock instances in sequence.
    patched_predict = mocker.patch(
        'dspy_pipeline.dspy.Predict', 
        side_effect=[mock_analyzer, mock_generator]
    )
    
    # Now, instantiate the pipeline. Its __init__ will use the patched dspy.Predict
    pipeline = SyntheticQuestionPipeline()
    
    # The pipeline instance's analyzer and generator attributes are now our mocks.
    # We can run the test assertions directly on them.
    return pipeline, patched_predict


def test_pipeline_forward_pass(configured_pipeline):
    """
    Tests the forward pass of the SyntheticQuestionPipeline.
    Ensures that the internal modules are called in the correct order
    with the correct data, and that the final output is structured correctly.
    """
    # Arrange
    pipeline, patched_predict_class = configured_pipeline
    chunk_text = "This is the input document chunk."
    persona_description = "This is the user persona."

    # Act
    # Call the forward method, which will use the mocked sub-modules
    result = pipeline.forward(
        chunk_text=chunk_text,
        persona_description=persona_description
    )

    # Assert
    
    # 0. Verify that dspy.Predict was instantiated correctly in the pipeline's __init__
    # The first call should have been with the AnalyzeQueryRelevance signature
    # The second call should have been with the GenerateFinalQuery signature
    patched_predict_class.assert_any_call(AnalyzeQueryRelevance)
    patched_predict_class.assert_any_call(GenerateFinalQuery)
    assert patched_predict_class.call_count == 2
    
    # 1. Check the final output from the forward pass
    assert isinstance(result, dspy.Prediction)
    assert result.query == 'This is the final mocked query.'

    # 2. Verify the 'analyzer' mock was called correctly during the forward pass
    pipeline.analyzer.assert_called_once_with(
        chunk_text=chunk_text,
        persona_description=persona_description
    )

    # 3. Verify the 'generator' mock was called correctly, using the output from the analyzer
    pipeline.generator.assert_called_once_with(
        chunk_text=chunk_text,
        persona_description=persona_description,
        query_type='mocked_type',      # This comes from the mocked analyzer
        query_format='mocked_format'   # This also comes from the mocked analyzer
    )
