import dspy
import json
import logging
import random
import re
import dspy
from dspy_pipeline import SyntheticQuestionPipeline # Assumes this is in dspy_pipeline.py
import config 

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Configure DSPy Settings for OpenRouter ---
optimizer_lm = dspy.LM(
    model="openrouter/moonshotai/kimi-k2",
    api_base="https://openrouter.ai/api/v1",
    api_key=config.OPEN_ROUTER_API_KEY,
)
pipeline_lm = dspy.LM(
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=config.OPEN_ROUTER_API_KEY,
)
eval_llm = dspy.LM(
    model="openrouter/moonshotai/kimi-k2",
    api_base="https://openrouter.ai/api/v1",
    api_key=config.OPEN_ROUTER_API_KEY,
)

# Configure the default language model for the pipeline being optimized.
# The optimizer and evaluator will specify their own powerful LMs.
dspy.settings.configure(lm=pipeline_lm, rm=None)

# --- 3. Define the Evaluation Metric ---
def llm_metric(gold, pred, trace=None):
    """
    An LLM-based metric to score the quality of a generated question.
    It assesses whether the predicted question is as good or better than the gold standard.
    """
    # Prompt for the judge LLM
    assessment_prompt = (
        "Assess the quality of the 'Predicted Question' against the 'Gold Standard Question' "
        "based on the provided 'Chunk Text'. Consider clarity, specificity, relevance and how much it fits the character "
        "to the text. Answer with a single integer score from 1 to 5, where 5 is best."
    )

    # Use a dspy.Predict call with the powerful evaluation model
    with dspy.context(lm=eval_llm):
        response = dspy.Predict(
            'chunk_text, gold_question, predicted_question, assessment_prompt -> assessment_score_text'
        )(
            chunk_text=gold.chunk_text,
            gold_question=gold.gold_question,
            predicted_question=pred.query,
            assessment_prompt=assessment_prompt
        )

    # Robustly parse the score from the response
    try:
        # Find the first number in the string
        score_match = re.search(r'\d+', response.assessment_score_text)
        if score_match:
            score = float(score_match.group())
            logging.debug(f"Metric Score: {score} for gold: '{gold.gold_question}' | pred: '{pred.query}'")
            # Consider scores of 4 or 5 as a success for the optimizer
            return score >= 4.0
        else:
            logging.warning(f"Could not parse score from response: {response.assessment_score_text}")
            return False
    except Exception as e:
        logging.error(f"Error parsing metric score: {e}")
        return False

# --- 4. Main Optimization Logic ---
def main():
    logging.info("🚀 Starting DSPy optimization process...")

    # Load the gold standard dataset
    logging.info("Loading gold_dataset.json...")
    with open('gold_dataset.json', 'r') as f:
        gold_data = json.load(f)

    # Convert to dspy.Example objects
    # The field names ('chunk_text', 'persona_description', 'gold_question') must
    # match the inputs and outputs of your pipeline.
    examples = [
        dspy.Example(
            chunk_text=item['chunk_text'],
            persona_description=item['persona_description'],
            gold_question=item['gold_question'] # Ground truth
        ).with_inputs('chunk_text', 'persona_description')
        for item in gold_data
    ]
    logging.info(f"Loaded {len(examples)} examples.")
    

    # Split data into training and development sets (70/30 split)
    random.shuffle(examples)
    split_point = int(len(examples) * 0.6)
    trainset = examples[:split_point]
    devset = examples[split_point:]
    logging.info(f"Data split: {len(trainset)} training examples, {len(devset)} development examples.")


    
    # Instantiate the optimizer (teleprompter)
    # This optimizer will generate prompts and few-shot examples.
    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=llm_metric,
        max_bootstrapped_demos=2, # For each module, generate 2 few-shot examples
        num_candidate_programs=5, # Explore 5 different sets of prompts/examples
        teacher_settings=dict(lm=optimizer_lm)
    )

    # Instantiate the pipeline module that we want to optimize
    student_pipeline = SyntheticQuestionPipeline()

    evaluate = dspy.Evaluate(devset=devset, metric=llm_metric, num_threads=4, display_progress=True, display_table=5)
    # print(evaluate(student_pipeline))

    # Run the compilation!
    logging.info("Starting compilation... This may take a significant amount of time.")
    optimized_pipeline = optimizer.compile(
        student=student_pipeline,
        trainset=trainset,
        valset=devset
    )
    logging.info("✅ Compilation complete!")

    print(evaluate(optimized_pipeline))
    # Save the optimized pipeline's state (prompts and examples) to a file
    output_path = "optimized_pipeline.json"
    optimized_pipeline.save(output_path)
    logging.info(f"Optimized pipeline saved to {output_path}")
    
    # You can inspect the optimized prompts
    logging.info("\n--- Optimized Prompts ---")
    logging.info("Analyzer Prompt:\n" + optimized_pipeline.analyzer.demos[0].prompt)
    logging.info("Generator Prompt:\n" + optimized_pipeline.generator.demos[0].prompt)


if __name__ == "__main__":
    main()
