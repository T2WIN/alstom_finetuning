# annotate_chunks.py
import json
import logging
import os
import instructor
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List

# Import project-specific modules
import config
import prompts

# --- Logging and Pydantic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SuggestedQuestions(BaseModel):
    """Pydantic model for parsing LLM-suggested questions."""
    questions: List[str] = Field(..., description="A list of 2 discting questions and 2 distinct keyword queries")

class CharacterSelector:
    """A class to handle the selection of the most relevant character persona."""
    def __init__(self):
        logging.info("Initializing CharacterSelector...")
        self.characters = [
            prompts.INDUSTRIALIZATION_ENGINEER_DESCRIPTION,
            prompts.MAINTENANCE_LOGISTICS_TECH_DESCRIPTION,
            prompts.ROLLING_STOCK_DESIGN_ENGINEER_DESCRIPTION,
            prompts.MAINTENANCE_DEPOT_TECHNICIAN,
            prompts.TRAIN_SAFETY_ENGINEER_DESCRIPTION,
        ]
        model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')
        prompt_embeddings = model.encode(self.characters, show_progress_bar=False)
        self.prompt_embeddings_norm = prompt_embeddings / np.linalg.norm(
            prompt_embeddings, axis=1, keepdims=True
        )
        self.model = model
        logging.info("CharacterSelector initialized.")

    def choose_appropriate_character(self, text_chunk: str) -> str:
        """Finds the most similar character persona for a given text chunk."""
        query_embedding = self.model.encode([text_chunk], show_progress_bar=False)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(self.prompt_embeddings_norm, query_embedding_norm.T).flatten()
        most_similar_prompt_index = np.argmax(similarities)
        return self.characters[most_similar_prompt_index]

def get_llm_suggestions(client: instructor.Instructor, persona: str, chunk: str) -> List[str]:
    """Calls the LLM to get question suggestions."""
    try:
        response: SuggestedQuestions = client.chat.completions.create(
            model="magistral-medium-2506",
            response_model=SuggestedQuestions,
            messages=[
                {"role": "user", "content": prompts.ANNOTATION_SUGGESTION_PROMPT.format(
                    persona_description=persona,
                    chunk_text=chunk
                )}
            ],
            max_retries=2,
        )
        return response.questions
    except Exception as e:
        logging.error(f"Failed to get LLM suggestions: {e}")
        return []

def annotate():
    """Main function to run the human-in-the-loop annotation process."""
    # --- Setup ---
    # NOTE: You must have one of your API keys set as an environment variable.
    # For example, `export OPENAI_API_KEY='your-key'` or `export MISTRAL_API_KEY='your-key'`.
    # The instructor library will automatically pick it up.
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY or OPENAI_API_KEY must be set in the environment.")

    # We use a standard OpenAI client, which can be pointed to any service
    # that has an OpenAI-compatible API endpoint.
    client = instructor.patch(OpenAI(base_url="https://api.mistral.ai/v1/", api_key=api_key)) # If using Mistral, you might need to set base_url
    
    char_selector = CharacterSelector()
    
    # --- File Paths ---
    raw_chunks_file = 'dspy_optimizer/data/raw/raw_chunks.jsonl'
    gold_dataset_file = 'gold_dataset.json'

    # --- Load Data ---
    if not os.path.exists(raw_chunks_file):
        logging.error(f"'{raw_chunks_file}' not found. Please ensure the file from Step 1.2 is present.")
        return

    with open(raw_chunks_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    gold_dataset = []
    if os.path.exists(gold_dataset_file):
        logging.info(f"Resuming from existing '{gold_dataset_file}'.")
        with open(gold_dataset_file, 'r', encoding='utf-8') as f:
            gold_dataset = json.load(f)

    # --- Annotation Loop ---
    start_index = len(gold_dataset)
    for i, chunk_data in enumerate(chunks[start_index:]):
        chunk_text = chunk_data.get("text")
        if not chunk_text:
            continue

        print("\n" + "="*80)
        print(f"Processing chunk {start_index + i + 1}/{len(chunks)}...")
        print("="*80)

        # 1. Persona Selection
        persona = char_selector.choose_appropriate_character(chunk_text)

        # 2. LLM Suggestion
        suggestions = get_llm_suggestions(client, persona, chunk_text)

        # 3. Human-in-the-Loop
        print("\n--- CHUNK TEXT ---")
        print(chunk_text)
        print("\n--- SELECTED PERSONA ---")
        print(persona)
        print("\n--- LLM SUGGESTED QUESTIONS ---")

        if not suggestions:
            print("Could not generate suggestions for this chunk.")
        else:
            for j, q in enumerate(suggestions):
                print(f"  [{j+1}] {q}")

        while True:
            print("\n--- ACTION ---")
            user_input = input(
                "Enter number to select, (w)rite your own, (s)kip, or (q)uit and save: "
            ).strip().lower()

            if user_input == 'q':
                break
            elif user_input == 's':
                gold_question = None
                break
            elif user_input == 'w':
                gold_question = input("Enter your question: ").strip()
                if not gold_question:
                    print("Question cannot be empty.")
                    continue
                break
            elif user_input.isdigit() and suggestions and 1 <= int(user_input) <= len(suggestions):
                gold_question = suggestions[int(user_input) - 1]
                break
            else:
                print("Invalid input. Please try again.")

        if user_input == 'q':
            break

        if gold_question:
            gold_dataset.append({
                "chunk_text": chunk_text,
                "persona_description": persona,
                "gold_question": gold_question,
            })
            logging.info(f"Annotation added. Total: {len(gold_dataset)}")

    # --- Save Final Dataset ---
    with open(gold_dataset_file, 'w', encoding='utf-8') as f:
        json.dump(gold_dataset, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(gold_dataset)} entries to '{gold_dataset_file}'. Annotation complete.")


if __name__ == "__main__":
    annotate()