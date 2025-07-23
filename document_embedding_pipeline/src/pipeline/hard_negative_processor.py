import asyncio
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os

# Assuming BaseProcessor is defined elsewhere, e.g.:
class BaseProcessor:
    def __init__(self, *args, **kwargs):
        pass
    async def process(self, input_file_path: Path, output_file_path: Path):
        raise NotImplementedError

class HardNegativeProcessor(BaseProcessor):
    """
    A processor to filter query-answer pairs and mine for hard negatives,
    with added support for queries having multiple positive answers.
    """

    def __init__(self, 
                 embedding_model_path: str, 
                 similarity_threshold: float = 0.5, 
                 top_k: int = 10, 
                 hardness_ratio: float = 0.95):
        """
        Initializes the HardNegativeProcessor.

        Args:
            embedding_model_path (str): The path or name of the SentenceTransformer model.
            similarity_threshold (float): The minimum cosine similarity for a query-answer pair to be considered valid.
            top_k (int): The number of top similar answers to consider for hard negative mining.
            hardness_ratio (float): A ratio to filter out negatives that are too similar to the positive. 
                                  A candidate negative is kept if its similarity to the query is less than
                                  (positive_similarity * hardness_ratio).
        """
        print("Initializing SentenceTransformer model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.hardness_ratio = hardness_ratio
        print("Initialization complete.")

    async def process(self, input_file_path: Path, output_file_path: Path):
        """
        Processes the input CSV file to generate hard negative triplets.

        Args:
            input_file_path (Path): Path to the input CSV file with 'query' and 'answer' columns.
            output_file_path (Path): Path to save the resulting CSV file with triplets.
        """
        print(f"\n--- Starting processing for {input_file_path.name} ---")

        # 1. Load Data
        if input_file_path.suffix == ".csv":
            try:
                df = pd.read_csv(input_file_path)
                if 'query' not in df.columns or 'answer' not in df.columns:
                    raise ValueError("Input CSV must contain 'query' and 'answer' columns.")
                df.dropna(subset=['query', 'answer'], inplace=True)
                df.reset_index(drop=True, inplace=True) # Ensure original indices are 0-based
                print(f"Loaded {len(df)} records from {input_file_path}.")
            except Exception as e:
                print(f"An error occurred while reading the CSV file: {e}")
                return
        elif input_file_path.suffix == ".json":
            try:
                df = pd.read_json(input_file_path, orient="records")
                df["query"] = df["title"]
                df["answer"] = df["abstract"]
                df.dropna(subset=['query', 'answer'], inplace=True)
                df.reset_index(drop=True, inplace=True) # Ensure original indices are 0-based
                print(f"Loaded {len(df)} records from {input_file_path}.")
            except Exception as e:
                print(f"An error occurred while reading the CSV file: {e}")
                return

        # 2. Compute Embeddings
        print("Computing embeddings for all queries and answers...")
        queries = df['query'].tolist()
        answers = df['answer'].tolist()
        query_embeddings = self.embedding_model.encode(queries, convert_to_tensor=True, device=self.device, show_progress_bar=True)
        answer_embeddings = self.embedding_model.encode(answers, convert_to_tensor=True, device=self.device, show_progress_bar=True)
        print("Embeddings computed.")

        # 3. Initial Similarity Filtering
        print(f"Filtering pairs with similarity below threshold: {self.similarity_threshold}")
        positive_similarities = util.cos_sim(query_embeddings, answer_embeddings).diag()
        df['positive_similarity'] = positive_similarities.cpu().numpy()
        
        filtered_df = df[df['positive_similarity'] >= self.similarity_threshold].copy()
        # Keep the original index from `df` to correctly map embeddings and answers
        filtered_df.reset_index(inplace=True) 
        
        print(f"Found {len(filtered_df)} pairs above the similarity threshold.")

        if len(filtered_df) == 0:
            print("No pairs met the similarity threshold. No output will be generated.")
            return

        # 4. Hard Negative Mining
        print(f"Starting hard negative mining (Top K={self.top_k}, Hardness Ratio={self.hardness_ratio})...")
        
        # *** NEW: Create a map from each query to the indices of ALL its positive answers ***
        # This is the key change to handle multiple positives for a single query.
        query_to_positives_indices = df.groupby('query').apply(lambda x: x.index.tolist()).to_dict()

        # Calculate all-to-all similarity between queries from the original df and all answers
        all_sims = util.cos_sim(query_embeddings, answer_embeddings)

        hard_negative_triplets = []
        
        # Iterate over the filtered pairs
        for _, row in filtered_df.iterrows():
            original_index = row['index']  # This is the original index in the unfiltered df
            query = row['query']
            positive_answer = row['answer']
            positive_similarity = row['positive_similarity']

            # Get similarities of the current query to all answers in the dataset
            query_to_all_answers_sim = all_sims[original_index].clone() # Use .clone() to avoid modifying the original tensor
            
            # *** MODIFIED: Exclude ALL positives for the current query ***
            # Get the list of indices for all answers associated with the current query
            indices_to_exclude = query_to_positives_indices.get(query, [])
            
            # Set their similarity scores to a very low value to ensure they are never picked as negatives
            if indices_to_exclude:
                query_to_all_answers_sim[indices_to_exclude] = -1.0

            # Get top K most similar answers (hard negative candidates)
            # Ensure k is not larger than the number of available candidates
            k = min(self.top_k, len(query_to_all_answers_sim) - len(indices_to_exclude))
            if k <= 0: continue # Skip if no candidates are left to choose from
            
            top_k_results = torch.topk(query_to_all_answers_sim, k=k)
            
            # Filter the top K based on the hardness ratio
            for score, neg_idx in zip(top_k_results.values, top_k_results.indices):
                hard_negative_similarity = score.item()
                
                # Apply the hardness ratio filter
                if hard_negative_similarity < (positive_similarity * self.hardness_ratio):
                    hard_negative_answer = df['answer'].iloc[neg_idx.item()]
                    hard_negative_triplets.append({
                        'query': query,
                        'positive': positive_answer,
                        'hard_negative': hard_negative_answer
                    })

        print(f"Generated {len(hard_negative_triplets)} hard negative triplets.")

        # 5. Store Output
        if hard_negative_triplets:
            output_df = pd.DataFrame(hard_negative_triplets)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_file_path, index=False)
            print(f"Successfully saved triplets to {output_file_path}")
        else:
            print("No hard negative triplets were generated.")
            
        print("--- Processing complete ---")


async def main():
    """Main function to demonstrate the HardNegativeProcessor."""
    model_name = '/home/grand/Finetuning/models/qwen3-embed-0.6b' 
    input_filename = "/home/grand/alstom_finetuning/data/query_answer_pairs/papers_dataset.json"
    output_filename = "hard_negatives_output.csv"

    # --- Instantiate and run the processor ---
    processor = HardNegativeProcessor(
        embedding_model_path=model_name,
        similarity_threshold=0,
        top_k=10,
        hardness_ratio=0.95
    )

    input_path = Path(input_filename)
    output_path = Path(output_filename)

    await processor.process(input_path, output_path)

    # --- Verify the output ---
    if output_path.exists():
        print(f"\n--- Contents of '{output_filename}' ---")
        output_df = pd.read_csv(output_path)
        print(output_df.head())
        print("...")
        print(f"Total triplets found: {len(output_df)}")
    else:
        print(f"\nOutput file '{output_filename}' was not created.")


if __name__ == "__main__":
    # To run the async main function
    asyncio.run(main())
