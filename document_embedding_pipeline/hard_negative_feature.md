# Goal
You are a professionnal Python programmer.
You have to build a pipeline component :
class HardNegativeProcessor(BaseProcessor):

    def __init__(self, embedding_model_path : Path):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        ...

    async def process(file_path : Path):
        ...


# How
### Input
This component should take a csv with two columns, one called query and the second called answer.

### Filtering
It should remove query answer pairs that have a semantic similarity below a threshold (that the user can set)

### Hard negative mining
For each pair, it should find the hard negatives.
The original answer associated with the query is called the positive.
For each query, it should select the top K closest answers from the whole dataset based on semantic similarity (do not consider the positive).
From that top-K, remove (query, answer) pairs where similarity is too close to the similarity (too close is defined via a user set ratio of the similarity between query and positive.)
You then obtain a set of triplets (query, positive, hard negative).

# Output
Store all triplets for the input file in one csv file.