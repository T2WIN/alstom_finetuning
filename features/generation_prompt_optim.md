This plan is structured in five major phases, designed to be executed sequentially.

-----

### **Phase 1: Project Setup and Gold Standard Dataset Creation**

The first step is to prepare your environment and, most importantly, create the high-quality dataset that will be used to teach and validate your DSPy pipeline. Without good data, the optimization process is meaningless.

**Step 1.1: Environment and Project Structure**

  * **Create a New Directory:** Set up a new project directory (e.g., `dspy_optimizer`) to keep the optimization code separate from your main application logic.
  * **Install Dependencies:** Your `requirements.txt` or `pyproject.toml` should include `dspy-ai`, `qdrant-client`, `sentence-transformers`, `numpy`, and any LLM provider libraries you use (e.g., `openai`, `anthropic`).
  * **Configuration:** Create a `config.py` file to store paths, model names, API keys, and Qdrant settings. This centralizes configuration for both data creation and optimization scripts.

**Step 1.2: Script for Random Chunk Retrieval**

  * **Objective:** To get a random, unbiased sample of \~150-200 document chunks from your Qdrant database. You'll aim for a final dataset of 100, so starting with more allows you to discard irrelevant or low-quality chunks. Remember that the database isn't on a server but stored as a folder.
  * **Action:**
    1.  Create a Python script named `fetch_chunks.py`.
    2.  In this script, connect to your local Qdrant instance.
    3.  Since Qdrant doesn't have a built-in "random record" function, implement a two-step process:
          * First, scroll through all points using the `.scroll()` method with `limit=1` and `with_payload=False` to efficiently retrieve all unique chunk IDs.
          * Use Python's `random.sample()` function to select 150-200 IDs from this list.
    4.  Retrieve the full payload (the document chunk text) for these randomly selected IDs.
    5.  Save the output to a `raw_chunks.jsonl` file, where each line is a JSON object like: `{"chunk_id": "...", "chunk_text": "..."}`.
  * **Test:** After running the script, manually inspect `raw_chunks.jsonl` to ensure it contains the correct number of diverse chunks.

**Step 1.3: Script for LLM-Assisted Annotation**

  * **Objective:** To create the "gold standard" question for each of your 100 chosen chunks. This involves generating suggestions with an LLM and having a human make the final choice.
  * **Action:**
    1.  Create a Python script named `annotate_chunks.py`.
    2.  This script will read from `raw_chunks.jsonl`.
    3.  For each chunk, it will perform the following steps in a loop:
          * **Persona Selection:** Reuse your existing `choose_appropriate_character` function to determine the most relevant persona for the chunk. This is a critical step, as the gold question must match the persona.
          * **LLM Suggestion:** Use a powerful LLM (e.g., GPT-4o, Claude 3 Opus) with a carefully crafted prompt. The prompt should ask the LLM to generate 3-5 *excellent and distinct* questions for the given chunk from the perspective of the chosen persona.
          * **Human-in-the-Loop:** Display the following in the command line:
              * The chunk text.
              * The selected persona.
              * The LLM's suggested questions (numbered).
              * A prompt for you to choose the best one, write your own, or skip the chunk.
    4.  Save your final, curated annotations to a `gold_dataset.json` file. Each entry should be a JSON object: `{"chunk_text": "...", "persona_description": "...", "gold_question": "..."}`.
  * **Test:** Write a small utility script to load `gold_dataset.json` and print 5 random entries. Verify that the persona, chunk, and question are all high-quality and logically connected.

-----

### **Phase 2: Refactoring the Pipeline with DSPy Signatures and Modules**

Now, you will translate your existing Python logic into the structured components that DSPy uses: `Signatures` and `Modules`.

**Step 2.1: Create DSPy Signatures**

  * **Objective:** To define the input/output contracts for your LLM calls.
  * **Action:** In a new file, `dspy_signatures.py`, define the following classes:
      * **`AnalyzeQueryRelevance(dspy.Signature)`**: This captures your second component.
          * `"""Given a text chunk and a persona, determine the most relevant query type and format."""`
          * **Input Fields:** `chunk_text = dspy.InputField()`, `persona_description = dspy.InputField()`
          * **Output Fields:** `query_type = dspy.OutputField()`, `query_format = dspy.OutputField()`
      * **`GenerateFinalQuery(dspy.Signature)`**: This captures your third component.
          * `"""Given a chunk, persona, query type, and format, generate the final question."""`
          * **Input Fields:** `chunk_text = dspy.InputField()`, `persona_description = dspy.InputField()`, `query_type = dspy.InputField()`, `query_format = dspy.InputField()`
          * **Output Field:** `query = dspy.OutputField()`
  * **Test:** In a temporary script, instantiate these signatures and print them. This will show you the raw prompt templates that DSPy generates, allowing you to check if they make sense.

**Step 2.2: Build the DSPy Pipeline Module**

  * **Objective:** To chain the signatures together into a single, cohesive pipeline.
  * **Action:** In a new file, `dspy_pipeline.py`, create your main module:
    ```python
    import dspy
    from dspy_signatures import AnalyzeQueryRelevance, GenerateFinalQuery

    class SyntheticQuestionPipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            # These are the "optimizable" components of your pipeline
            self.analyzer = dspy.Predict(AnalyzeQueryRelevance)
            self.generator = dspy.Predict(GenerateFinalQuery)

        def forward(self, chunk_text, persona_description):
            # 1. First LLM call to determine query type/format
            analysis = self.analyzer(chunk_text=chunk_text, persona_description=persona_description)
            
            # 2. Second LLM call to generate the final query
            final_query = self.generator(
                chunk_text=chunk_text,
                persona_description=persona_description,
                query_type=analysis.query_type,
                query_format=analysis.query_format
            )

            # The final output field must match the ground truth in your dataset
            return dspy.Prediction(query=final_query.query)
    ```
  * **Note:** The persona selection (your first component) happens *before* this module is called. The `forward` method expects the `persona_description` to already be determined. This correctly keeps your semantic search logic outside the LLM optimization loop.

-----

### **Phase 3: Integrating Your Asynchronous LLM Dispatcher**

This is a crucial step to ensure you can leverage your sophisticated, rate-limited, round-robin system within the DSPy framework.

**Step 3.1: Create a Custom DSPy LM**

  * **Objective:** To wrap your `LLMDispatcher` in a class that DSPy can use as its language model.
  * **Action:** Create a new file, `custom_lm.py`. In it, define a class that inherits from `dspy.LM`.
    ```python
    # custom_lm.py
    import dspy
    import asyncio
    from your_code.llm_services import LLMDispatcher # Import your existing class

    class DispatcherLM(dspy.LM):
        def __init__(self, dispatcher: LLMDispatcher):
            self.dispatcher = dispatcher
            # You might need to implement basic_request and __call__
            # The goal is to make dspy.Predict use your dispatcher's logic
            # to make an LLM call.
            
        # DSPy's main entry point for requests.
        def __call__(self, prompt, **kwargs):
            # This is a simplified example. You'll need to adapt it.
            # DSPy v3 will have better async support. For now, you might need
            # to run a new asyncio event loop for each call if not in an async context.
            return asyncio.run(self.arequest(prompt, **kwargs))

        async def arequest(self, prompt, **kwargs):
            # Your logic to get a client from the dispatcher
            client_config = await self.dispatcher.get_available_client()
            
            try:
                # Your logic to make the actual API call using the client
                # You'll need to map DSPy's kwargs to your client's expected format
                response = await client_config.client.chat.completions.create(...)
                
                # Release client on success
                await self.dispatcher.release_client_on_success(client_config)
                
                # Return the response in the format DSPy expects
                # This is typically a list of strings.
                return [choice.message.content for choice in response.choices]

            except Exception as e:
                # Release client on failure
                await self.dispatcher.release_client_on_failure(client_config)
                raise e # Re-raise the exception for DSPy to handle
    ```
  * **Test:** Write a small script to instantiate `DispatcherLM` with your `LLMDispatcher`. Attempt a simple completion call (e.g., `lm("What is 2+2?")`) to ensure it successfully acquires a client, gets a response, and releases the client.

-----

### **Phase 4: Optimization (Compilation)**

With the data, pipeline, and custom LLM ready, you can now perform the optimization.

**Step 4.1: Create the Optimization Script**

  * **Objective:** To use a DSPy `teleprompter` to find the best prompts and few-shot examples for your pipeline.
  * **Action:** Create a script named `optimize_pipeline.py`.
    1.  **Load Data:** Load `gold_dataset.json` and convert each entry into a `dspy.Example` object, making sure to mark the inputs and the ground truth `gold_question`.
    2.  **Split Data:** Split your examples into a training set (`trainset`) and a development set (`devset`) (e.g., 70/30 split).
    3.  **Configure DSPy:**
          * Instantiate your `LLMDispatcher`.
          * Instantiate your custom LM: `dspy_lm = DispatcherLM(dispatcher)`.
          * Instantiate a powerful LLM for the optimizer itself (this should be a standard `dspy.OpenAI` or similar, as the optimization process itself needs a teacher LLM): `optimizer_lm = dspy.OpenAI(model='gpt-4o', max_tokens=4000)`.
          * Configure DSPy: `dspy.settings.configure(lm=dspy_lm, rm=None)`.
    4.  **Define Evaluation Metric:** Create a metric that an LLM will use to score the generated questions against the gold questions.
        ```python
        eval_llm = dspy.OpenAI(model='gpt-4o') # Use a powerful model for judging

        def llm_metric(gold, pred, trace=None):
            # This prompt asks the LLM to act as a judge
            assessment_prompt = f"Is the Predicted Question as good or better than the Gold Standard Question, based on the provided text? Assess on clarity, specificity, and relevance. Answer with a score from 1 to 5."
            
            with dspy.context(lm=eval_llm):
                response = dspy.Predict('gold_question, predicted_question -> assessment')(
                    gold_question=gold.gold_question, 
                    predicted_question=pred.query
                )
            
            # A simple check to parse the score
            score = float(response.assessment.split()[-1].strip('.')) 
            return score >= 4.0 # Consider scores of 4 or 5 as success
        ```
    5.  **Run Compilation:**
          * Instantiate an optimizer: `optimizer = dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=llm_metric, max_bootstrapped_demos=2, num_candidate_programs=5)`.
          * Run the compile step: `optimized_pipeline = optimizer.compile(SyntheticQuestionPipeline(), trainset=trainset, evalset=devset)`.
    6.  **Save the Result:** Save the optimized prompt templates and few-shot examples: `optimized_pipeline.save("optimized_pipeline.json")`.
  * **Test:** After the script finishes, check that `optimized_pipeline.json` has been created. Open it to see the optimized prompts and the few-shot examples that DSPy has generated.

-----

### **Phase 5: Evaluation and Integration**

Finally, verify the performance of the new pipeline and integrate it back into your workflow.

**Step 5.1: Evaluate Performance**

  * **Objective:** To quantitatively and qualitatively compare the original pipeline to the optimized one.
  * **Action:** Create a script named `evaluate_pipeline.py`.
    1.  Load your `devset` (or a separate `testset`).
    2.  Instantiate the un-optimized pipeline: `unoptimized_pipeline = SyntheticQuestionPipeline()`.
    3.  Load the optimized pipeline: `optimized_pipeline = SyntheticQuestionPipeline(); optimized_pipeline.load("optimized_pipeline.json")`.
    4.  Use the `dspy.evaluate.Evaluate` utility to get scores for both pipelines on the same data using your `llm_metric`.
    5.  Print the scores side-by-side to see the improvement.
    6.  For a qualitative check, loop through 5 examples and print the `gold_question`, the `unoptimized_pipeline`'s output, and the `optimized_pipeline`'s output. This will give you a direct feel for the improvement in question quality.

**Step 5.2: Final Integration**

  * **Objective:** To replace your old `generate` logic with the new, optimized DSPy pipeline.
  * **Action:**
    1.  Create a new version of `question_generator.py`, let's call it `optimized_question_generator.py`.
    2.  In the `_llm_consumer` method, remove the two separate `structured_output_llm_async` calls.
    3.  Instead, load the optimized pipeline once during initialization: `self.pipeline = SyntheticQuestionPipeline(); self.pipeline.load("optimized_pipeline.json")`.
    4.  Configure `dspy.settings` to use your `DispatcherLM`.
    5.  In `_llm_consumer`, make a single call: `prediction = self.pipeline(chunk_text=doc_text, persona_description=relevant_character)`.
    6.  Save the `prediction.query` to your vector store as you did before.


## Additionnal info
The Qdrant database is running from a file. That means there cannot be two clients instantiated at once.
The data payload have the same format as the following example :
{'file_path': '/home/grand/alstom_finetuning/data/raw/31_SIRF_Vehicle_Safety_Guideline.pdf', 'file_name': '31_SIRF_Vehicle_Safety_Guideline.pdf', 'file_type': 'application/pdf', 'file_size': 3056637, 'creation_date': '2025-07-08', 'last_modified_date': '2025-07-08', 'schema_name': 'docling_core.transforms.chunker.DocMeta', 'version': '1.0.0', 'doc_items': [{'self_ref': '#/texts/2139', 'parent': {'$ref': '#/body'}, 'children': [], 'content_layer': 'body', 'label': 'text', 'prov': [{'page_no': 73, 'bbox': {'l': 83.664, 't': 445.0269829101563, 'r': 555.686, 'b': 407.2849829101563, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 193]}]}, {'self_ref': '#/texts/2140', 'parent': {'$ref': '#/body'}, 'children': [], 'content_layer': 'body', 'label': 'text', 'prov': [{'page_no': 73, 'bbox': {'l': 83.664, 't': 393.1869829101563, 'r': 376.699, 'b': 383.2849829101563, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 58]}]}, {'self_ref': '#/texts/2141', 'parent': {'$ref': '#/body'}, 'children': [], 'content_layer': 'body', 'label': 'text', 'prov': [{'page_no': 73, 'bbox': {'l': 141.74, 't': 369.5809829101563, 'r': 522.415, 'b': 360.6469829101563, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 88]}]}, {'self_ref': '#/texts/2142', 'parent': {'$ref': '#/body'}, 'children': [], 'content_layer': 'body', 'label': 'text', 'prov': [{'page_no': 73, 'bbox': {'l': 84.984, 't': 369.5809829101563, 'r': 422.791, 'b': 347.8069829101563, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 71]}]}, {'self_ref': '#/tables/46', 'parent': {'$ref': '#/body'}, 'children': [], 'content_layer': 'body', 'label': 'table', 'prov': [{'page_no': 73, 'bbox': {'l': 128.87083435058594, 't': 336.7027587890625, 'r': 553.3882446289062, 'b': 251.2171630859375, 'coord_origin': 'BOTTOMLEFT'}, 'charspan': [0, 0]}]}], 'headings': ['4.4 Measures within the hardware design'], 'origin': {'mimetype': 'application/pdf', 'binary_hash': 4729834476830034746, 'filename': '31_SIRF_Vehicle_Safety_Guideline.pdf'}, 'generated_query': 'What measures must be included in the system design to prevent malfunctions and errors, according to DIN EN50129 standards, to ensure safety in train operations?', '_node_content': '{"id_": "116238f1-73f4-4f7b-8934-74759c18357a", "embedding": null, "metadata": {"file_path": "/home/grand/alstom_finetuning/data/raw/31_SIRF_Vehicle_Safety_Guideline.pdf", "file_name": "31_SIRF_Vehicle_Safety_Guideline.pdf", "file_type": "application/pdf", "file_size": 3056637, "creation_date": "2025-07-08", "last_modified_date": "2025-07-08", "schema_name": "docling_core.transforms.chunker.DocMeta", "version": "1.0.0", "doc_items": [{"self_ref": "#/texts/2139", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "text", "prov": [{"page_no": 73, "bbox": {"l": 83.664, "t": 445.0269829101563, "r": 555.686, "b": 407.2849829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 193]}]}, {"self_ref": "#/texts/2140", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "text", "prov": [{"page_no": 73, "bbox": {"l": 83.664, "t": 393.1869829101563, "r": 376.699, "b": 383.2849829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 58]}]}, {"self_ref": "#/texts/2141", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "text", "prov": [{"page_no": 73, "bbox": {"l": 141.74, "t": 369.5809829101563, "r": 522.415, "b": 360.6469829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 88]}]}, {"self_ref": "#/texts/2142", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "text", "prov": [{"page_no": 73, "bbox": {"l": 84.984, "t": 369.5809829101563, "r": 422.791, "b": 347.8069829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 71]}]}, {"self_ref": "#/tables/46", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "table", "prov": [{"page_no": 73, "bbox": {"l": 128.87083435058594, "t": 336.7027587890625, "r": 553.3882446289062, "b": 251.2171630859375, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 0]}]}], "headings": ["4.4 Measures within the hardware design"], "origin": {"mimetype": "application/pdf", "binary_hash": 4729834476830034746, "filename": "31_SIRF_Vehicle_Safety_Guideline.pdf"}, "generated_query": "What measures must be included in the system design to prevent malfunctions and errors, according to DIN EN50129 standards, to ensure safety in train operations?"}, "excluded_embed_metadata_keys": ["schema_name", "version", "doc_items", "origin", "file_name", "file_type", "file_size", "creation_date", "last_modified_date", "last_accessed_date"], "excluded_llm_metadata_keys": ["schema_name", "version", "doc_items", "origin", "file_name", "file_type", "file_size", "creation_date", "last_modified_date", "last_accessed_date"], "relationships": {"1": {"node_id": "4cc32c14-0257-451e-b7b3-d8222080b75a", "node_type": "4", "metadata": {"file_path": "/home/grand/alstom_finetuning/data/raw/31_SIRF_Vehicle_Safety_Guideline.pdf", "file_name": "31_SIRF_Vehicle_Safety_Guideline.pdf", "file_type": "application/pdf", "file_size": 3056637, "creation_date": "2025-07-08", "last_modified_date": "2025-07-08"}, "hash": "ba8d1eff50cf50166b0bb016bb0fdf5aba7f822b33ab640bce6d1a95fe1fefdc", "class_name": "RelatedNodeInfo"}, "2": {"node_id": "bdc309ef-ce9c-497b-9f47-e033e0568a57", "node_type": "1", "metadata": {"file_path": "/home/grand/alstom_finetuning/data/raw/31_SIRF_Vehicle_Safety_Guideline.pdf", "file_name": "31_SIRF_Vehicle_Safety_Guideline.pdf", "file_type": "application/pdf", "file_size": 3056637, "creation_date": "2025-07-08", "last_modified_date": "2025-07-08", "schema_name": "docling_core.transforms.chunker.DocMeta", "version": "1.0.0", "doc_items": [{"self_ref": "#/texts/2136", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "caption", "prov": [{"page_no": 73, "bbox": {"l": 85.104, "t": 745.1109829101563, "r": 552.012, "b": 736.1769829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 101]}]}, {"self_ref": "#/texts/2137", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "text", "prov": [{"page_no": 73, "bbox": {"l": 141.74, "t": 732.2709829101562, "r": 267.166, "b": 723.3369829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 28]}]}, {"self_ref": "#/tables/45", "parent": {"$ref": "#/body"}, "children": [], "content_layer": "body", "label": "table", "prov": [{"page_no": 73, "bbox": {"l": 128.83053588867188, "t": 712.7241363525391, "r": 552.8355712890625, "b": 498.5643615722656, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 0]}]}], "headings": ["-Appendix F -Criteria catalog for hardware -"], "origin": {"mimetype": "application/pdf", "binary_hash": 4729834476830034746, "filename": "31_SIRF_Vehicle_Safety_Guideline.pdf"}}, "hash": "67e0aa589539547f13e49e5c376c82be2b7edd7ee8264bd207c0dffdbe4348b5", "class_name": "RelatedNodeInfo"}, "3": {"node_id": "d963be82-04e1-4f59-8023-79ad8a0c11cd", "node_type": "1", "metadata": {"schema_name": "docling_core.transforms.chunker.DocMeta", "version": "1.0.0", "doc_items": [{"self_ref": "#/texts/2148", "parent": {"$ref": "#/tables/47"}, "children": [], "content_layer": "body", "label": "caption", "prov": [{"page_no": 74, "bbox": {"l": 84.984, "t": 745.1109829101563, "r": 522.534, "b": 723.3369829101563, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 179]}]}, {"self_ref": "#/tables/47", "parent": {"$ref": "#/body"}, "children": [{"$ref": "#/texts/2148"}], "content_layer": "body", "label": "table", "prov": [{"page_no": 74, "bbox": {"l": 128.7692413330078, "t": 713.1270141601562, "r": 553.6492919921875, "b": 458.26129150390625, "coord_origin": "BOTTOMLEFT"}, "charspan": [0, 0]}]}], "headings": ["-Appendix F -Criteria catalog for hardware -"], "origin": {"mimetype": "application/pdf", "binary_hash": 4729834476830034746, "filename": "31_SIRF_Vehicle_Safety_Guideline.pdf"}}, "hash": "3c309fa37c7f21d329879a1656caa31f00be8a4aac8fd343e50ddf280b23bc5c", "class_name": "RelatedNodeInfo"}}, "metadata_template": "{key}: {value}", "metadata_separator": "\\n", "text": "The system design must include measures against malfunctions and errors, such as program sequence monitoring for the detection of coarse errors in the program run, including \\"hanging\\" programs.\\nRequirements following table E.5 of DIN EN 50129 [11, 12].\\nMeasures for the hardware architecture for the protection against individual failures of\\nTable 22: discrete components depending on the safety requirement level\\nAll dangerous failure types must result in safety directed failure reactions or the nonvolatile safety (see DIN EN 50129) must be reached based on non- volatile physical characteristics., SAS 1 = R. All dangerous failure types must result in safety directed failure reactions or the nonvolatile safety (see DIN EN 50129) must be reached based on non- volatile physical characteristics., SAS 2 = . All dangerous failure types must result in safety directed failure reactions or the nonvolatile safety (see DIN EN 50129) must be reached based on non- volatile physical characteristics., SAS 3 - 4 = HR", "mimetype": "text/plain", "start_char_idx": null, "end_char_idx": null, "metadata_seperator": "\\n", "text_template": "{metadata_str}\\n\\n{content}", "class_name": "TextNode"}', '_node_type': 'TextNode', 'document_id': '4cc32c14-0257-451e-b7b3-d8222080b75a', 'doc_id': '4cc32c14-0257-451e-b7b3-d8222080b75a', 'ref_doc_id': '4cc32c14-0257-451e-b7b3-d8222080b75a'}

By following this plan, you will systematically build a high-quality dataset, refactor your logic into a testable and optimizable DSPy module, integrate your existing async dispatcher, and produce a final, high-performance pipeline with optimized prompts.

Wait for further instructions.