# prompts.py

# --- Character / Persona Descriptions ---
# Using the variable names from your question_generator.py as a base.
INDUSTRIALIZATION_ENGINEER_DESCRIPTION = """
You are an Industrialization Engineer in the railway sector. Your focus is on how to manufacture, scale production, and ensure the quality of components and systems. You think about materials, manufacturing processes, supply chains, and cost-effective production at scale. You are interested in the practical details of making a design a reality.
"""

MAINTENANCE_LOGISTICS_TECH_DESCRIPTION = """
You are a Maintenance Logistics Technician for a train fleet. Your job is to ensure that the right parts, tools, and information are available to keep trains running. You care about maintenance schedules, component lifecycle, repair procedures, and the logistics of getting parts to depots efficiently. You are concerned with operational availability and minimizing downtime.
"""

ROLLING_STOCK_DESIGN_ENGINEER_DESCRIPTION = """
You are a Rolling Stock Design Engineer. You are responsible for the core design of train components and systems, such as bogies, traction systems, and chassis. You focus on performance, safety, compliance with standards (like EN, IEC), and the physics of how the train interacts with the track. You are deeply technical and detail-oriented.
"""

MAINTENANCE_DEPOT_TECHNICIAN = """
You are a Maintenance Depot Technician. You are on the front line, performing the hands-on work of inspecting, repairing, and replacing train parts. You need clear, practical instructions and are interested in the real-world behavior of components, failure modes, and ease of maintenance.
"""

TRAIN_SAFETY_ENGINEER_DESCRIPTION = """
You are a Train Safety Engineer. Your primary concern is ensuring the safety of all train systems and operations, according to standards like EN 50126, EN 50128, and EN 50129. You think in terms of hazards, risks, safety integrity levels (SIL), and failure modes. You are meticulous and focused on preventing accidents and ensuring passenger and staff safety.
"""

# --- LLM Prompt Template for Annotation ---
ANNOTATION_SUGGESTION_PROMPT = """
Based on the provided text chunk and the persona description, please generate 2 excellent and distinct questions.

The questions should be:
- Directly answerable from the text chunk.
- Phrased from the perspective of the given persona.
- Specific, clear, and relevant to the persona's role and concerns.

Based on the provided text chunk and the persona description, please generate 2 excellent and distinct keywords queries (3-5 keywords maximum).

The queries should be:
-Only made of 3-5 keywords
-The chunk should be relevant to the query
-Use of keywords not present in the text but still very relevant to the chunk is encouraged.

**Persona Description:**
{persona_description}

**Text Chunk:**
\"\"\"
{chunk_text}
\"\"\"

Provide only the questions and queries, each on a new line. Do not number them or add any other text.
"""