CONFIG_GEN_PROMPT = """
Given a passage and a character description, select the appropriate option from two fields : Query type, Query Format.

First, select the query type that the character might ask about the passage. Then, choose the format of the possible query based on the passage, the character and the query type.
There can be multiple possibilities. Generate up to five distinct (query_type, query_format) pairs.

Query types :
- Fact Retrieval (Explicit facts directly from a knowledge base)
- Analytical/Explanatory (Require synthesis or explanation of retrieved information)
- Reasoning (Need cross-source reasoning (connecting multiple facts or documents)
- Hypothetical/Scenario-Based (Explore "what-if" scenarios requiring both knowledge and creativity)
- Summarization (Request condensed overviews of large datasets or documents)
- Evaluative (Demand critical analysis or judgment of retrieved content)

Query format :
- keywords
- formal question

Now, generate the output based on the passage and character. Ensure to generate only JSON.

Passage :
{passage}

Character :
{character}
"""

TRAIN_SAFETY_ENGINEER_DESCRIPTION = """
A train safety engineer's goal is to ensure the inherent safety of train operations and design throughout the railway system. 
This involves identifying potential hazards (from vehicles, signaling, infrastructure), 
designing safety systems and procedures, performing rigorous risk assessments, 
ensuring compliance with stringent safety standards,
analyzing incidents to prevent recurrence, and fostering a pervasive safety culture within the organization.
"""

MAINTENANCE_LOGISTICS_TECH_DESCRIPTION = """
A maintenance logistics technician's goal is to efficiently manage the physical flow 
and availability of materials, tools, spare parts, and resources required to keep rolling stock and depot operations running reliably. 
This involves forecasting needs, managing inventory (procurement, storage, distribution), kitting parts, coordinating repair logistics,
ensuring technicians have what they need when they need it, and optimizing parts and material costs within the maintenance workflow.
"""


ROLLING_STOCK_DESIGN_ENGINEER_DESCRIPTION = """
A rolling stock design engineer's goal is to design, develop, and technically specify rail vehicles (locomotives, carriages, freight cars) or subsystems (bogies, braking, interiors, HVAC). 
Their focus is on ensuring vehicles meet functional requirements (performance, capacity, comfort), 
technical specifications, engineering standards (structural integrity, dynamics, electrical systems), safety norms, operational demands, cost targets,
and lifecycle requirements like maintainability and reliability.
"""

INDUSTRIALIZATION_ENGINEER_DESCRIPTION = """
An industrialization engineer's goal is translate finalized rolling stock designs or new maintenance/modernization processes into 
efficient, high-quality, and scalable production within manufacturing facilities or maintenance depots. 
They focus on designing production/assembly layouts, developing detailed work instructions and bill of materials (BOMs), 
selecting tools/equipment, implementing quality control points, optimizing costs and lead times, qualifying production systems, 
and handing over validated processes to operations for stable serial production or repair execution.
"""

MAINTENANCE_DEPOT_TECHNICIAN = """
A maintenance depot technician's goal is to perform the hands-on scheduled and unscheduled maintenance, 
diagnostics, repairs, inspections, and servicing of rolling stock to ensure fleet availability, reliability, 
safety, and compliance with maintenance plans and regulations. Their daily work directly executes tasks on the vehicles 
(mechanical, electrical, pneumatic systems), troubleshoots faults, ensures proper documentation of work performed, 
and adheres strictly to safety protocols within the depot environment.
"""

QUESTION_GEN_PROMPT = """
Given a character description, a passage and requirement lists, generate a set of queries from the character's perspective that satifies the requirements and can be used to retrieve the passage. Assume you are the character. 
Do not repeat what character you are. The queries must be relevant to the passage. Generate 5 queries and use each requirement pair at least once.
Return the result in JSON format.

Character : 
{character}

Passage : 
{passage}

Requirement list :
{req_list}
"""