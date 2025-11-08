"""
Centralized schema definitions for LLM providers.
This ensures consistency across all providers and reduces code duplication.
"""


from pydantic import BaseModel


def get_schema_definition(schema: type[BaseModel]) -> str:
    """
    Get a consistent schema definition for any Pydantic model.
    Returns a string description suitable for LLM prompts.
    """
    schema_name = schema.__name__
    
    # Define schema descriptions for common models
    SCHEMA_DEFINITIONS = {
        "GraphUpdate": """Return JSON with these fields:
- target_graph: string (graph name)
- new_nodes: array of node objects, each with:
  - id: string (unique node identifier)
  - type: string (node type)
  - label: string (human-readable label)
  - refs: array of strings (references to other nodes)
- new_edges: array of edge objects, each with:
  - type: string (edge type)
  - src: string (source node ID)
  - dst: string (destination node ID)
- node_updates: array of node update objects, each with:
  - id: string (node ID to update)
  - description: string or null (new description)
  - properties: string or null (JSON string of properties)
  - new_observations: array (LEAVE EMPTY - only for agent phase)
  - new_assumptions: array (LEAVE EMPTY - only for agent phase)
- is_complete: boolean (whether this graph is complete)
- completeness_reason: string or null (why complete or what's missing)""",

        "GraphDiscovery": """Return JSON with these fields:
- graphs_needed: array of {name: string, focus: string}
- suggested_node_types: array of strings
- suggested_edge_types: array of strings""",

        "InvestigationPlan": """Return JSON with these fields:
- investigations: array of investigation items, each containing:
  - goal: string (investigation goal or question)
  - focus_areas: array of strings
  - priority: integer (1-10, where 10 is highest)
  - reasoning: string (rationale for why this is promising)
  - category: string ("aspect" or "suspicion")
  - expected_impact: string ("high", "medium", or "low")
  
IMPORTANT: You MUST return an array with the exact number of investigations requested.""",

        "PlanBatch": """Return JSON with these fields:
- investigations: array of investigation items; for each item include:
  - goal: string (investigation goal or question)
  - focus_areas: array of strings (may be empty)
  - priority: integer 1-10 (10 = highest urgency)
  - reasoning: string explaining why now and exit criteria
  - category: string ("aspect" or "suspicion")
  - expected_impact: string ("high", "medium", or "low")

Return exactly the number of investigations requested; if none apply, return an empty array.""",

        "AgentDecision": """Return JSON with these fields:
- action: string (one of: load_graph, load_nodes, update_node, form_hypothesis, update_hypothesis, complete)
- reasoning: string (your reasoning for this action)
- parameters: object with action-specific fields:
  - For load_graph: {"graph_name": "string"}
  - For load_nodes: {"node_ids": ["array", "of", "strings"]}
  - For update_node: {"node_id": "string", "observations": ["array"], "assumptions": ["array"]}
  - For form_hypothesis: {"title": "string", "description": "string", "confidence": number 0-1}
  - For update_hypothesis: {"hypothesis_id": "string", "confidence": number, "evidence": ["array"]}
  - For complete: {} or omit entirely

IMPORTANT: Only include the parameters required for your chosen action.""",
    }
    
    # Return predefined schema if available
    if schema_name in SCHEMA_DEFINITIONS:
        return SCHEMA_DEFINITIONS[schema_name]
    
    # Otherwise, generate schema from Pydantic model
    schema_fields = []
    for field_name, field_info in schema.model_fields.items():
        field_type = str(field_info.annotation).replace('typing.', '')
        description = field_info.description or ""
        schema_fields.append(f"- {field_name}: {field_type} {f'({description})' if description else ''}")
    
    return "\nReturn JSON with these fields:\n" + "\n".join(schema_fields)
