"""KB search tool definition for the agentic loop."""
from ..llm.provider import Tool

search_kb_tool = Tool(
    name="search_knowledge_base",
    description=(
        "Search this server's knowledge base for relevant information. "
        "Try this before answering from general knowledge."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or phrase to search for.",
            },
        },
        "required": ["query"],
    },
)
