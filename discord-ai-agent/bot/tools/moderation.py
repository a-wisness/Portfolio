"""Moderation tool definitions for the agent — Phase 4. See constitution/roadmap.md."""
from ..llm.provider import Tool

warn_user = Tool(
    name="warn_user",
    description="Issue a formal warning to a Discord user and log it.",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Discord user ID to warn"},
            "reason": {"type": "string", "description": "Reason for the warning"},
        },
        "required": ["user_id", "reason"],
    },
)

delete_message = Tool(
    name="delete_message",
    description="Delete a specific message from a channel.",
    parameters={
        "type": "object",
        "properties": {
            "channel_id": {"type": "integer", "description": "Discord channel ID"},
            "message_id": {"type": "integer", "description": "Discord message ID to delete"},
            "reason": {"type": "string", "description": "Reason for deletion"},
        },
        "required": ["channel_id", "message_id", "reason"],
    },
)
