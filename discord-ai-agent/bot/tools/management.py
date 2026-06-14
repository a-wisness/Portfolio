"""Server management tool definitions — role and channel operations."""
from ..llm.provider import Tool

list_roles_tool = Tool(
    name="list_roles",
    description="List all assignable roles in the guild with their IDs.",
    parameters={"type": "object", "properties": {}, "required": []},
)

find_member_tool = Tool(
    name="find_member",
    description="Find a guild member by display name substring or exact Discord user ID.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Display name substring or numeric Discord user ID.",
            },
        },
        "required": ["query"],
    },
)

assign_role_tool = Tool(
    name="assign_role",
    description="Assign a role to a guild member. Requires the role_assign action to be enabled.",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Discord user ID."},
            "role_id": {"type": "integer", "description": "Role ID to assign."},
            "reason": {"type": "string", "description": "Reason for the role assignment."},
        },
        "required": ["user_id", "role_id", "reason"],
    },
)

remove_role_tool = Tool(
    name="remove_role",
    description="Remove a role from a guild member. Requires the role_remove action to be enabled.",
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Discord user ID."},
            "role_id": {"type": "integer", "description": "Role ID to remove."},
            "reason": {"type": "string", "description": "Reason for removing the role."},
        },
        "required": ["user_id", "role_id", "reason"],
    },
)

list_channels_tool = Tool(
    name="list_channels",
    description="List text channels and categories in the guild with their IDs.",
    parameters={"type": "object", "properties": {}, "required": []},
)

create_channel_tool = Tool(
    name="create_channel",
    description="Create a new text channel. Requires the channel_create action to be enabled.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Channel name (lowercase, hyphens). Discord enforces slug format.",
            },
            "topic": {"type": "string", "description": "Optional channel topic/description."},
            "category_name": {
                "type": "string",
                "description": "Optional category name to place the channel in.",
            },
        },
        "required": ["name"],
    },
)

archive_channel_tool = Tool(
    name="archive_channel",
    description=(
        "Archive a text channel by renaming it with an 'archive-' prefix and locking it for "
        "@everyone. Requires the channel_archive action to be enabled."
    ),
    parameters={
        "type": "object",
        "properties": {
            "channel_id": {"type": "integer", "description": "ID of the channel to archive."},
            "reason": {"type": "string", "description": "Reason for archiving."},
        },
        "required": ["channel_id", "reason"],
    },
)
