"""Agent command queue constants.

These command_type strings are stored in `agent_command.command_type`.
"""

COMMAND_COMPACT_SESSION_MEMORY = "assistant_compact_session_memory_v1"

# Orchestrator loop (internal/self-work).
COMMAND_ORCHESTRATOR_TICK = "orchestrator_tick_v1"
COMMAND_REVIEW_SUPPORT_SESSION = "assistant_review_support_session_v1"
COMMAND_REVIEW_REPO = "assistant_review_repo_v1"
COMMAND_BUILD_SUPPORT_DIGEST = "assistant_build_support_digest_v1"

# Integrations
COMMAND_SLACK_POST_MESSAGE = "slack_post_message_v1"

# Data sync
COMMAND_LEGACY_SYNC_ALL = "legacy_sync_all_v1"
