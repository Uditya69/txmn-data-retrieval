import logging

from chat.repository import append_turn

logger = logging.getLogger(__name__)


async def record_conversation_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> None:
    try:
        await append_turn(conversations, conversation_id, user_id, title, messages)
    except Exception:
        logger.warning("conversation turn write failed for user %r, conversation %r", user_id, conversation_id, exc_info=True)
