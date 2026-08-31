import logging

from chat.repository import append_turn, save_retrieval_trace

logger = logging.getLogger(__name__)


async def record_conversation_turn(conversations, conversation_id: str, user_id: str, title: str, messages: list[dict]) -> None:
    try:
        await append_turn(conversations, conversation_id, user_id, title, messages)
    except Exception:
        logger.warning("conversation turn write failed for user %r, conversation %r", user_id, conversation_id, exc_info=True)


async def record_retrieval_trace(
    retrieval_traces, conversation_id: str, user_id: str, mode: str, query: str,
    langfuse_trace_id: str | None, langfuse_observation_id: str | None,
    instant: dict | None = None, ai_mode: dict | None = None,
) -> None:
    try:
        await save_retrieval_trace(
            retrieval_traces, conversation_id, user_id, mode, query,
            langfuse_trace_id, langfuse_observation_id, instant=instant, ai_mode=ai_mode,
        )
    except Exception:
        logger.warning("retrieval trace write failed for user %r, conversation %r, mode %r", user_id, conversation_id, mode, exc_info=True)
