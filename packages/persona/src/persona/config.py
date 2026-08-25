from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PersonaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongo_uri: str
    mongo_db: str

    # Interest-scoring recency decay rate (per day) - interest_score() weights an
    # event's contribution by exp(-persona_decay_lambda * age_in_days). Higher =
    # faster forgetting. See design.md decision #3.
    persona_decay_lambda: float = 0.03
    # Minimum combined similarity (embedding cosine + entity overlap + temporal
    # proximity) for a new query event to join an existing topic rather than
    # start a new one. See design.md decision #2.
    persona_cluster_similarity_threshold: float = 0.6
    # Gap (hours) since a topic's last event beyond which a new event on that
    # topic starts a new research episode rather than extending the current one.
    persona_episode_gap_hours: float = 168.0  # 7 days
    # Minimum distinct corroborating sessions (calendar days with an event)
    # since a topic's last state transition before a further promotion is
    # allowed - prevents a single session's spike from promoting a topic.
    persona_min_sessions_per_transition: int = 2
    # Minimum interest score for a topic to be considered "trusted" enough to
    # surface in rendered persona context.
    persona_context_confidence_floor: float = 0.15


@lru_cache
def get_persona_settings() -> PersonaSettings:
    return PersonaSettings()
