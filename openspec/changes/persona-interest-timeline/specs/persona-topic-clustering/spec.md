## Purpose

Groups a user's query events into research topics and time-bounded research episodes, so
interest tracking operates on coherent subjects rather than raw per-query category tags.

## ADDED Requirements

### Requirement: Queries are grouped into research topics
The system SHALL assign each Query Understanding Record to a research topic (existing or newly
created) based on a combination of semantic similarity, legal-entity overlap, intent similarity,
temporal proximity, and behavioral continuity with the user's recent query events.

#### Scenario: Related queries join the same topic
- **WHEN** a user submits "Section 7 IBC", then shortly after "financial creditor filing under
  section 7", then "limitation for Section 7 application"
- **THEN** all three query events are assigned to the same research topic.

#### Scenario: Unrelated query starts a new topic
- **WHEN** a user with an established "IBC / Section 7" topic submits "GST registration
  cancellation" with no semantic, entity, or temporal continuity with that topic
- **THEN** the system creates a new research topic rather than attaching the query to the
  existing one.

### Requirement: A topic can contain multiple time-bounded research episodes
The system SHALL be able to represent multiple, separately time-bounded research episodes under
the same research topic (e.g. "GST → ITC research" in January and "GST → registration
cancellation" in March both under the "GST" topic).

#### Scenario: Same topic, separate episodes
- **WHEN** a user researches "GST input tax credit" intensively for two weeks in January, has no
  GST-related queries for six weeks, then researches "GST registration cancellation" for a week
  in March
- **THEN** the system records two distinct research episodes under the "GST" topic, each with
  its own start/end bounds.

### Requirement: Clustering assignment is queryable and explainable
For any query event, the system SHALL be able to report which topic (and episode, if
applicable) it was assigned to and, at a coarse level, why (which similarity signals
contributed).

#### Scenario: Explain an assignment
- **WHEN** an operator inspects a specific query event's topic assignment
- **THEN** the system reports the assigned topic/episode identifiers and the contributing
  similarity signals (e.g. "entity overlap: IBC; temporal proximity: 4 minutes").

### Requirement: Clustering degrades safely on ambiguous input
When a query event's similarity to all existing topics is below the threshold for a confident
match, the system SHALL create a new topic rather than force an assignment to the closest
existing one.

#### Scenario: Ambiguous single-word query
- **WHEN** a query is too short or generic to match any existing topic with sufficient
  confidence
- **THEN** the system starts a new topic for it rather than misattributing it to an unrelated
  existing topic.
