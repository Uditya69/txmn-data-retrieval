## Purpose

Tracks each topic's lifecycle through explicit states (discovered, emerging, active, fading,
reactive, dormant) with persistence rules, so a short rabbit hole cannot masquerade as a durable
shift in a user's profile.

## ADDED Requirements

### Requirement: Every topic has one current state from a fixed enumeration
Each (user, topic) pair SHALL have exactly one current state, drawn from: discovered, emerging,
active, fading, reactive, dormant. A newly created topic SHALL start in the discovered state.

#### Scenario: New topic
- **WHEN** a query event creates a new research topic for a user
- **THEN** that topic's state is "discovered".

### Requirement: State transitions require sustained evidence
A transition from a lower-engagement state to a higher-engagement state (e.g. emerging → active)
SHALL require evidence sustained across more than one query event or session, not a single
session's spike in interest score.

#### Scenario: Single-session spike does not promote a topic
- **WHEN** a user submits several related queries with strong engagement in one session and then
  nothing further
- **THEN** the topic does not reach "active" state from that single session alone.

#### Scenario: Sustained activity promotes a topic
- **WHEN** a user shows related, corroborating engagement with a topic across multiple sessions
  over time
- **THEN** the topic progresses through discovered → emerging → active accordingly.

### Requirement: Inactivity moves a topic toward dormant, not an abrupt drop
A topic with declining interest score SHALL pass through "fading" before reaching "dormant"; the
system SHALL NOT transition a topic directly from "active" to "dormant" in one step.

#### Scenario: Gradual decline
- **WHEN** an active topic's interest score declines over consecutive periods with no new
  activity
- **THEN** the topic's state moves to "fading" first, and only reaches "dormant" after continued
  inactivity beyond that.

### Requirement: A dormant topic can return via "reactive," not by silently resetting
When new activity appears on a topic currently in "dormant" state, the system SHALL transition it
to "reactive" first, distinct from a brand-new "discovered" topic, before it can return to
"active".

#### Scenario: User returns to an old topic
- **WHEN** a user with a dormant "GST" topic submits new GST-related queries six months later
- **THEN** the "GST" topic transitions from dormant to reactive, and its prior history (episodes,
  discovery date) is preserved rather than discarded.

### Requirement: Pivot detection requires corroboration, not a single signal
The system SHALL only report a "pivot" (one topic declining while another rises) once the rising
topic's evidence is corroborated by repeated queries and meaningful interactions sustained over
more than one session, not merely a rank crossover in a single day's scores.

#### Scenario: False-positive pivot avoided
- **WHEN** one atypical query briefly raises a new topic's score above a declining topic's score
  for a single day
- **THEN** the system does not report a pivot from that single day's data alone.

#### Scenario: Genuine pivot detected
- **WHEN** a new topic's interest score rises and remains elevated with repeated, corroborating
  engagement across multiple sessions while a previously active topic's score continues to
  decline
- **THEN** the system reports a pivot from the declining topic to the rising topic.
