## Purpose

Computes a per-topic, per-time interest score from frequency, semantic coherence, interaction
strength, recency, and repetition, producing a time series that later drives state transitions —
replacing a single cumulative-mean affinity value.

## ADDED Requirements

### Requirement: Interest score is a function of time, not a lifetime constant
For each (user, topic) pair, the system SHALL maintain an interest score that is a function of
time, computed from frequency, semantic coherence, accumulated interaction/evidence strength,
recency, and repetition, such that the score for the same topic can differ between two points in
time.

#### Scenario: Score changes across time
- **WHEN** a user is highly active on a topic in month 1 and inactive on it thereafter
- **THEN** the topic's interest score at the end of month 1 is higher than its interest score
  three months later, all else equal.

### Requirement: Recency discounts older evidence
Interest scoring SHALL weight more recent query events more heavily than older ones for the same
topic, so a topic with no recent activity trends toward a lower score over time even without new
negative evidence.

#### Scenario: Topic goes quiet
- **WHEN** a topic had frequent activity two months ago and no activity since
- **THEN** its current interest score is lower than it was two months ago.

### Requirement: A single query event cannot dominate a topic's score
No single query event's evidence weight SHALL be able to move a topic's interest score from a low
value directly to the maximum representable value; sustained or repeated evidence is required to
reach high scores.

#### Scenario: One high-engagement query
- **WHEN** a user submits a single query on a brand-new topic and saves one document from it
- **THEN** the resulting interest score for that topic is elevated but not at the maximum
  attainable score.

### Requirement: Scoring is deterministic given the same event history
Given an identical sequence of query events (with their evidence weights and timestamps) for a
topic, the computed interest score at any given time SHALL be reproducible — no non-deterministic
or model-call-dependent step in the scoring computation itself.

#### Scenario: Recompute from history
- **WHEN** the interest score for a topic is recomputed from the same stored event history twice
- **THEN** both computations produce the same score.
