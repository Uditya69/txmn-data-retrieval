## Purpose

Renders the current persona snapshot into prompt-facing context text — including confidence-
weighted interpretations of the current query — for use in AI Mode synthesis, replacing the flat
"frequently asks about X, Y; expertise: Z" sentence.

## ADDED Requirements

### Requirement: Rendered context reflects current, not lifetime-averaged, interest
The persona context rendered for a request SHALL be based on the user's current active/dominant
topics and their states, derived from the timeline, rather than a lifetime-cumulative average
across all history.

#### Scenario: Pivoted user
- **WHEN** a user's dominant topic six months ago was "IBC" and their currently active topic is
  "GST"
- **THEN** the rendered persona context reflects "GST" as the current focus, not a blended
  average of "IBC" and "GST".

### Requirement: Low-evidence users receive no persona context
When a user has insufficient accumulated evidence (no topic has reached a minimum confidence/
evidence threshold), the system SHALL render empty persona context, equivalent to a guest.

#### Scenario: New user
- **WHEN** a newly logged-in user has made only one or two queries
- **THEN** the rendered persona context is empty.

### Requirement: Persona context is advisory, not authoritative, for the current query
Rendered persona context SHALL be accompanied by an explicit instruction that it is a prior
about the user's typical usage, not a fact about the current query, and that the model should
disregard it when the current query's own content conflicts with or is unrelated to it.

#### Scenario: Conflicting current query
- **WHEN** a user with a strong "GST" persona submits a query clearly about "IBC liquidation"
- **THEN** the rendered context still includes the disregard-if-conflicting instruction, and the
  system does not force a GST framing onto the response.

### Requirement: Ambiguous current-query terms may surface topic-hypothesis confidences
For a current query whose terms are ambiguous across more than one of the user's known topics,
the system SHALL be capable of rendering candidate topic interpretations with their confidence
weights (e.g. "IBC limitation 0.81 / Civil limitation 0.12 / GST limitation 0.07") as part of the
persona context; when it does render such candidates, it SHALL present them as weighted
possibilities, never as one asserted-certain interpretation.

#### Scenario: Ambiguous term with skewed history
- **WHEN** a user with heavy prior "IBC" activity submits the ambiguous query "limitation
  period"
- **THEN** the rendered context may present "IBC" as the higher-confidence interpretation
  alongside other candidates, without asserting it as the only possible meaning.

### Requirement: Persona context never feeds retrieval ranking or routing
Persona context, including any topic-hypothesis confidence, SHALL be used only in prompt text for
synthesis (and, if applicable, SLM query rewriting) and SHALL NOT be used as a raw-score or
weighting input to RRF fusion or Milvus collection routing.

#### Scenario: No routing side effect
- **WHEN** persona context (including topic-hypothesis confidences) is available for a request
- **THEN** the Milvus collections searched and the RRF fusion weights used for that request are
  identical to what they would be with no persona context at all.
