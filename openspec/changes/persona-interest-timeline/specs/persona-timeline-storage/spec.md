## Purpose

Persists a user's query understanding records and derived state transitions as an append-only
event history, so the current persona is always a reconstructable, explainable derivation rather
than a single mutable document that silently loses prior evidence on overwrite.

## ADDED Requirements

### Requirement: Persona data is stored as an append-only event history
Query Understanding Records and derived interest-state transitions for a user SHALL be persisted
as an append-only sequence of events. Recording a new event SHALL NOT delete, overwrite, or
mutate any prior event.

#### Scenario: New event does not erase history
- **WHEN** a new query event is recorded for a user who already has prior history
- **THEN** all prior events remain retrievable, unmodified, after the new event is appended.

### Requirement: Current persona is a derived snapshot, not stored state
The system SHALL compute a user's current persona (active/dominant topics, states, confidences)
by deriving it from the event history at read time (or from a cache that is invalidateable and
reproducible from that history), never by reading a single hand-maintained mutable "current
persona" field as the sole source of truth.

#### Scenario: Snapshot reproducibility
- **WHEN** the current persona snapshot for a user is derived twice from the same underlying
  event history
- **THEN** both derivations produce the same snapshot.

### Requirement: History supports "why" and "when" queries
Given a user and a topic, the system SHALL be able to answer when that topic was first
discovered, when it changed state, and which query events contributed to a given state
transition.

#### Scenario: Explain a state change
- **WHEN** an operator asks why a topic transitioned from "emerging" to "active" on a given date
- **THEN** the system can identify the query events in the window leading up to that transition.

### Requirement: A down or unreachable event store degrades to guest-equivalent behavior
If the event store is unreachable when handling a request, the system SHALL proceed as if the
user were a guest for that request (no persona context, no event recorded) rather than failing
the request.

#### Scenario: Store unavailable during read
- **WHEN** the event/derivation store is unreachable while loading persona context for a
  logged-in user's request
- **THEN** the request proceeds with empty persona context, and the failure is logged, not
  raised to the caller.

#### Scenario: Store unavailable during write
- **WHEN** the event store is unreachable while recording a query event after a response has
  been sent
- **THEN** the write is dropped and logged; the user-visible response is unaffected.

### Requirement: Existing flat persona documents are handled on transition
For a user with an existing flat (pre-timeline) persona document, the system SHALL define and
apply a cold-start policy (e.g. treat as prior/seed evidence, or start a fresh history) rather
than leaving that user's persona in an undefined or inconsistent state after this change ships.

#### Scenario: Pre-existing user
- **WHEN** a user who already has a flat persona document from before this change makes a new
  query
- **THEN** the system applies the defined cold-start policy consistently, without erroring or
  silently discarding the user's identity/account linkage.
