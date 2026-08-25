## Purpose

Turns each incoming query into a structured Query Understanding Record — concepts, legal
entities, research objective, and confidence — so downstream topic clustering and interest
scoring have a consistent unit of evidence instead of raw query text.

## ADDED Requirements

### Requirement: Structured extraction per query
For every AI Mode/Agentic query issued by a logged-in user, the system SHALL produce a Query
Understanding Record containing: the original query text, a list of legal concepts, a list of
legal entities, a list of research objectives, a specificity score, and a confidence score.

#### Scenario: Well-formed legal query
- **WHEN** a logged-in user submits "Can a director be personally liable for company's debt
  under IBC?"
- **THEN** the system produces a Query Understanding Record whose concepts include
  "director liability" and "corporate debt", whose legal entities include "IBC", whose research
  objective includes "determine liability", and whose confidence score reflects the extractor's
  certainty.

#### Scenario: Guest query
- **WHEN** a query is submitted with no resolvable `user_id`
- **THEN** the system SHALL NOT produce or persist a Query Understanding Record for that query.

### Requirement: Extraction failure does not block the response
Query Understanding Record extraction SHALL run without delaying or altering the user-visible
response, and SHALL degrade to "no record produced" on any extraction failure rather than raising
an error into the request path.

#### Scenario: Extraction call fails
- **WHEN** the model call used to produce a Query Understanding Record fails or returns
  malformed output
- **THEN** the user-visible response is unaffected, no Query Understanding Record is persisted for
  that query, and the failure is logged.

### Requirement: Confidence is bounded and explicit
Every Query Understanding Record's specificity and confidence values SHALL be numeric scores in a
fixed, documented range (e.g. 0.0–1.0), never omitted, so downstream scoring can treat low-
confidence extractions differently from high-confidence ones.

#### Scenario: Low-confidence extraction
- **WHEN** the extractor produces concepts/entities it is not confident about
- **THEN** the Query Understanding Record still records a confidence score reflecting that
  uncertainty rather than omitting the record or asserting full confidence.

### Requirement: Extracted content is validated before reuse
Any structured field in a Query Understanding Record that will later be interpolated into a
prompt (directly or via a derived summary) SHALL be validated/sanitized before storage, so a
malicious or malformed query cannot inject unvalidated content into a future prompt for the same
account.

#### Scenario: Adversarial query text
- **WHEN** a query's text or the model's extracted fields contain content resembling prompt
  injection
- **THEN** the stored Query Understanding Record's fields are constrained to their expected
  types/shapes, and no unvalidated free-text field is later interpolated verbatim into a system
  prompt.
