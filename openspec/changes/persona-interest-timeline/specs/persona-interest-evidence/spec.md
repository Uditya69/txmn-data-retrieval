## Purpose

Assigns a behavioral evidence weight to each query event based on how the user interacted with
it, so a single search and a search followed by deep engagement contribute differently to a
topic's interest score.

## ADDED Requirements

### Requirement: Every query event carries an evidence weight
Each query event SHALL be assigned a numeric evidence weight derived from the interaction signals
available for that event (query submitted, result clicked, document opened, document read
deeply, document saved, repeated related query, return to topic later). A query with no further
interaction SHALL still carry a minimum non-zero weight for having been submitted.

#### Scenario: Search with no follow-up
- **WHEN** a user submits a query and takes no further action
- **THEN** the event's evidence weight reflects only the "query submitted" signal.

#### Scenario: Search with deep engagement
- **WHEN** a user submits a query, opens multiple results, and saves a document
- **THEN** the event's evidence weight is higher than a query-only event, reflecting the combined
  submitted/opened/saved signals.

### Requirement: Evidence weighting is monotonic and additive per event
Adding a stronger interaction signal to an event (e.g. upgrading "clicked" to "saved") SHALL
never decrease that event's evidence weight relative to the weaker signal alone.

#### Scenario: Signal upgrade
- **WHEN** an event initially has only a "result clicked" signal and later gains a "document
  saved" signal for the same query
- **THEN** the event's recomputed evidence weight is greater than or equal to its prior weight.

### Requirement: Missing interaction signals degrade gracefully
When interaction signals beyond "query submitted" are unavailable (not yet instrumented, or the
frontend surface does not report them), the system SHALL still compute a valid evidence weight
using only the signals it has, rather than failing or blocking topic scoring.

#### Scenario: Frontend does not report click-through
- **WHEN** a query event has no click/open/save signals available at all
- **THEN** the system computes an evidence weight from the submitted-query signal alone, and
  interest scoring proceeds using that weight.
