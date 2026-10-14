# Business Entity Resolution

Matching noisy business records from three independent sources to the real-world businesses they describe (Amazon ML Challenge 2026).

## Language

### Records and sources

**Record**:
One row in a source file: an entity_id, business name, business address and country.
_Avoid_: Row, entry, listing

**Reference Record**:
A Record from Source 1. Source 1 is deduplicated, so each Reference Record is a distinct real-world business.
_Avoid_: S1 entity, master record, anchor

**Mention**:
A Record from Source 2 or Source 3. Several Mentions can describe the same business.
_Avoid_: S2/S3 entity, duplicate, fragment

### Matching

**Match**:
A link between a Reference Record and a Mention that describes the same real-world business. A Mention matches at most one Reference Record.
_Avoid_: Link, pair, merge

**Singleton**:
A Reference Record with no Matches.
_Avoid_: Unmatched entity, orphan

**Distractor**:
A Mention that describes a business with no Reference Record. Every Distractor must stay unmatched.
_Avoid_: Orphan, unmatched Mention, noise record

**Ghost**:
A Distractor made to look like a nearby business: it has a Reference Record's core name, a changed legal form or one added word, and a house number a little higher. A Ghost can have several Mentions of its own.
_Avoid_: Sibling, near-duplicate, look-alike

**Candidate**:
A Mention that blocking puts forward as a possible Match for a Reference Record, before the matching model decides.
_Avoid_: Candidate pair, neighbour

**Blocking**:
The step that picks a small set of Candidates for each Reference Record, so the model never compares every Mention with every Reference Record.
_Avoid_: Indexing, retrieval, pre-filtering

### Scoring

**Entity Score**:
The F0.5 score of one Reference Record's predicted Matches against its true Matches. A Singleton scores 1.0 for an empty prediction and 0.0 otherwise.

**Leaderboard Score**:
The mean Entity Score over all Reference Records in the evaluation set.
_Avoid_: Accuracy, F1
