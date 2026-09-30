---
name: data-review
description: Audit or repair MobileMem Struct evidence data across event, bill, calendar, document, note, todo, voice, screen, and video sources. Use for schema, provenance, temporal, realism, or cross-record consistency checks; exclude image assets and benchmark cases.
---

# Data Review

Audit one persona's active evidence corpus independently of benchmark cases.
Follow explicit user requirements first, then any project policy supplied at
runtime, then the public schema, cleaned data, and this skill. Do not inspect or
modify `image`, `image.zip`, `case.csv`, `case.md`, or case-construction Skills.

## Publication Boundary

Treat this file as public. Do not add credentials, private service endpoints,
absolute workstation paths, employee or customer information, real user
records, or unpublished acceptance criteria. Report such material if found;
never reproduce it in examples. Apply confidential release rules only when the
operator supplies them separately at runtime.

When the sibling `data-construct` Skill is available, read
it first and treat it as the construction contract. For report-only requests,
make no changes. For repair requests, preserve valid records and stable IDs and
change only verified defects.

## Review Scope

Inspect only:

```text
event/user.json
bill/batch.json
calendar/batch.json
document/batch.json
note/batch.json
todo/batch.json
voice/batch.json
screen/*.html
video/description.json
```

Ignore archives, `memory_data.json`, field-cleaning variants, generation
intermediates, images, and cases even if present.

## Review Order

1. Discover the actual files and infer the current schema from the cleaned
   target and its sibling personas.
2. Validate `event/user.json` as the persona, relationship, and event spine.
3. Validate each source independently for structure and app-native realism.
4. Resolve every declared event reference and compare the derived record with
   the exact source event or `*_info` payload.
5. Review trajectory projections against the profile without mistaking
   plausibility for sourced fact.
6. Perform global ID, timeline, relationship, duplication, and distribution
   checks.
7. After any repair, rerun the complete review rather than checking only edited
   records.

## Mechanical Checks

- Every required JSON file parses; six batch files and the video file are arrays,
  while the event file is one object.
- Every active evidence ID is lowercase 12-character hexadecimal and globally
  unique within the persona, including addressable event/profile records,
  batches, screen pages, and video rows.
- Screen filenames are evidence IDs and every file is readable UTF-8 HTML.
- App records contain the seven shared fields and the correct `data_type`.
- App `user_id` matches `user_NN`; event `user_id` matches the numeric persona
  suffix used by the cleaned schema.
- Arrays, booleans, numeric values, nulls, and timestamp strings have the same
  types as the current cleaned records.
- `created_at` parses. Bill epoch milliseconds agree with the intended
  transaction/create time and currency/amount are valid.
- Calendar start/end times parse and `end_time >= start_time`.
- Todo due/completion times agree with `is_completed`; incomplete items do not
  claim a completion time.
- A non-null voice duration is positive and compatible with transcript length;
  preserve null only where the current cleaned schema explicitly permits it.
- Video rows contain `id`, `video_name`, `text`, and `创建时间`.

Use the repository's active evidence loader when available as a final smoke test.
Do not add a bundled validation script unless the user separately requests one.

## Provenance Checks

For a direct event projection:

- the referenced event exists;
- the date, participants, location, amount, counterparty, status, and activity do
  not contradict the event or its structured `*_info`;
- `tags` and `time_references` identify the correct source using the target
  dataset's existing convention;
- paraphrasing does not introduce new facts.

For a trajectory projection:

- it is marked as projection rather than direct event evidence;
- it fits the person's role, location, finances, habits, relationships, and time
  window;
- it does not create a new stable relationship or consequential personal fact;
- repeated routines vary naturally in time, wording, merchant, action, and
  outcome where appropriate.

Treat top-level profile and graph objects as generation context. Treat nested
`Persona_Profile` and `Events` objects carrying valid 12-character IDs as active
`persona_profile` and `persona_event` evidence, respectively.

## Semantic Checks by Source

- **bill:** verify transaction direction, amount, merchant/product coherence,
  payment channel, status, duplicates, and unrealistic amount patterns.
- **calendar:** verify feasible duration, attendance, place, travel, overlap, and
  whether the item is truly schedulable.
- **todo:** verify one actionable intent, meaningful deadline, priority, and
  completion semantics; reject calendar duplicates with no separate action.
- **note:** reject essay-like prose, empty boilerplate, and unsupported facts.
- **document:** reject thin notes disguised as documents, empty outlines,
  repeated skeletons, and long unsupported narratives.
- **voice:** reject written-report style, implausible duration, repetitive tone,
  and metadata that contradicts the transcript.
- **screen:** verify that the page, title, URL, and captured text are real and
  mutually consistent. Topic relevance does not make page content a personal
  event.
- **video:** verify unique ID, timestamp, title/body agreement, persona relevance,
  and absence of unsupported personal outcomes.

## Cross-Record Checks

- Check all active sources for exact and near duplicates.
- Flag one event expanded into several records that serve the same mobile intent.
- Allow cross-source companions only when each represents a distinct action or
  artifact.
- Check people against `Social_Graph` and event participants; distinguish named
  merchants, public figures, and content creators from personal relationships.
- Check month and type distributions for obvious mechanical regularity or severe
  imbalance, but do not enforce a fixed count or ratio.
- Check that business times form a feasible chronology and that `created_at`
  correctly represents visibility. A future calendar or todo deadline is valid
  when its record was already created.
- Flag contradictory versions of the same amount, state, place, relationship, or
  event outcome unless the records clearly represent an update over time.

## Repair Policy

Automatically repair only deterministic defects whose intended value is already
present elsewhere in the same record or its direct source, such as an obvious
type error, exact duplicate, or recoverable timestamp mismatch. Do not silently
rewrite narrative meaning, invent missing sources, smooth distributions, or
change IDs already used downstream. Report uncertain semantic issues for human
decision.

## Deliverable

Report files reviewed or changed, counts by source, blocking errors, warnings,
safe fixes, provenance coverage, and unresolved decisions. State separately
whether mechanical validation and semantic review passed.
