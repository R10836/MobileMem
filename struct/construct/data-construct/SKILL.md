---
name: data-construct
description: Construct or extend MobileMem Struct evidence data from a persona/event seed across bill, calendar, document, note, todo, voice, screen, and video sources. Use for evidence-corpus generation or substantial evidence rewrites; exclude image assets and benchmark cases.
---

# Data Construction

Construct synthetic, internally consistent device evidence in the current
MobileMem Struct layout. Follow explicit user requirements first, then any
project policy supplied at runtime, then the public schema and this skill. Do
not modify `image`, `image.zip`, `case.csv`, `case.md`, or any case-construction
Skill.

## Publication Boundary

This is a public, reusable workflow. Never copy credentials, private endpoints,
absolute workstation paths, employee or customer information, real user
records, or unpublished project policy into this Skill. Use relative paths,
schema placeholders, and synthetic examples. Keep release-specific quotas,
generation targets, and internal acceptance rules in a separately supplied
private policy.

## Active Output Scope

Work only with these paths inside one `personaNN/` directory:

```text
event/user.json
bill/batch.json
calendar/batch.json
document/batch.json
note/batch.json
todo/batch.json
voice/batch.json
screen/<evidence_id>.html
video/description.json
```

All listed files are active inputs to the current evidence catalog.
`event/user.json` has two roles: its top-level profile and graph define the
persona world, while addressable records under `Persona_Profile` and `Events`
are evidence. Do not create legacy `memory_data.json`, `batch_con.json`,
`batch_rad.json`, archive packages, canonical-context files, generation configs,
distribution plans, or queries.

## Evidence Lineage

Treat `event/user.json` as the world model, not as a literal parent record for
every item. Keep these provenance classes distinct:

- **Direct event projection:** an app record represents an explicit event or an
  exact `*_info` payload. Preserve its people, date, amount, status, place, and
  meaning. Use the target persona's existing source-tag convention and include
  the event reference in `time_references` when available.
- **Trajectory projection:** a plausible routine record derived from the
  profile, role, habits, relationships, and time window. Mark it as a projection;
  never present it as an explicit source event.
- **Verified external capture:** a real screen page selected because it fits the
  persona or an event. Retrieved page text is external content, not a fact about
  what happened to the user.
- **Media description:** a video description grounded in an event, interest, or
  plausible mobile activity without claiming unsupported personal outcomes.

Projection may add ordinary synthetic activity, but must not invent stable
identity, relationships, diagnoses, major assets or debts, crimes, credentials,
or other consequential facts absent from the seed.

## Construction Workflow

1. Inspect the target persona directory and current sibling files before
   choosing fields, time formats, or labels.
2. Read `event/user.json` completely. If building a new persona from an explicit
   brief, create this file first and validate it before generating app records.
3. Build an internal ledger of stable profile facts, allowed people and
   organizations, important dates, event anchors, exact money fields, and
   unresolved conflicts. The ledger need not be saved.
4. Plan coverage by month, life domain, and source type. Let the persona drive
   counts and proportions; do not impose fixed totals or uniformity.
5. Give each explicit event one primary app representation. Add companion
   records only for distinct mobile intents, such as an appointment, its payment,
   and a genuinely separate preparation task.
6. Add routine projections to make the year feel lived-in without drowning the
   explicit event spine in repetitive filler.
7. Generate the active files directly, then run the evidence-review Skill before
   delivery.

## Event Seed Contract

Match the cleaned data's top-level structure:

```text
user_id, role_identity, Basic_Profile, Init_State, Important_Dates,
Social_Graph, Persona_Profile, Events
```

`Persona_Profile` is a list of flattened, retrievable profile facts. Every item
has a 12-character `id`, `data_type=persona_profile`, `profile_type`, a title or
content representation, and only the fields needed by that profile fact.

Every item under `Events` has a 12-character `id`,
`data_type=persona_event`, and the event fields `event_id`, `event_name`,
`event_start_time`, `event_end_time`, `duration_type`, `participants`,
`description`, `importance`, and `additional_info`, plus only the applicable
`*_info` objects. Numeric `event_id` values are unique within the persona.
Participants and relationship claims must agree with the profile and social
graph. Explicit structured fields override narrative paraphrases when they
conflict.

In the cleaned corpus, `event.user_id` is the numeric persona suffix while app
records use `user_NN`. Preserve that mapping unless the current schema changes.

## Structured Record Contract

Every record in the six batch files has these shared fields:

```text
id, user_id, data_type, created_at, entities, time_references, tags
```

- `id` is a lowercase 12-character hexadecimal value, unique across all active
  evidence for that persona. Once referenced, it is immutable.
- `created_at` is when the record became visible, in the local ISO-style format
  used by the target data. Type-specific timestamps describe business time.
- `entities`, `time_references`, and `tags` contain only useful retrieval anchors.
- Direct projections carry an event-source tag; routine records carry the
  established trajectory-projection tag. Never attach a false event reference.
- Use the exact `data_type`: `bill`, `calendar`, `document`, `note`, `todo`, or
  `voice_memo`.

Preserve the required type-specific fields found in the cleaned data:

- **bill:** transaction direction, amount, currency, merchant, product, category,
  payment source, transaction/create epoch milliseconds, and status fields.
- **calendar:** title, description, start/end time, attendees, locations, topic,
  and time category.
- **todo:** one actionable task with title/content/description, due date,
  completion state/time, priority, locations, topic, and time category.
- **note:** a compact title and phone-note body with a relevant topic.
- **document:** a title and substantively structured body whose length is
  justified by the artifact, plus topic and format.
- **voice:** a short spoken transcript with plausible scene, duration,
  fragmentation, filler-word, location, and noise metadata.

Use optional legacy fields only when the target file already uses them and their
values are meaningful. Do not normalize open topic/category labels merely for
cosmetic consistency.

## Type Realism

- Bills must use plausible merchants, amounts, channels, categories, and
  statuses. Exact `money_info` values remain exact. Epoch and readable times must
  agree.
- Calendar entries represent scheduled blocks, not generic intentions. Require
  `end_time >= start_time` and avoid impossible overlaps.
- Todos use one concrete verb and a coherent deadline. Completed items require a
  completion time no earlier than creation.
- Notes look quickly typed: fragments, lists, or compact observations are valid.
- Documents justify being longer than notes and contain real usable content, not
  meta-descriptions or empty outlines.
- Voice memos sound spoken. Duration, transcript length, noise, and scene must be
  mutually plausible.

## Screen and Video

For `screen`, select a concrete topic from the persona or event context, retrieve
an existing Chinese Wikipedia page, and save a faithful UTF-8 HTML snapshot as
`screen/<12-hex-id>.html`. Preserve the real title, source URL, summary, and page
text. Do not invent a page or rewrite external claims as personal history. If
retrieval cannot be verified, omit the candidate and report it.

`video/description.json` is a JSON array. Each record contains exactly the
current cleaned schema's `id`, `video_name`, `text`, and `创建时间`. Keep the text
specific enough to be retrievable and consistent with the persona, while not
claiming unsupported actions or outcomes.

## Global Constraints

- Keep all records inside the requested generation window except genuinely
  future calendar or todo business times stored by an already-created record.
- Preserve chronological feasibility, travel feasibility, relationship
  semantics, and completed/pending state consistency.
- Do not duplicate one fact across apps with only wording changes.
- Do not manufacture records merely to fill a type quota.
- Avoid repeated templates, round-number-heavy bills, identical monthly rhythms,
  and generic titles that could belong to any persona.
- Keep synthetic people and organizations internally fictional and do not add
  real private information.

## Delivery

Return the active files created or changed, counts by source type, a provenance
summary, dropped or revised candidate reasons, and unresolved risks. Do not
claim completion until the evidence review passes.
