# MobileMem-Struct

[简体中文](README.zh-CN.md)

MobileMem-Struct uses simulated structured data from on-device applications to
evaluate memory-system retrieval, reasoning, and personalized question
answering.

## Setup

Use Python 3.10 or later. From `MobileMem/struct`, create the environment once,
then fill `.env` with an OpenAI-compatible Judge configuration:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

## Data

**Download**

All commands below assume the current directory is `MobileMem/struct`. Use the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to
download the Struct data from the
[MobileMem dataset](https://huggingface.co/datasets/zjunlp/MobileMem):

```bash
hf download zjunlp/MobileMem \
  --type dataset \
  --include "struct/*" \
  --local-dir ../data
```

The code and data should now be neighboring directories; there is no need to
copy the data into the code directory:

```text
MobileMem/
├── struct/                  # this directory; run commands here
└── data/struct/persona01/   # evaluation data for persona01
```

**Layout**

Every persona follows the same layout:

```text
persona01/
├── event/user.json
├── bill/batch.json
├── calendar/batch.json
├── document/batch.json
├── note/batch.json
├── todo/batch.json
├── voice/batch.json
├── image.zip
├── screen/<evidence_id>.html
├── video/description.json
└── case.csv
```

| Source | Contents |
| --- | --- |
| `event` | Persona profile, relationships, and event history |
| Six `batch.json` files | Structured records from common mobile apps |
| `image.zip`, `screen`, `video` | Common multimodal phone data; extract `image.zip` before using image evidence |
| `case.csv` | Evaluation cases across memory evolution, memory insight, safety boundaries, and context awareness |

## Data Construction (Optional)

The data was generated with LLM assistance and underwent **substantial** human
design, cleaning, review, and revision both before and after generation.

[`construct/`](construct/) provides example Skills for the generation and
review workflow. They are **not** intended to be a reproducible data pipeline.

The reference flow is to generate and review structured evidence first, call
Omni to generate images, and then generate and review the evaluation cases:

| Stage | Example resource |
| --- | --- |
| Generate structured evidence | [`data-construct`](construct/data-construct/SKILL.md) |
| Review structured evidence | [`data-review`](construct/data-review/SKILL.md) |
| Generate images | Reuse the [`omni`](../omni/) image pipeline |
| Generate evaluation cases | [`case-construct`](construct/case-construct/SKILL.md) |
| Review evaluation cases | [`case-review`](construct/case-review/SKILL.md) |

> Struct does not duplicate the image-generation code. This stage directly uses
> `event/user.json` as input to Omni.

## Evaluation

The evaluated target is the complete Agent, which may include its backbone,
retrieval, tools, prompts, and final answer. Running the benchmark takes two
steps.

**Infer**

Choose one persona directory, such as `../data/struct/persona01`. Your inference
driver reads `Query编号` and `Query` from its `case.csv`, sends each Query to the
system under test, and lets the system retrieve device data from that same
persona directory before answering. Write all turns in order to one
OpenAI-style JSONL trace, with one JSON object per line. Each turn should contain:

```text
user Query → Agent tool call → retrieved tool result → Agent final answer
```

Keep the real `evidence_id` in every retrieved record so the evaluator can
measure exact evidence recall. A minimal format example is shown below:

```jsonl
{"role":"user","content":"How much was that breakfast?"}
{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_memory","arguments":"{\"query\":\"breakfast\"}"}}]}
{"role":"tool","tool_call_id":"call_1","name":"search_memory","content":"{\"results\":[{\"evidence_id\":\"a1b2c3d4e5f6\",\"amount\":12}]}"}
{"role":"assistant","content":"It was 12."}
```

Keep the original Query text from `case.csv` in `user.content`, and pair each
tool call and result with the same `id` / `tool_call_id`. Any Agent or memory
system that can produce this trace can be evaluated. If your inference entry
point is named `your_agent_runner`, a reference command is:

```bash
mkdir -p outputs/persona01
python3 -m your_agent_runner \
  --cases ../data/struct/persona01/case.csv \
  --evidence-root ../data/struct/persona01 \
  --output outputs/persona01/trace.jsonl
```

Your runner may use different option names, but these three responsibilities
should remain unchanged:

| Argument | Purpose |
| --- | --- |
| `--cases` | Case file; the driver exposes only `Query编号` and `Query` to the system under test |
| `--evidence-root` | Persona device-data directory and the only evidence scope available to the system |
| `--output` | One JSONL trace containing all evaluation turns |

**Prevent gold leakage**

During inference, the system under test may **only** use the current Query and
the device evidence under the selected persona. It **must not** read or use
`evidence_ids`, GT, gold answers, scoring checkpoints, deductions, or pass
thresholds from `case.csv`; previous reports, Judge feedback, and hidden indexes
are also prohibited. Every `evidence_id` in tool results must come from actual
retrieval rather than manual gold-ID injection. The inference driver may read
the complete `case.csv`, but it must keep all gold fields isolated from the
tested system.

To try the benchmark before integrating your own system, use the
[Minimal Agent Example](#minimal-agent-example-optional) below.

**Eval**

```bash
python3 -m eval batch \
  --tasks ../data/struct/persona01/case.csv \
  --log outputs/persona01/trace.jsonl \
  --evidence-root ../data/struct/persona01 \
  --out outputs/persona01/report.json
```

Run this command from `MobileMem/struct` and use the same `persona01` data as in
the inference step:

| Argument | Purpose |
| --- | --- |
| `batch` | Evaluate every case in `case.csv` |
| `--tasks` | Official case file containing Queries, gold data, and per-case rubrics |
| `--log` | JSONL trace produced during inference |
| `--evidence-root` | Matching persona directory, used to validate evidence and calculate exact evidence recall |
| `--out` | JSON report path and resume checkpoint |
| `--overwrite` | Optional; ignore an existing checkpoint and restart evaluation |

The evaluator matches each Query to its trace, applies the corresponding
rubric, and writes `report.json`. The report contains per-case scores and
pass/fail results, overall score and pass rate, exact evidence recall, and
capability aggregates.

Retrieval GT items are equally weighted. The final case score is:

```text
task score = earned checkpoint points - deductions
```

Evidence recall is reported separately from answer quality, so a missed record
is not penalized twice. `report.json` is updated after every case and also acts
as a checkpoint. Run the same command to resume, or add `--overwrite` to start
over.

## Minimal Agent Example (Optional)

[`example/minimal_agent.py`](example/minimal_agent.py) lets benchmark users run
the complete evaluation workflow with minimal setup. It is an integration demo,
not a baseline: retrieval uses simple character overlap, and the script never
reads GT, gold answers, scoring rules, or gold evidence IDs.

Run the two steps back to back:

```bash
python3 -m example.minimal_agent \
  --cases ../data/struct/persona01/case.csv \
  --evidence-root ../data/struct/persona01 \
  --output outputs/persona01/trace.jsonl

python3 -m eval batch \
  --tasks ../data/struct/persona01/case.csv \
  --log outputs/persona01/trace.jsonl \
  --evidence-root ../data/struct/persona01 \
  --out outputs/persona01/report.json
```

Both files stay under `outputs/persona01/`: `trace.jsonl` is the Agent run and
`report.json` is the evaluation result.

## License

This track uses the repository-level [MIT License](../LICENSE).
