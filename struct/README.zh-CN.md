# MobileMem-Struct

[English](README.md)

MobileMem-Struct 面向模拟端侧设备的结构化数据，评估记忆系统完成检索、推理和个性化问答的整体能力。

## 安装

使用 Python 3.10 或更高版本，在 `.env` 中填写兼容 OpenAI 接口的 Judge 配置，并在 `MobileMem/struct` 下完成一次配置：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

## 数据

**数据下载**

所有命令默认从 `MobileMem/struct` 执行。使用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 下载 [MobileMem 数据集](https://huggingface.co/datasets/zjunlp/MobileMem)中的 Struct 数据：

```bash
hf download zjunlp/MobileMem \
  --type dataset \
  --include "struct/*" \
  --local-dir ../data
```

下载后，代码和数据应保持为相邻目录，不需要把数据复制进代码目录：

```text
MobileMem/
├── struct/                  # 本目录；在这里运行命令
└── data/struct/persona01/   # persona01 的评测数据
```

**数据情况**

每个用户采用相同的目录结构：

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

| 数据源 | 内容 |
| --- | --- |
| `event` | 用户画像、人物关系和事件经历 |
| 六类 `batch.json` | 常见手机应用中的结构化记录 |
| `image.zip`、`screen`、`video` | 常见手机中的多模态数据，使用图片证据前请解压 `image.zip`。 |
| `case.csv` | 评测参考，覆盖记忆养成、记忆洞察、安全边界和场景感知四类能力。 |

## 数据生成（可选）

数据由大模型辅助生成，并在生成前后经过了**大量的**人工设计、清洗、审核和修改。

我们在 [`construct/`](construct/) 中提供了生成与审核流程的示例 Skills，但并**不作为**具备可复现性的代码文件。

可以参考的流程为，先生成并审核结构化证据，再调用 Omni 生成图片，最后生成并审核评测用例：

| 阶段 | 示例资源 |
| --- | --- |
| 生成结构化证据 | [`data-construct`](construct/data-construct/SKILL.md) |
| 审核结构化证据 | [`data-review`](construct/data-review/SKILL.md) |
| 生成图片 | 复用 [`omni`](../omni/) 图片生成代码 |
| 生成评测用例 | [`case-construct`](construct/case-construct/SKILL.md) |
| 审核评测用例 | [`case-review`](construct/case-review/SKILL.md) |

> Struct 不重复提供图片生成代码；该环节直接使用 `event/user.json` 作为 Omni 的输入。

## 评测

评测对象是完整的 Agent，可以包括Backbone、检索、工具、提示词和最终回答等等。Benchmark 跑出分数只用两步。

**推理**

选择一个用户目录，例如 `../data/struct/persona01`。你自己的推理驱动程序从其中的 `case.csv` 逐条读取 `Query编号` 和 `Query`，把 Query 交给被测系统；被测系统检索同一用户目录中的设备数据并作答。所有轮次按顺序写入一个 OpenAI 风格的 JSONL 轨迹，每行是一个 JSON 对象，每轮应包含：

```text
用户 Query → Agent 工具调用 → 工具召回结果 → Agent 最终回答
```

每条召回记录必须保留 `evidence_id`，评测器据此计算精确证据召回率。最小格式示例如下：

```jsonl
{"role":"user","content":"那笔早餐多少钱？"}
{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_memory","arguments":"{\"query\":\"早餐\"}"}}]}
{"role":"tool","tool_call_id":"call_1","name":"search_memory","content":"{\"results\":[{\"evidence_id\":\"a1b2c3d4e5f6\",\"amount\":12}]}"}
{"role":"assistant","content":"12 元。"}
```

`user.content` 应保留 `case.csv` 中的原始 Query，工具调用与工具结果通过相同的 `id` / `tool_call_id` 配对。任何能够生成上述轨迹的 Agent 或记忆系统都可以参加评测。假设你的推理入口叫 `your_agent_runner`，参考命令为：

```bash
mkdir -p outputs/persona01
python3 -m your_agent_runner \
  --cases ../data/struct/persona01/case.csv \
  --evidence-root ../data/struct/persona01 \
  --output outputs/persona01/trace.jsonl
```

这里的参数名可以按你的程序调整，但三项职责应保持不变：

| 参数 | 作用 |
| --- | --- |
| `--cases` | 用例文件；驱动程序只把 `Query编号` 和 `Query` 交给被测系统 |
| `--evidence-root` | 当前用户的设备数据目录，也是被测系统允许检索的数据范围 |
| `--output` | 汇总全部轮次的 JSONL 轨迹输出路径 |

**防止金标泄漏** 

被测系统在推理时**只能**使用当前 Query 和该用户目录中的设备证据。**不得**读取或使用 `case.csv` 中的 `evidence_ids`、GT、标准答案、得分点、扣分点和通过阈值，也**不得**使用既有报告、Judge 反馈或隐藏索引。工具结果中的 `evidence_id` 必须来自系统真实召回，不能手工注入金标 ID。推理驱动程序可以读取完整 `case.csv`，但必须把这些金标字段隔离在被测系统之外。

只是想先体验 Benchmark，可以直接使用下文的[极简 Agent 示例](#极简-agent-示例可选)。

**评估**

```bash
python3 -m eval batch \
  --tasks ../data/struct/persona01/case.csv \
  --log outputs/persona01/trace.jsonl \
  --evidence-root ../data/struct/persona01 \
  --out outputs/persona01/report.json
```

这一步仍在 `MobileMem/struct` 下运行，并使用推理阶段的同一份 `persona01` 数据：

| 参数 | 作用 |
| --- | --- |
| `batch` | 评估 `case.csv` 中的全部用例 |
| `--tasks` | 官方用例文件，包含 Query、金标与逐题评分规则 |
| `--log` | 推理阶段生成的 JSONL 轨迹 |
| `--evidence-root` | 对应用户的数据目录，用于校验证据并计算精确证据召回率 |
| `--out` | JSON 报告路径，同时也是断点文件 |
| `--overwrite` | 可选；忽略已有断点并从头评估 |

评测器会将每条 Query 与轨迹匹配，按照对应规则评分，并生成一个 `report.json`。其中包含逐题得分与通过情况、总体得分与通过率、精确证据召回率和能力维度汇总。

检索类 GT 等权。每题最终得分为：

```text
题目总分 = 得分点合计 - 扣分合计
```

证据召回率与回答质量分开统计，因此漏召不会被重复处罚。`report.json` 每完成一题便会更新，同时作为断点；重复执行同一命令即可续跑，添加 `--overwrite` 可重新开始。

## 极简 Agent 示例（可选）

[`example/minimal_agent.py`](example/minimal_agent.py) 的目的，是让 Benchmark 使用者用最少配置完整体验一次评测流程。它只是接入示例，不是基线：检索采用简单的字符重合，也不会读取 GT、标准答案、评分规则或金标证据 ID。

直接连续执行两步：

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

两个结果统一放在 `outputs/persona01/`：`trace.jsonl` 是 Agent 运行轨迹，`report.json` 是评测结果。

## 许可证

本赛道沿用仓库根目录的 [MIT License](../LICENSE)。
