# P5：从离线评测到 Agent ROI

学习路径：`离线评测 → Agent API → Prometheus Metrics → Grafana Dashboard → LangSmith Trace → ROI`。
运行层是三层质量框架的工程补充，不是第四个质量归因层；它关注系统能否稳定地把三层能力交付出来。

所有 `python -m` 命令都必须在 `code/P5` 目录运行。最短完整实验如下：

```bash
cd code/P5
pip install -r requirements.txt
python -m app.evaluation.run_eval
# 终端 A
docker compose up --build
# 终端 B（API 启动后）
python -m app.observability.generate_traffic --count 100
```

终端 A 周期性出现 `/health`、`/metrics` 表示 Compose 已进入持续运行状态，无需等待额外的“构建完成”。
依次查看 Agent `/metrics`、Prometheus、Grafana；结束时在终端 A 按 `Ctrl+C`，再执行 `docker compose down`。

首期使用 YAML 示例快照，不依赖模型 API、Prometheus、LangSmith 或 Harbor。后续接入监控系统时，只需用真实的月任务量、转人工率、任务成功率、风险事件率和单任务成本替换配置值。

## 运行

要求 Python 3.10+。

```bash
cd code/P5
pip install -r requirements.txt
python -m app.roi.calculator --config config/roi_scenarios.yaml --output-dir outputs
```

命令生成：

```text
outputs/roi_report.md
outputs/roi_report.json
outputs/scenario_comparison.csv
```

这是一条**批量对比命令**：它会在一次运行中读取配置内的保守、基准、乐观三种情景，并同时写入三种报告。无需为每个情景分别执行命令。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## Agent API 与 Prometheus Metrics

阶段 3 增加了离线 Knowledge Agent API。默认仍然不调用真实 LLM，也不依赖外部数据库。

```bash
cd code/P5
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

可用接口：

```text
GET  /health
POST /ask
GET  /metrics
```

示例请求：

```bash
curl -sS -H 'Content-Type: application/json' \
  -d '{"query":"退款要在多久内提交？","expected_answer_contains":"7 天内"}' \
  http://localhost:8000/ask
curl http://localhost:8000/metrics
```

`/metrics` 暴露 `agent_requests_total`、`agent_tasks_total`、`agent_retrieval_hit_total`、
`agent_handoff_total`、`agent_tool_calls_total`、`agent_tool_errors_total` 等低基数指标。
Prometheus label 只使用 `agent_version`、`environment`、`status`、`tool_name`、`error_type`，
不会把用户问题、回答正文、用户 ID 或 trace ID 写入 label。

## LangSmith 可选 Trace

阶段 4 增加了可选 LangSmith Trace。默认不开启，未设置 API Key 时不会影响 Agent API
和 Prometheus Metrics。

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=你的 LangSmith Key
export LANGSMITH_PROJECT=easy-data-x-ai-p5
export AGENT_VERSION=mock-v1
export ENVIRONMENT=local
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

开启后，每次 `POST /ask` 会写入一个 `Agent Trace`，包含 `Retrieval Run`、
`Generation Run`、可选 `Tool Run` 和 `Final Response`。Trace metadata 包含
`agent_version`、`environment`、`task_id`、`category` 和 `evaluator_score`。
用户问题和回答正文只进入 LangSmith Trace 的 inputs/outputs，不进入 Prometheus label。

## Prometheus 与 Grafana Dashboard

阶段 5 增加 Docker Compose 监控栈，一条命令启动 Agent、Prometheus 和 Grafana：

```bash
cd code/P5
GRAFANA_ADMIN_PASSWORD='请替换为本地强密码' docker compose up --build
```

`GRAFANA_ADMIN_PASSWORD` 必须显式提供；可选设置 `GRAFANA_ADMIN_USER`，
未设置用户名时默认为 `admin`。匿名访问仍限制为只读 `Viewer`。

启动后访问：

```text
Agent health:  http://localhost:8000/health
Agent metrics: http://localhost:8000/metrics
Prometheus:    http://localhost:9090
Grafana:       http://localhost:3000
```

- Agent `/metrics` 查看 API 原始指标；Prometheus 查看抓取 Target 与 PromQL；Grafana 查看分层趋势。

Grafana 会自动加载 Prometheus datasource 和 `Agent Three-Layer Metrics Dashboard`，
不需要手动导入 JSON。刚启动时 Dashboard 可能没有数据；调用几次 `/ask` 后，
Prometheus 抓取到的请求、任务成功率、转人工率、检索命中率、工具成功率、Token 和成本指标会开始变化。

## Demo Traffic Generator

流量脚本固定轮转成功、漏召回、检索故障、知识缺失、幻觉、工具失败、业务转人工七类样本。
先启动 Agent API，再运行：

```bash
cd code/P5
python -m app.observability.generate_traffic --count 100
```

默认请求 `http://127.0.0.1:8000/ask`。如果 Agent 跑在其他地址，可以指定：

```bash
python -m app.observability.generate_traffic --count 100 --base-url http://localhost:8000
```

脚本会在请求失败、API 返回非 2xx、响应不是 JSON 或缺少标准 Agent 字段时返回明确错误；
成功运行后 `/metrics` 中的 counter 会增长，Grafana Dashboard 会在 Prometheus 抓取后看到变化。

### 流量发送后怎么看结果

`POST /ask HTTP/1.1\" 200 OK` 表示一条请求处理成功。流量完成后，在浏览器打开
`http://127.0.0.1:8000/metrics`，或执行：

```bash
curl -s http://127.0.0.1:8000/metrics | rg '^agent_'
```

指标会随请求累计；重启 Uvicorn 后从零开始。Grafana 图表需要先启动 Prometheus/Grafana
监控栈：`docker compose up --build`，然后访问 `http://localhost:3000`。

## Metrics 字典

所有 Prometheus 指标都至少带 `agent_version` 和 `environment` 两个低基数标签。请求状态类指标额外带
`status`，工具指标额外带 `tool_name`，错误指标额外带 `error_type`。

| 指标 | 层级 | 类型 | 含义 |
| --- | --- | --- | --- |
| `agent_requests_total` | 运行层 | Counter | Agent API 请求总数，按 `status` 区分成功或异常 |
| `agent_errors_total` | 运行层 | Counter | API 异常总数，按 `error_type` 区分错误类型 |
| `agent_request_duration_seconds` | 运行层 | Histogram | API 请求耗时，用于计算 P95 latency |
| `agent_token_usage_total` | 运行层 | Counter | Agent 返回的估算 token 使用量累计值 |
| `agent_cost_total` | 运行层 / ROI | Counter | Agent 估算可变成本累计值 |
| `agent_knowledge_available_total` | 数据层 | Counter | 本地知识库可支持回答的任务数 |
| `agent_knowledge_evaluated_total` | 数据层 | Counter | 排除检索故障后可判断知识是否存在的任务数 |
| `agent_retrieval_total` | 数据层 | Counter | 知识存在且检索结果可评测的任务数（含漏召回） |
| `agent_retrieval_hit_total` | 数据层 | Counter | 成功命中文档的任务数 |
| `agent_retrieval_errors_total` | 运行层 | Counter | 检索服务异常任务数 |
| `agent_answer_evaluated_total` | 模型层 | Counter | 有独立答案标注的任务数 |
| `agent_answer_correct_total` | 模型层 | Counter | 答案包含期望关键事实的任务数 |
| `agent_hallucination_total` | 模型层 | Counter | 被标记为幻觉的任务数 |
| `agent_tasks_total` | 业务层 | Counter | Agent 处理的任务总数 |
| `agent_task_success_total` | 业务层 | Counter | Agent 独立完成的任务数 |
| `agent_handoff_total` | 业务层 / ROI | Counter | 转人工任务数 |
| `agent_tool_calls_total` | 运行层 | Counter | 工具调用次数 |
| `agent_tool_errors_total` | 运行层 | Counter | 工具调用失败次数 |

常用 PromQL：

```text
任务成功率 = sum(increase(agent_task_success_total[5m])) / sum(increase(agent_tasks_total[5m]))
转人工率 = sum(increase(agent_handoff_total[5m])) / sum(increase(agent_tasks_total[5m]))
知识覆盖率 = sum(increase(agent_knowledge_available_total[5m])) / sum(increase(agent_knowledge_evaluated_total[5m]))
检索命中率 = sum(increase(agent_retrieval_hit_total[5m])) / sum(increase(agent_retrieval_total[5m]))
回答准确率 = sum(increase(agent_answer_correct_total[5m])) / sum(increase(agent_answer_evaluated_total[5m]))
回答评测覆盖率 = sum(increase(agent_answer_evaluated_total[5m])) / sum(increase(agent_tasks_total[5m]))
检索故障率 = sum(increase(agent_retrieval_errors_total[5m])) / sum(increase(agent_tasks_total[5m]))
幻觉率 = sum(increase(agent_hallucination_total[5m])) / sum(increase(agent_tasks_total[5m]))
工具成功率 = (sum(increase(agent_tool_calls_total[5m])) - sum(increase(agent_tool_errors_total[5m]))) / sum(increase(agent_tool_calls_total[5m]))
P95 延迟 = histogram_quantile(0.95, sum(rate(agent_request_duration_seconds_bucket[5m])) by (le))
```

无分母时 Dashboard 显示“暂无可评测数据”。`expected_answer_contains` 仅为课堂制造已标注流量；
生产准确率应来自抽样人工评测或独立 Judge，而不是普通调用方提供的标准答案。

## 和 ROI 的衔接

Evaluation、Prometheus 和 Grafana 负责回答“Agent 运行得怎么样”；ROI 计算器负责回答“这个效果值不值得投入”。

| ROI 输入 | 可来自的观测指标 | 说明 |
| --- | --- | --- |
| `monthly_task_count` | `agent_tasks_total` 的月度增量 | 月任务量 |
| `agent_handoff_rate` | `agent_handoff_total / agent_tasks_total` | 转人工率，影响人工节省 |
| `agent_task_success_rate` | `agent_task_success_total / agent_tasks_total` | 任务成功率，影响收入提升或业务价值 |
| `agent_variable_cost_per_task` | `agent_cost_total / agent_tasks_total` | 单任务可变成本 |
| 风险事件率 | 后续风险类 counter 或人工标注 | 当前阶段未单独实现风险事件，ROI 示例中仍用配置假设 |

知识覆盖率、检索命中率、回答准确率和幻觉率是诊断指标，不应直接各自折算成收益后相加。
它们应通过“转人工减少、成功任务增加、风险事件减少”进入 ROI。

## 常见问题

**缺少 `yaml` 或 `fastapi` 等依赖**

先安装依赖：

```bash
cd code/P5
pip install -r requirements.txt
```

如果你使用仓库根目录的 `.venv`，运行测试可用：

```bash
../../.venv/bin/python -m unittest discover -s tests -v
```

**8000、9090 或 3000 端口被占用**

本地 Uvicorn 与容器 Agent 默认都会占用 `8000`，不要同时启动。本地 API 可换端口：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
python -m app.observability.generate_traffic --count 100 --base-url http://localhost:8001
```

Docker Compose 端口需要修改 `docker-compose.yml` 中对应的 `ports`。

**没有 LangSmith API Key**

不影响默认 demo。保持 `LANGSMITH_TRACING=false` 或不设置 `LANGSMITH_API_KEY` 即可。

**Dashboard 没有数据**

先确认 Agent API 有流量：

```bash
python -m app.observability.generate_traffic --count 100
curl http://localhost:8000/metrics
```

再确认 Prometheus target 中 `p5-knowledge-agent` 为 `UP`。Grafana 面板依赖 Prometheus 抓取，
可能需要等待一个 scrape interval。

**Docker 镜像下载很慢**

这只影响阶段 5 的 Prometheus/Grafana Dashboard，不影响本地 Evaluation、Agent API、LangSmith no-op 和流量脚本。
可稍后重试：

```bash
docker pull prom/prometheus:v2.55.1
docker pull grafana/grafana:11.4.0
docker pull python:3.12-slim
docker compose up --build
```

## 配置口径

`roi_scenarios.yaml` 固定包含 `conservative`、`base`、`optimistic` 三种情景，每个情景包含：

* `metrics`：月任务量、转人工率、任务成功率、风险事件率、单任务可变成本；
* `costs.data`、`costs.model`、`costs.business`：数据、模型、业务三层的初始和月度投入；
* `business_baseline`：人工处理、任务成功和风险事件的基线，以及业务价值假设。

计算周期固定为 12 个月，币种由 `report.currency` 指定。Dashboard 成本是 CNY Mock 演示估算；
ROI 必须替换为包含模型、工具和基础设施的真实完整单任务成本。所有比例必须介于 `0` 与 `1`，金额不能为负，月任务量必须大于 `0`。

### 三种情景说明

示例配置不是只计算一个“默认场景”，而是用同一套成本和收益公式比较不同的运营预期：

| 情景 | 月任务量 | 转人工率 | 任务成功率 | 单任务成本 | 首年 ROI | 含义 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conservative`（保守） | 80 | 35% | 68% | 2.5 元 | -24.46% | 业务量低、转人工和模型调用成本较高 |
| `base`（基准） | 100 | 20% | 80% | 2 元 | 67.65% | 课程案例的预期运营水平 |
| `optimistic`（乐观） | 140 | 12% | 88% | 1.6 元 | 188.92% | 业务量增长，Agent 质量和调用成本持续改善 |

运行后优先查看 `outputs/scenario_comparison.csv` 或 `outputs/roi_report.md` 中的横向对比；`roi_report.json` 适合被其他程序继续读取。以上结果来自仓库内的示例假设，不应直接视为真实业务预测。

## 公式

```text
总成本（Total Cost, TotalCost）
  = 初始投入（Initial Cost, InitialCost）
  + 12 × [月固定成本（Monthly Fixed Cost, MonthlyFixedCost）
          + 月可变成本（Monthly Variable Cost, MonthlyVariableCost）]

总收益（Total Benefit, TotalBenefit）
  = 12 × [人力节省（Human Saving, HumanSaving）
          + 收入提升（Revenue Lift, RevenueLift）
          + 风险损失降低（Risk Reduction, RiskReduction）
          + 独立效率收益（Efficiency Gain, EfficiencyGain）]

净收益（Net Benefit, NetBenefit）= 总收益 - 总成本
投资回报率（Return on Investment, ROI）= 净收益 / 总成本 × 100%
```

| 中文术语 | 英文术语 | 报告字段 | 说明 |
| --- | --- | --- | --- |
| 总成本 | Total Cost | `total_cost` | 首年初始投入、固定成本和任务可变成本之和 |
| 总收益 | Total Benefit | `total_benefit` | 首年人力、收入、风险和独立效率收益之和 |
| 净收益 | Net Benefit | `net_benefit` | 总收益减去总成本 |
| 投资回报率 | Return on Investment | `roi_percent` | 净收益占总成本的比例 |
| 投资回收期 | Payback Period | `payback_period_months` | 初始投入被月净收益覆盖所需的月数 |
| 盈亏平衡任务量 | Break-even Monthly Task Volume | `break_even_monthly_task_volume` | 首年净收益为零所需的每月任务量 |

人力节省、收入提升和风险降低使用不同的业务假设。知识覆盖率、检索命中率和回答准确率是诊断指标，不能各自折算收益后直接相加；只有它们带来的人工处理、成功任务或风险事件变化才进入 ROI。
