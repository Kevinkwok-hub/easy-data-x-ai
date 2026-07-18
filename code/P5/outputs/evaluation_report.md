# Knowledge Agent 评测报告

评测样本数：30

## 指标分组

### 数据层

| 指标 | 数值 |
| --- | ---: |
| 知识覆盖率 | 82.76% |
| 检索命中率 | 95.83% |
| 检索故障率 | 3.33% |

### 模型层

| 指标 | 数值 |
| --- | ---: |
| 回答准确率 | 96.67% |
| 幻觉率 | 3.33% |

### 业务层

| 指标 | 数值 |
| --- | ---: |
| 任务成功率 | 73.33% |
| 转人工率 | 23.33% |
| 行为一致率 | 100.00% |

### 运行层

| 指标 | 数值 |
| --- | ---: |
| 工具成功率 | 80.00% |
| 平均延迟（毫秒） | 108.67 |
| 平均 Token 使用量 | 23.53 |

## 样本明细

| 任务 | 类别 | 命中文档 | 答案正确 | 行为一致 | 差异字段 | 成功 | 转人工 | 幻觉 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| general-001 | general | kb-general-refund | 是 | 是 | - | 是 | 否 | 否 |
| general-002 | general | kb-general-shipping | 是 | 是 | - | 是 | 否 | 否 |
| general-003 | general | kb-general-login | 是 | 是 | - | 是 | 否 | 否 |
| general-004 | general | kb-general-invoice | 是 | 是 | - | 是 | 否 | 否 |
| general-005 | general | kb-general-password | 是 | 是 | - | 是 | 否 | 否 |
| general-006 | general | kb-general-hours | 是 | 是 | - | 是 | 否 | 否 |
| general-007 | general | kb-general-security | 是 | 是 | - | 是 | 否 | 否 |
| general-008 | general | - | 是 | 是 | - | 否 | 是 | 否 |
| product-001 | product | kb-product-x100-v1.2 | 是 | 是 | - | 是 | 否 | 否 |
| product-002 | product | kb-product-x100-v2.0 | 是 | 是 | - | 是 | 否 | 否 |
| product-003 | product | kb-product-promax-2025 | 是 | 是 | - | 是 | 否 | 否 |
| product-004 | product | kb-product-databridge-3.4 | 是 | 是 | - | 是 | 否 | 否 |
| product-005 | product | kb-product-litebot-1.0 | 是 | 是 | - | 是 | 否 | 否 |
| product-006 | product | kb-product-edgenode-a2 | 是 | 是 | - | 是 | 否 | 否 |
| error-001 | error | kb-error-e1001 | 是 | 是 | - | 是 | 否 | 否 |
| error-002 | error | kb-error-e2002 | 是 | 是 | - | 是 | 否 | 否 |
| error-003 | error | kb-error-e3003 | 是 | 是 | - | 是 | 否 | 否 |
| error-004 | error | kb-error-e4004 | 是 | 是 | - | 是 | 否 | 否 |
| error-005 | error | kb-error-e5005 | 是 | 是 | - | 是 | 否 | 否 |
| tool-001 | tool | kb-tool-reset-api-key | 是 | 是 | - | 是 | 否 | 否 |
| tool-002 | tool | kb-tool-check-order | 是 | 是 | - | 是 | 否 | 否 |
| tool-003 | tool | kb-tool-create-ticket | 是 | 是 | - | 是 | 否 | 否 |
| tool-004 | tool | kb-tool-schedule-callback | 是 | 是 | - | 是 | 否 | 否 |
| tool-005 | tool | kb-tool-export-usage | 是 | 是 | - | 否 | 是 | 否 |
| missing-001 | missing | - | 是 | 是 | - | 否 | 是 | 否 |
| missing-002 | missing | - | 是 | 是 | - | 否 | 是 | 否 |
| missing-003 | missing | - | 是 | 是 | - | 否 | 是 | 否 |
| missing-004 | missing | - | 是 | 是 | - | 否 | 是 | 否 |
| exception-001 | exception | - | 是 | 是 | - | 否 | 是 | 否 |
| exception-002 | exception | - | 否 | 是 | - | 否 | 否 | 是 |
