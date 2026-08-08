# D3：Agentic RAG 实战

> Easy Data x AI 课程 · 术篇 · 第三期
>
> D1 你学会了 Tool Use——Agent 与外部数据之间的桥梁。D2 你搭建了数据层——桥对面的目的地。这一期，我们把桥和目的地连起来，构建一个真正能用你的数据回答问题的知识库助手。

## 开场：该把两件事连起来了

回顾一下你目前拥有的能力。

D1 给了你一个机制：Tool Use。模型不再只能“说话”，它可以声明“我需要调用某个工具”，你的代码去执行，结果传回来，模型继续推理。你跑通了一个完整的循环——定义工具、模型调用、处理结果。

D2 给了你一个基座：seekdb 数据层。你用三行代码建了一个支持向量搜索和全文搜索的知识库，体验了混合检索的效果，也直观看到了纯向量搜索在遇到精确匹配需求时的软肋。

现在的问题是：**这两样东西怎么接到一起？**

说白了就是一句话：让 Agent 通过 Tool Use 去调用 seekdb 的检索能力，拿到结果后基于真实数据回答用户的问题。这就是 Agentic RAG——Agent 自主决定“要不要查”、“查什么”、“查到的结果够不够用”，然后基于检索到的内容生成回答。

这一期有三个目标：

1. **跑通 Agentic RAG 的完整代码链路**——从用户提问到 Agent 回答，中间经历 Tool Use 调用 seekdb 检索，整个过程完整串通
2. **通过对比实验亲眼看到差距**——同一组查询，纯向量检索和混合检索的结果差距到底有多大；并用扩充场景的 RAGAS 四指标把「检索之后生成」也读清楚
3. **把 P2 的工程痛点逐一落地**——加入查询分析、策略路由、权限过滤、融合重排、引用校验、失败重试和 60 条离线评测

第二个目标是这节课的重头戏。不需要看论文，不需要听人讲道理——跑一次实验，数据替你说话。

如果你身边有人还在说“用向量数据库做 RAG 就够了”，这节课结束后你会有一组实验数据可以回应他——不是观点对观点，是数据对观点。

## 第一部分：知识库构建的基本流程

在构建 Agentic RAG 之前，我们需要先有一个“装满了知识的数据库”供 Agent 查询。知识库构建的完整流程是：

**文档解析 → 分块（Chunking） → 向量化（Embedding） → 存储**

简单解释一下每个环节做了什么：

- **文档解析**：把 PDF、Word、网页等格式的文档转成纯文本。你的原始知识可能散落在各种格式里，第一步是统一提取出文字内容
- **分块**：大模型的上下文窗口有限（F1 讲过），一篇几万字的文档没法整篇塞给模型。所以需要把长文档切成较小的段落（chunk），每个 chunk 是一个独立的检索单元
- **向量化**：用 Embedding 模型把每个 chunk 转成一个向量，用于后续的语义搜索
- **存储**：把 chunk 的文本内容、向量和元数据（来源、版本号、更新时间等）一起存入数据库

分块策略的选择（按固定长度切还是按语义边界切）对检索质量有直接影响，但这属于偏深的工程决策——本模块的延伸阅读会提供参考，主线不在这里展开。

**好消息是**：本模块提供了一个预处理好的示例数据集，你可以跳过文档解析和分块的环节，直接从“存入数据库”开始。这样你可以把精力集中在 Agentic RAG 的核心逻辑和对比实验上。

### 把示例数据集加载到 seekdb

我们的示例数据集模拟了一个技术产品的知识库——包含产品文档、错误码手册、版本发布说明、性能调优指南等内容。这些内容已经完成了分块处理，每条记录就是一个 chunk。

```python
from d3_1_ingest import build_knowledge_base, create_db_client

# get_or_create_collection + upsert 让重复运行保持幂等，
# 不会为了演示而删除读者已有的数据。
with create_db_client() as database:
    collection = build_knowledge_base(database)
    print(f"已写入或更新 {collection.count()} 个知识片段")
```

完整示例数据和元数据位于 `code/D3/data/knowledge_base.json`，由 `code/D3/rag_data.py` 统一加载；`d3_1_ingest.py` 只负责写入 seekdb。代码使用当前仓库统一的 seekdb 连接配置，不再复制一套容易漂移的旧 API。

数据就位了。接下来，我们让 Agent 用起来。

## 第二部分：Agentic RAG 的代码结构

这一步的本质是把 D1 和 D2 连接起来：Agent 通过 Tool Use（D1 的机制）将检索请求转交给 seekdb（D2 的数据层），检索结果传回 Agent 进行推理，生成最终回答。

整个流程用一张图来表示：

```
用户提问
    ↓
Agent（大模型）分析问题
    ↓
Agent 决定调用 search_knowledge_base 工具
    ↓
你的代码收到 tool_call，调用 seekdb 执行混合检索
    ↓
seekdb 返回检索结果（最相关的文档片段）
    ↓
检索结果作为 tool 消息传回 Agent
    ↓
Agent 基于检索到的真实内容生成最终回答
```

是不是看着很眼熟？这就是 D1 里 Tool Use 的五步循环，只不过这次“工具”不是一个模拟的函数，而是真实的知识库检索。

这里有一个关键的区别值得强调：传统 RAG 是一个固定流程——用户提问，系统自动检索，结果塞给模型，模型回答。整个过程像流水线一样从头到尾执行一遍。而 Agentic RAG 中，**Agent 自己决定要不要检索**。它会先分析用户的问题：这个问题我已经知道答案了吗？还是需要查知识库？查了之后结果够不够用？需不需要换个关键词再查一次？这种主动判断的能力，就是 “Agentic” 这个词的含义——Agent 不是被动执行流程，而是主动做决策。

### 核心代码：多工具、多轮的安全循环

```python
while True:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        return message.content or "模型未返回有效内容。"

    # 同一轮可能有多个工具调用，必须逐个执行并回传。
    messages.append(message)
    for tool_call in tool_calls:
        result = _tool_result(tool_call, search_fn)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
```

这里省略了参数 JSON 校验、未知工具处理、重复调用 ID 检查和最大轮数保护。它们都在 `code/D3/d3_2_agentic_rag.py` 的 `run_agent_loop()` 中，并有离线回归测试覆盖。不要只复制这一小段作为生产实现。

完整运行：

```bash
cd code
python D3/d3_1_ingest.py
python D3/d3_2_agentic_rag.py
```

你会看到这样的过程：Agent 收到用户问题，判断需要查知识库，通过 Tool Use 声明调用 `search_knowledge_base`，你的代码执行 seekdb 混合检索，结果返回给 Agent，Agent 基于检索到的 OB-4.2.1 兼容性文档生成一个准确的回答——引用了具体的版本号、兼容性关系和已知问题。

这就是 Agentic RAG 的核心链路。模型负责理解用户意图和组织语言，seekdb 负责提供准确的数据，Tool Use 负责把两者连接起来。**三者各司其职，缺一不可。**

![](https://raw.githubusercontent.com/datawhalechina/easy-data-x-ai/main/docs/public/images/dev/D3/01-agentic-rag-flow.png)

## 第三部分：对比实验——数据说话

到目前为止，你可能对“混合检索比纯向量检索好”这个判断已经有了直觉印象——D2 的简单示例给了你一些感觉。但直觉不够。这一节，我们用一组系统性的对比实验来验证这个判断。

实验设计很简单：**同一组查询，分别用纯向量检索和混合检索，对比 Top-1 结果是否命中了用户真正需要的内容。**

### 构建对比实验

```python
def vector_only(query, collection):
    return collection.query(
        query_texts=[query],
        n_results=3,
    )

def hybrid_with_keyword(query, keyword, collection):
    return collection.hybrid_search(
        query={"where_document": {"$contains": keyword}, "n_results": 5},
        knn={"query_texts": [query], "n_results": 5},
        rank={"rrf": {}},
        n_results=3,
    )
```

完整的五组对照及命中统计位于 `code/D3/d3_3_compare.py`。版本号包含点号时，示例会改用元数据过滤，避免把所有精确标识符都硬塞给全文分词器。

### 运行实验

我们特意选了三类最能体现差距的查询——都是真实场景中高频出现的类型：

```python
test_queries = [
    "OB-4.2.1版本的兼容性",          # 包含精确版本号
    "error code E-4012",             # 包含精确错误码
    "2024年Q3营收数据",               # 包含精确时间和数据类型
    "怎么优化数据库的查询性能",          # 纯语义查询（作为对照）
    "DBMS_HYBRID_SEARCH 函数怎么用",  # 包含精确函数名
]

for query in test_queries:
    compare_search(query)
```

### 实验结果

跑完上面的代码，你会得到一张类似这样的对比表：

| 查询 | 纯向量检索 Top-1 | 混合检索 Top-1 | 命中正确？ |
| --- | --- | --- | --- |
| “OB-4.2.1版本的兼容性” | OB-4.1.0 兼容性说明 ❌ | **OB-4.2.1 兼容性说明** ✅ | 向量 ❌ / 混合 ✅ |
| “error code E-4012” | E-4013 认证握手超时 ❌ | **E-4012 连接池耗尽** ✅ | 向量 ❌ / 混合 ✅ |
| “2024年Q3营收数据” | 2024年Q2营收数据 ❌ | **2024年Q3营收数据** ✅ | 向量 ❌ / 混合 ✅ |
| “怎么优化数据库的查询性能” | 并行查询优化指南 ✅ | 并行查询优化指南 ✅ | 向量 ✅ / 混合 ✅ |
| “DBMS_HYBRID_SEARCH 函数怎么用” | 索引设计最佳实践 ❌ | **DBMS_HYBRID_SEARCH 函数说明** ✅ | 向量 ❌ / 混合 ✅ |

仔细看这张表。

前三条查询都包含需要**精确匹配**的内容——版本号“4.2.1”、错误码“E-4012”、时间“Q3”。纯向量检索在这三条上全部失手：它返回的是语义上“差不多”的内容，而不是用户真正需要的那条。

- 用户问的是 4.2.1 的兼容性，向量检索返回了 4.1.0 的——因为在向量空间中“版本兼容性说明”这个语义概念几乎一样，模型分不清 4.2.1 和 4.1.0
- 用户问的是 E-4012，向量检索返回了 E-4013——D2 已经演示过这个问题，这里再次验证
- 用户问的是 Q3 的营收，向量检索返回了 Q2 的——“营收数据”的语义向量高度相似，Q2 和 Q3 在向量空间中几乎无差别

第四条查询是纯语义查询——“怎么优化数据库的查询性能”。这条没有任何需要精确匹配的关键词。两种方式都能正确命中。这说明混合检索并没有牺牲语义能力——它只是在语义的基础上**补上了精确匹配**。

第五条查询包含一个精确的函数名 `DBMS_HYBRID_SEARCH`。纯向量检索完全找不到它——因为这个函数名在语义空间中没有特殊含义，它就是一个“看起来像技术术语”的字符串。但全文搜索能精确匹配到它。

### 关键发现

这五条是用于观察差异的教学案例，不预设你的运行结果一定是 20% 对 100%。Embedding 模型、数据库版本和数据内容都会影响排名，请以 `d3_3_compare.py` 打印的实际 Top-1 与命中统计为准。

而且这不是我们刻意挑的“刁钻查询”。你回想一下自己日常工作中查文档的场景：查某个版本的功能、查某个错误码的解决方案、查某个季度的数据、查某个 API 的用法——**这些就是最常见的查询类型，而它们恰恰是纯向量检索最容易翻车的地方**。

### 把优势放进三角：精度、延迟和成本

前面的五条案例只回答了“能不能找对”，还没有回答两个生产问题：**混合检索慢多少，又贵多少？** `code/D3/d3_6_benchmark.py` 把比较范围收窄到检索阶段，避免把模型生成速度和输出长度混进来。

实验遵循四条控制变量：

1. 两种策略使用同一批 50 条可回答案例、同一份知识库和同一个 Top-3。
2. 先预热 5 轮，再正式测量 30 轮；每种策略共记录 1500 次检索。
3. 精度看 Hit@1、Hit@3 和 MRR；延迟看每次检索的 P50、P95。
4. 成本不绑定某家云厂商，分别记录查询分支、平均上下文 Token，并用可替换单价换算千次查询成本。

> 知识扩展：Hit@1、Hit@3 和 MRR 是检索评测里最常用的三个指标。Hit@1 表示Top-1 是否命中金标；Hit@3 表示Top-3 里是否至少有一条金标；MRR（Mean Reciprocal Rank，平均倒数排名）则是看第一条正确结果排在第几，用倒数打分，再对所有问题平均。

先运行不需要数据库和 API Key 的可复现基线：

```bash
cd code
python D3/d3_6_benchmark.py
```

默认按 `CNY 1 / 百万输入 Token` 演示成本公式。它不是市场报价；假设你的实际输入价是每百万 Token 3 元，可以这样重跑：

```bash
python D3/d3_6_benchmark.py --input-price-per-million 3 --currency CNY
```

本仓库的一次离线运行结果如下，完整报告位于 `code/D3/reports/retrieval-triangle.md`：

| 方案 | Hit@1 | Hit@3 | MRR | P50（ms） | P95（ms） | 平均上下文 Token / 查询 | 示例上下文成本 / 千次查询 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯向量 | 0.72 | 0.76 | 0.74 | 0.7524 | 0.7927 | 61.78 | CNY 0.0618 |
| 混合检索 | 0.96 | 0.98 | 0.97 | 0.8173 | 0.8573 | 76.54 | CNY 0.0765 |

在这次运行里，混合检索让 Hit@1 提高了 **24 个百分点**，代价是 P95 增加 **8.1%**、平均上下文 Token 增加 **23.9%**。

这才是完整的结论：**混合检索不是“免费变准”，而是用一条全文分支、一次 RRF 融合和稍长的上下文换精度。**

seekdb 的两条检索分支在一次 `hybrid_search` 数据库请求中完成，所以应用侧并不会发两次请求。两种方案也都只需要一次查询 Embedding。示例中的上下文成本按下面的公式计算：

```text
千次查询上下文成本 = 平均上下文 Token / 查询 × 1000
                     ÷ 1,000,000 × 每百万输入 Token 单价
```

这里没有把数据库实例、索引存储、Embedding 服务和答案输出 Token 伪装成一个“通用价格”。这些费用跟部署形态和供应商强相关，生产实验应该用实际账单补齐。

默认模式使用确定性离线检索器，适合复现三角分析方法，但不能代表真实 seekdb 或 Embedding 延迟。初始化 D3 知识库后，可用同一套指标实测当前 seekdb 集合：

```bash
cd code
python D3/d3_1_ingest.py
python D3/d3_6_benchmark.py --backend seekdb --rounds 50
```

macOS / Windows 需要按 `code/README.md` 启动隔离的 seekdb Server；Linux 可使用 Embedded 模式。真实压测还要在目标数据规模和并发量下补 P99、CPU、内存与索引存储，不能拿几十条教学数据外推线上容量。

| 业务条件 | 推荐策略 | 原因 |
| --- | --- | --- |
| 错误码、版本号、函数名多，答错代价高 | 混合检索 | 本次离线样例的精度收益明显高于本机延迟增幅；生产环境仍需复测 |
| 几乎都是语义改写，延迟预算极紧 | 纯向量基线 | 先用业务集确认精度已经达标，再换取更短路径 |
| 查询类型混合、流量较大 | 自适应路由 | 普通语义查询走向量；检测到精确标识符时再走混合检索 |

### 这意味着什么

如果你的 Agentic RAG 系统只用纯向量检索，那么上面这些场景中，Agent 拿到的就是**错误的检索结果**。模型再聪明，基于错误的数据推理出的答案也是错的——甚至比没有检索更危险，因为 Agent 会信心满满地给出一个“看起来很对但其实答非所问”的回答。

用 P2 的归因框架来说：这不是模型层的问题（模型的推理能力没问题），这是**数据层的问题**（检索策略选错了，导致模型拿到了错误的数据）。

而且这类错误特别隐蔽。如果 Agent 直接说“我不知道”，用户至少知道需要换个方式问。但当 Agent 基于“语义相近但实际错误”的检索结果回答时，它的语气完全是自信的——因为它确实拿到了“来源数据”，只不过拿错了。用户很难察觉回答是错的，除非他自己去核实原始数据。这种“自信的错误”比“坦诚的无知”危害大得多。

![](https://raw.githubusercontent.com/datawhalechina/easy-data-x-ai/main/docs/public/images/dev/D3/02-search-comparison-result.png)

把实验结果放到 Agentic RAG 的上下文中更容易理解。我们拿“2024年Q3营收数据”这条查询举个例子：

```bash
cd code
python D3/d3_3_compare.py
```

脚本会在同一个知识库上运行纯向量、混合检索和元数据过滤，并根据实际 Top-1 内容统计结果。正文中的表格是帮助理解的示例，不应替代你本机的真实输出。

## 第四部分：从实验到生产——你需要知道的几件事

跑完对比实验之后，你对混合检索的价值应该已经有了直觉判断。在把这个 Agentic RAG 系统推向更真实的场景之前，有几个工程层面的要点值得提一下。

### 检索结果的质量直接决定回答质量

这听起来像废话，但很多团队在 RAG 系统出问题时的第一反应是调 Prompt 或者换模型。对照实验告诉你：如果 Agent 拿到的检索结果本身就是错的，Prompt 调得再好也没用。**先确保检索层给出了正确的结果，再考虑模型层的优化。**

### 工具描述（Tool Description）的质量也很关键

你在 D1 学过这一点：如果工具的 `description` 写得不清楚，模型不知道什么时候该调用它。在 Agentic RAG 中，这意味着模型可能在应该查知识库的时候没有查——直接“猜”了一个回答。这就是幻觉的一个常见来源。

所以 Prompt Engineering 在 Agentic RAG 里仍然重要——但它解决的是“Agent 要不要查”和“怎么组织答案”的问题，不是“查到什么”的问题。后者是数据层的事。

### 检索数量（top_k）的取舍

top_k 设太小，可能遗漏关键信息；设太大，会给模型传入大量不相关内容，影响推理质量（噪声太多反而干扰判断）。实践中，top_k = 3~5 是一个常见的起点，可以根据你的数据密度和查询类型调整。

### 数据更新与一致性

知识库不是一次性构建完就不管了。文档会更新，错误码会新增，版本会迭代。你需要一个机制来保持知识库数据和源文档的同步。最简单的方案是定期全量重建，但更实际的做法是增量更新——只处理变更的文档。seekdb 支持对已有集合追加和更新数据，你不需要每次都删库重建。

在生产环境中，“数据新鲜度”往往比“检索算法的精细调优”更影响用户体验——Agent 回答了一个三个月前就已经修复的 bug 的解决方案，用户的信任会立刻崩塌。这也是为什么 P2 讲知识库是产品决策：**更新频率本身就是一个需要 PM 和开发共同决定的产品策略**。

## 第五部分：让 D3 回答 P2 的工程问题

到这里，我们已经跑通 Demo，但 P2 提到的生产问题还不能停留在提醒。`code/D3/rag_engineering.py` 把六段链路拆成可测试的节点，再通过依赖注入接入真实数据库、业务 API 和模型。这样既能离线跑通，也能清楚看到每次查询为什么改写、走了哪些检索路由、为什么重试或拒答。

| P2 的阶段与风险 | D3 的落地实现 | 如何验证 |
| --- | --- | --- |
| 数据准备：重复写入、来源与元数据漂移 | 固定集合 + `upsert`，保留稳定文档 ID 和元数据 | 重复执行写入测试 |
| 查询改写：口语表达、精确标识符和复合问题混在一起 | `analyze_query()` 完成口语归一、实体抽取、意图判断和问题拆解 | 改写、过滤条件与子问题测试 |
| 自适应检索：所有问题都走同一条路径 | `build_retrieval_plan()` 按问题选择 `vector`、`keyword`、`structured`，失败后增加 `fallback` | 多查询、多路由与纠错检索测试 |
| 融合重排：重复结果、越权内容、旧文档混入 | 权限先过滤，RRF 去重；打平时参考来源等级和更新时间 | 权限、稳定排序与上下文预算测试 |
| 检索纠错：搜到内容就直接交给模型 | `grade_evidence()` 在生成前过滤弱证据，证据不足时触发补搜 | 相关性阈值与补搜测试 |
| 答案验证：有引用格式不等于有证据支持 | `validate_answer()` 逐条检查引用，可注入支持度评估器做语义校验 | 缺失引用、未知引用、无支持声明测试 |
| 评估反馈：只看几个 Demo，不知道改好还是改坏 | `PipelineTrace` 记录查询、路由、评分、失败原因、延迟和调用量 | 60 条六类回归集与 Markdown 报告 |

### 查询改写：先把用户语言变成检索语言

P2 提到的问题分析、实体抽取、问题改写和问题拆解，在这里由同一个入口完成：

```python
from rag_engineering import analyze_query

analysis = analyze_query(
    "查询 OB-4.2.1 当前状态，同时给出 E-4012 的排查步骤"
)

print(analysis.rewritten_query)
print(analysis.filters)
print(analysis.sub_queries)
print(analysis.routes)
```

这段代码不会把 `OB-4.2.1`、`E-4012` 交给语义模型自由发挥，而是保留为稳定过滤条件。显式并列的问题会拆成多个子查询；包含状态、数量、余额等提示的问题会增加结构化路由。离线示例只内置少量可审计的口语归一规则，生产环境可以在调用前接入领域词典或模型改写，但必须保留原问题和精确实体，方便回放。

### 自适应检索：让问题选择路径

`adaptive_retrieve()` 会执行“子查询 × 检索路由”，每一路都返回统一的 `Evidence`。第一次没有足够证据时，第二轮会扩展查询，并在注册了备用检索器时增加 `fallback` 路由。

```python
from rag_engineering import run_engineering_pipeline

result = run_engineering_pipeline(
    question=user_question,
    route_retrievers={
        "vector": vector_search,
        "keyword": full_text_search,
        "structured": query_business_api,
        "fallback": search_official_docs,
    },
    evidence_grader_fn=grade_relevance,
    generate_fn=generate_with_citations,
    answer_support_fn=check_claim_support,
    max_retries=1,
)
```

这四类回调的边界是明确的：

| 回调 | 输入 | 输出 |
| --- | --- | --- |
| 路由检索器 | `query, analysis` | `Evidence` 列表或多组列表 |
| 证据评估器 | `question, evidence` | `0～1` 的相关性分数 |
| 生成器 | `question, context` | 带 `[doc_id]` 的答案 |
| 支持度评估器 | `question, claim, cited_evidence` | 当前声明是否被引用证据支持 |

示例中的函数名代表生产环境需要接入的实现。仓库自带的 `d3_5_evaluate.py` 使用确定性离线替身跑通同一套接口，不需要 API Key 或数据库。

### 答案验证：从“有引用”推进到“引用支持结论”

只检查答案末尾有没有 `[doc_id]` 还不够。当前实现会逐条检查答案声明：引用必须来自本次上下文；声明不能漏引；如果传入 `answer_support_fn`，还会判断引用内容是否真的支持这条声明。`[doc_id] 段落内容` 表示引用覆盖当前段落，句末引用只覆盖当前声明。

校验失败不会把可疑答案直接返回给用户。流水线会带着原始精确实体补搜一次；仍然失败时稳定拒答。权限过滤发生在融合和证据评分之前，因此补搜不会泄漏无权查看的标题、ID 或摘要。

整个控制回路可以压缩成下面几步：

```plain
改写与拆解 → 生成检索计划 → 多路召回 → 权限过滤与 RRF
                                           ↓
回答或拒答 ← 声明级答案校验 ← 生成 ← 证据相关性评分
     ↑                                      ↓
     └──────── 最多一次扩展查询与备用检索 ────────┘
```

### 先跑不需要 API Key 的 60 条离线评测

```bash
cd code
PYTHONPATH=D3 python D3/d3_5_evaluate.py
```

评测集位于 `code/D3/data/evaluation_cases.jsonl`，共六类、每类 10 条：

- 精确标识符
- 语义改写
- 口语别名
- 多证据问题
- 边界与干扰项
- 证据不足与安全拒答

当前离线基线会生成 `code/D3/reports/offline-evaluation.md`。它用于稳定验证检索和编排逻辑，不调用模型，也不代表线上模型质量。真实模型的 Faithfulness、Answer Relevancy、延迟和账单，需要在你的环境里单独测量。

### 再用 RAGAS 把「检索之后生成」也打上分

上面 60 条离线评测回答了一个问题：检索和编排有没有按金标走对。

但这还不够。P2 的四指标框架关心的是整条链路——查到的上下文够不够、答案忠不忠实、答得是否切题。所以还得在同一知识库、同一批问题上，对纯向量和混合检索各跑一遍：**检索 → 生成 → 打分**。

课程为此准备了一套小而完整的 RAGAS 评测集：共 **20 条**，覆盖精确匹配、多跳、时效、模糊四类场景，每类 5 条。知识库也补了升级路径、接口废弃、运维 FAQ 等片段（`kb_013`–`kb_019`），专门撑住时效和模糊题。完整标注在 `code/D3/data/eval_dataset.json`。

四类场景各自要观察的重点不同：

| 场景 | 代表问题 | 重点观察 |
| --- | --- | --- |
| 精确匹配 | 错误码、函数名、参数名、季度 | 全文锚点能不能补上向量检索的短板 |
| 多跳问题 | 跨两个知识片段比较、计算或串联 | 必需事实有没有全部进入 Top-K |
| 时效性查询 | 截止日期、最新季度、当前版本 | 新证据有没有盖住旧证据 |
| 模糊表达 | 口语化、同义改写、缺少专有名词 | 查询分析能不能抽出有效短关键词 |

这里有一个容易踩坑的细节：**混合检索的全文查询只来自用户问题本身**。

错误码、函数名、参数名、季度会提成短锚点；单一版本号改走元数据过滤；模糊和时效问法会做固定的同义词归一化。代码不会偷看金标答案来造关键词，也不会把整句问题原样塞进 `$contains`。如果抽不出独立的全文查询，评测会直接失败——避免把「其实只跑了向量分支」误记成混合检索。

这 20 条是课程尺度的小样本：适合把评测流程跑通、看清效应方向，**不能只凭一次平均分就宣称统计显著**。脚本会对同一道题算「混合 − 纯向量」的配对差值，并给出 95% 置信区间和 p-value；只有区间不跨 0 且 `p < 0.05` 时，摘要才会标成显著。

动手时，先把知识库写好，再按顺序跑：

```bash
cd code
pip install -r requirements.txt
PYTHONPATH=D3 python D3/d3_1_ingest.py
PYTHONPATH=D3 python D3/d3_5_ragas_eval.py --check-config
PYTHONPATH=D3 python D3/d3_5_ragas_eval.py --mode retrieval   # 先看金标检索指标
PYTHONPATH=D3 python D3/d3_5_ragas_eval.py --mode ragas       # 完整 RAGAS（消耗评测 LLM）

# 只有完整成功且准备发布时，才额外生成可提交摘要
PYTHONPATH=D3 python D3/d3_5_ragas_eval.py --mode ragas \
  --publish-summary D3/data/live_ragas_summary.json
```

核心逻辑其实很简单：**换策略时只换检索函数，问题集和打分方式保持不变。**

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy

rows = []
for case in cases:
    contexts, _ = retrieve_fn(case["question"])
    answer = generate_fn(case["question"], contexts)
    rows.append({
        "question": case["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": case["reference"],
    })

result = evaluate(
    Dataset.from_list(rows),
    metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=True,
)
```

每次运行都会在 `code/D3/data/runs/<UTC 时间>/` 落下三组结果：

- `run_manifest.json`：数据集哈希、知识库指纹、模型与依赖版本——用来回答「这次跑的是什么」
- `case_results.vector.jsonl` / `case_results.hybrid.jsonl`：每一题的检索、答案和评分——出了问题可以对着单题复算
- `summary.json` / `summary.md`：整体指标、分场景指标，以及配对置信区间与 p-value

有一点要强调：检索、生成或 RAGAS 打分任何一步失败，脚本都会直接非零退出。它不会悄悄丢掉缺失值再给你一个好看的平均分。`--publish-summary` 也只有完整跑通才会写入，并且带着本次运行一致的数据集哈希、知识库指纹和两种策略的逐例结果，方便事后复算。

课程正文故意不固化某一次易过期的模型分数。你要发布新结果时，请连同逐例工件和溯源摘要一起审查。

读表时建议按这个顺序：

1. 先确认 `n_scored == n_total`，且 `failures` 为空——分数才可信
2. 再看 Context Recall / Evidence Coverage——检索侧有没有把该拿的证据拿全
3. 最后看生成侧指标和配对区间——答案是否忠实、是否切题，以及混合是否真的越了纯向量

Answer Relevancy 受评测模型影响较大，单次绝对值不要过度解读。维护者还可以跑 `python -m unittest D3.test_d3_ragas_eval`，校验评测集、查询分析、失败门禁和统计聚合是否仍成立。

同一轮本机离线运行的三策略对比如下，详细口径见 `code/D3/reports/strategy-comparison.md`：

| 方案 | Hit@3 | 拒答准确率 | P95（ms） | 平均检索调用 | 估算 Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯向量基线 | 0.76 | 0.70 | 1.11 | 1.00 | 9299 |
| 混合检索 | 0.98 | 1.00 | 1.16 | 1.00 | 10645 |
| 工程管线 | 1.00 | 1.00 | 2.75 | 2.73 | 10878 |

这组数据没有证明「工程管线更便宜」。相反，它为了补搜和校验多花了调用与 Token，换来的是更高的复杂问题覆盖率。离线 P95 只代表本机编排开销；真实数据库与模型延迟通常会高几个数量级。

### 失败时怎么恢复

工程流水线最多补搜一次。

第一次没有证据、证据相关性评分不足，或答案校验失败时，它会保留错误码、版本号等精确信息，扩展查询并按需启用备用检索器，再试一轮。第二次仍失败，就稳定拒答——不会为了「看起来有答案」硬编。

如果候选资料存在，但当前身份无权访问，则直接返回「无权访问」。重试过程中也不会泄漏标题、ID 或摘要。

这一段闭环，正是 P2 所说的 Agentic：**系统能根据中间结果决定补搜、停止或拒答**，而不是把一条固定流水线包装成 Agent。

## 我们的思考

对比实验是我们说服自己（和客户）“混合检索不是可选优化”的方式。

每次有人问“纯向量搜索够用吗”，我们都不争论——我们建议他拿自己的业务数据跑一次对比。特别是那些包含产品型号、版本号、错误码、精确数值的查询——这些在几乎每个企业的知识库中都大量存在，而且恰恰是用户最关心精确性的场景。跑完之后，结果替我们说话。

seekdb 的混合检索通过单条查询完成（`DBMS_HYBRID_SEARCH`），向量搜索和全文搜索在引擎内部执行并通过 RRF 算法融合排序——不需要在应用层维护两套检索系统，也不需要自己写结果合并逻辑。这让做对比实验变得非常简单：你只需要把 `hybrid_search` 换成 `vector_search`，其他代码一行不用改，就能直接看到两种策略的差异。

这也是我们做 seekdb 时反复验证的一个设计原则：**正确的做法应该同时是最简单的做法。** 如果“做得对”需要你额外维护两套系统、写大量胶水代码、处理复杂的分数归一化——那大多数团队都会“先凑合用向量搜索”。但当混合检索只是一个参数的差别时，没有理由不用它。

![](https://raw.githubusercontent.com/datawhalechina/easy-data-x-ai/main/docs/public/images/dev/D3/03-hybrid-search-advantage.png)

回到课程的主线来看这件事：F1 讲了大模型的三个局限本质上都是数据问题，F2 讲了 Agent 的每一项能力拆到底都是数据的存储与检索。D3 的对比实验是这条主线的一次具体验证——同样的模型、同样的 Prompt、同样的数据，仅仅因为检索策略不同，最终结果就是“对”和“错”的区别。这不是理论推演，是你刚刚亲手跑出来的实验结论。

## 这节课要留下的印象

如果这节课的所有内容你只记住一段话，记住这段：

> **同样是 RAG，检索策略的差异导致肉眼可见的结果差距——不需要看论文，跑一次对比实验就明白了。混合检索不是高级优化，而是生产级 RAG 的基本要求。**

## 课后行动

1. **用你自己的数据重复这个实验**。本模块提供了示例数据集，但真正有说服力的实验是用你自己业务场景的数据来跑。把你的产品文档、API 文档、内部 FAQ 导入 seekdb，然后用你和同事日常真正会问的问题作为查询。

2. **记录 3 个最能体现差距的查询案例**。特别关注那些包含专有名词、版本号、精确术语的查询——这些是纯向量检索最容易翻车、混合检索优势最明显的场景。

3. **分享给觉得“纯向量搜索够用了”的同事**。不需要争论，把对比实验的结果发给他——数据说话比任何论点都有说服力。

## 延伸阅读

如果你对本期提到的概念想做进一步了解，以下是一些推荐资源：

- **分块策略（Chunking Strategies）**：固定长度分块（按 Token 数或字符数切分）实现简单但可能切断上下文；语义分块（按段落、章节或主题边界切分）保留了上下文完整性但实现更复杂。实践中，很多团队从固定长度分块开始，在遇到检索质量问题后再切换到语义分块——这是一个典型的“先跑通再优化”的工程决策
- **RRF（Reciprocal Rank Fusion）算法**：混合检索中，如何将向量搜索的排序结果和全文搜索的排序结果融合成一个统一排名？RRF 是业界最常用的方案。核心思想：不直接比较分数（因为两种搜索的分数含义不同），而是基于**排名**做融合——排名靠前的结果得到更高权重。公式简洁、效果稳定，seekdb 的 `DBMS_HYBRID_SEARCH` 内部使用的就是这个算法
- **LangChain RAG 文档**：[RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)，LangChain 的 RAG 教程，展示了如何用框架组织检索和生成流程。本课程选择不依赖框架而是用原生 API，是为了让你看清每一步发生了什么——理解原理之后，框架只是效率工具

> **下一期预告**：D4 · Agent 开发与记忆系统——你的 Agent 现在能查知识库了，但它还没有”记忆”。每次对话都是从零开始，不知道你是谁、之前聊过什么。D4 会给 Agent 加上记忆系统——用 PowerMem 实现记忆的存储、检索和遗忘。你会亲眼看到”有记忆”和”没记忆”的 Agent 在对话体验上的差距。

---

欢迎各位老师在 https://github.com/datawhalechina/easy-data-x-ai 参与课程共建。

也欢迎各位老师加入 Data x AI 交流群~

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/easy-data-x-ai/main/docs/public/images/base_knowledge/F0/F0-20.png" width="200" />
</div>
