---
title: 从 Harness 到 Loop，再到 Graph Engineering
description: 从模型外部运行系统、单个反馈循环到多个 Loop 的协作图，理解 Harness、Loop 与 Graph Engineering 的工程边界。
---

![《深度“解剖”AI Agent Harness》总结图](/images/extra/X6/harness-14-summary-cover.png)

# 从 Harness 到 Loop，再到 Graph Engineering

如果把一个 AI Agent 拆开看，模型只是其中负责推理的部分。模型之外，还需要工具、记忆、上下文管理、状态持久化、错误处理、权限控制和验证机制。这些基础设施构成 Harness。

Harness 让模型能够行动，而 Loop 让它根据行动结果继续调整。当一个 Loop 装不下执行、检查和方向判断时，把职责拆到多个执行单元是一种常见信号：谁负责前进，谁负责纠错，谁能够叫停，失败后又回到哪里。Graph Engineering 也可以组织工具、确定性程序、验证器和人工节点，并不要求图中一定存在多个 Agent Loop。

![Agent Harness 的整体结构](/images/extra/X6/01-harness-anatomy.png)

## 一、Harness：模型之外的完整运行系统

你可能已经搭过聊天机器人，甚至用几个工具写过一个 ReAct 循环。Demo 跑起来一切顺利，但进入生产环境后，模型可能忘记几步之前做过什么，工具调用可能静默失败，上下文窗口也可能逐渐被低价值信息占满。

问题往往不只在模型，还在模型外面的基础设施。

![AI Agent 与 Agent Harness 的区别](/images/extra/X6/harness-03-agent-vs-harness.png)

两个很直观的例子：LangChain 在模型和参数不变的情况下，仅调整模型外部架构，就在 TerminalBench 2.0 上从 30 名开外提升到第 5；另一项让大模型参与优化 Harness 的研究，则把通过率提升到了 76.4%。这些数字来自来源文章，具体基准和实验条件仍应回到原始资料核对，但它们说明了同一件事：模型之外的工程结构会显著影响 Agent 的实际表现。

AI Agent Harness，就是套在大模型外面的整套软件架构：编排循环、工具、记忆、上下文管理、状态持久化、错误处理、护栏、安全执行和生命周期管理等都属于这一层。

“Agent”通常指用户看到的表现：一个有目标、会使用工具、能够根据结果修正行为的实体；“Harness”则是支撑这种表现的幕后系统。可以把裸模型类比为一颗能计算、但没有内存、硬盘和输入输出设备的 CPU：上下文窗口像内存，外部数据库像硬盘，工具集成像设备驱动，而 Harness 更接近把这些部分组织起来的操作系统。



![从裸模型到完整计算机系统的类比](/images/extra/X6/harness-04-cpu-operating-system.png)

### 从提示词工程到 Harness 工程

围绕模型的工程化工作可以分为几个逐层扩大的范围：

- **提示词工程（Prompt Engineering）**：把给模型的指令写清楚。
- **上下文工程（Context Engineering）**：管理模型在什么时间看到什么信息。
- **Harness 工程（Harness Engineering）**：包含前两者，并进一步负责工具编排、状态持久化、错误恢复、验证循环、安全执行和生命周期管理。

![提示词工程、上下文工程与 Harness 工程](/images/extra/X6/harness-05-engineering-levels.png)

Harness 不是简单的提示词套壳，而是让模型能够在真实环境中持续工作的运行系统。

## 二、生产级 Harness 的核心组件

![生产级 Harness 的核心组件](/images/extra/X6/harness-06-core-components.png)

### 1. 编排循环（Orchestration Loop）

常见的 Agent 运行方式是“思考—行动—观察”（TAO），ReAct 是其中一种常见模式：拼装提示词，调用模型，解析输出，执行工具调用，把结果返回给模型，然后继续下一轮，直到任务完成。

代码层面可能只是一个 `while` 循环，困难之处却不在循环语句本身，而在循环中的状态、权限、异常、预算和退出条件。第一篇来源文章把这种运行时概括为“笨循环”：推理主要由模型完成，Harness 负责执行和循环。

![Harness 编排循环](/images/extra/X6/harness-07-orchestration-loop.png)

### 2. 工具（Tools）

工具是 Agent 的“手”。每个工具通常通过结构化 Schema 描述名称、用途和参数类型，并注入模型上下文，让模型知道自己可以调用什么。

工具层需要负责注册、参数校验、参数提取、沙箱执行、结果捕获，以及把结果格式化为模型能够理解的“观察”。例如，Claude Code 提供文件操作、搜索、执行、网页访问、代码分析和子智能体创建等工具；OpenAI Agents SDK 支持函数工具、托管工具以及 MCP（Model Context Protocol）服务器工具。

### 3. 记忆（Memory）

记忆可以分为短期和长期两类。短期记忆通常是单次会话内的对话历史；长期记忆则跨会话存在，可以保存在项目文件、结构化存储或会话数据库中。

第一篇来源文章将 Claude Code 的记忆结构描述为三层。结合当前官方文档，更准确的边界是：跨会话信息主要由项目指令文件和 Auto Memory 承载；Auto Memory 内部使用常驻上下文的轻量索引和按需读取的详细主题文件。原始会话记录可以作为回溯材料，但不应与产品记忆机制混为一层。

一条重要原则是：Agent 应当把记忆视为提示，而不是未经验证的事实。真正行动之前，仍要用当前环境的实际状态进行核对。

### 4. 上下文管理（Context Management）

上下文管理是 Agent 容易失效的环节。关键信息如果落在长上下文的中间位置，可能更容易被忽略。第一篇来源文章援引“迷失在中间”现象称，关键信息落到窗口中部时，模型表现可能下降 30% 以上；即使上下文窗口很大，信息变长之后，模型遵循指令和提取信号的能力也可能下降。

![记忆系统与上下文管理策略](/images/extra/X6/harness-08-context-management.png)

生产环境常见的应对方式包括：

- **压缩（Compaction）**：接近窗口上限时总结历史，保留架构决策和未解决问题，移除冗余工具输出。
- **观察掩码（Observation Masking）**：隐藏旧的工具输出，但保留调用记录。
- **即时检索（Just-in-time Retrieval）**：上下文中只保留轻量标识符，需要时再读取具体内容。
- **子智能体委托**：让子智能体完成局部调研，只返回浓缩结果。来源文章给出的参考范围是每个子智能体只带回约 1000～2000 Token 的摘要。

目标不是向上下文窗口塞入尽可能多的 Token，而是找到信号足够强、体积尽可能小的信息集合。

### 5. 提示词构建（Prompt Construction）

提示词构建决定模型在每一步能够看到什么。典型内容包括系统提示词、工具定义、记忆文件、对话历史和当前用户消息。

不同信息还需要明确优先级。以 Codex 这类 Agent 为例，系统指令和开发者指令优先于用户指令；历史中的消息仍保留各自角色和优先级。工具 Schema 是运行时提供的能力描述，工具输出则是供模型继续判断的上下文，不能把它们简单排列成独立的指令层级。

### 6. 输出解析（Output Parsing）

现代 Harness 通常使用原生工具调用：模型直接返回结构化的 `tool_calls` 对象，而不是由程序从自由文本中提取调用意图。

Harness 的基本判断可以很直接：如果存在工具调用，就执行并继续循环；如果没有工具调用，则把模型回复作为最终答案。对于需要固定结构的结果，还可以使用 Pydantic 等模型定义 Schema 约束。

### 7. 状态管理（State Management）

LangGraph 通过 State Schema 建模在图节点之间流动的状态。配置 Checkpointer 后，系统可以按执行步骤保存检查点，从中断位置恢复，也可以回看不同阶段的状态；普通本地图并不会在未配置持久化组件时无条件保存检查点。

OpenAI 给出的状态策略包括应用内存、SDK 会话、服务器端 API 和响应 ID 链。Git 提交与进度文件则来自 Anthropic 面向长时间运行 Agent 的示例 Harness，用来让不同上下文窗口中的 Agent 接力；它不是 Claude Code 默认状态机制。

### 8. 错误处理（Error Handling）

多步骤系统中，错误会累积。假设一个流程有 10 个步骤，每一步成功率都是 99%，全部成功的概率约为 90.4%。

LangGraph 把错误分为几类：临时故障可以退避重试；模型有能力处理的错误可以作为工具消息返回，让模型自行调整；需要人工判断的错误应暂停并等待干预；意外错误则需要上报和调试。重试也必须有上限，否则只会继续消耗 Token 和时间。第一篇来源文章还提到，Stripe 的生产级 Harness 把重试上限设为两次。

### 9. 护栏与安全（Guardrails and Safety）

护栏可以覆盖输入、输出和工具调用。输入护栏在任务开始时检查请求，输出护栏检查最终结果，工具护栏则在每次调用前判断参数、权限和风险。触发预设条件后，Harness 可以根据策略拒绝本次调用、替换工具结果，或者中断整个执行流程。

一个关键的架构原则，是把模型推理与权限执行分开。模型可以提出要做什么，但最终是否允许执行，应由 Harness 根据权限规则决定。

![错误处理与三层护栏](/images/extra/X6/harness-09-guardrails.png)

### 10. 验证循环（Verification Loops）

验证是 Demo 与生产系统之间的重要分界。常见方法包括：

- 用测试、Lint 和规则检查提供确定性反馈；
- 用截图等视觉反馈检查界面结果；
- 让另一个模型或独立 Agent 评估输出质量。

关键不只是生成结果，而是让系统有办法验证自己做出的改变。

第一篇来源文章引用 Claude Code 创造者 Boris Cherny 的经验判断：当模型能够验证自己的工作时，产出质量可能提升 2～3 倍。这个数字同样依赖具体任务，但验证能力的重要性并不只体现在最终评审，它应当贯穿每一轮执行。

### 11. 子智能体编排（Subagent Orchestration）

Claude Code 支持复制父上下文、使用独立上下文协作，以及通过 Git Worktree 隔离工作等方式。OpenAI Agents SDK 则支持把 Agent 当作工具调用，或者将控制权移交给另一个 Agent。

子智能体不是越多越好。拆分会引入路由成本、上下文丢失和协调开销，因此应先判断任务是否真的能够独立分解。

## 三、一个 Harness 循环如何运作

把这些组件放在一起，一轮 Harness 通常包含以下步骤：

1. **提示词组装**：组合系统提示词、工具定义、记忆、对话历史和当前用户消息。
2. **模型推理**：把完整上下文发送给模型，模型返回文本、工具调用或两者的组合。
3. **输出分类**：如果没有工具调用，循环结束；如果有工具调用，则进入执行阶段；如果发生移交，则切换当前 Agent。
4. **工具执行**：校验参数和权限，在受控环境中运行工具并捕获结果。只读操作可以并发，写操作通常需要更谨慎地排队或审批。
5. **结果打包**：把正常结果或错误格式化为模型可理解的消息，让模型获得自我修正的机会。
6. **上下文更新**：把结果加入历史；如果接近窗口上限，则触发压缩或检索策略。
7. **继续循环**：回到第一步，直到满足退出条件。

![Harness 的七步运行循环](/images/extra/X6/harness-10-seven-step-loop.png)

退出条件也要分层设计，包括：模型给出无工具调用的最终回复、达到最大轮次、Token 预算耗尽、护栏触发、用户中断或安全拒绝。

对于跨越多个上下文窗口的长任务，Anthropic 介绍过一个两阶段示例 Harness：初始化 Agent 先准备环境、初始化脚本、进度文件、功能清单和初始 Git 提交；后续编码 Agent 在每个会话中读取 Git 日志和 `claude-progress.txt`，定位当前状态，选择优先级最高的未完成工作，完成后提交并留下摘要。文件系统在这里承担跨窗口的“接力棒”。第一篇来源文章把这类做法与“Ralph 循环”联系在一起，但 Anthropic 的官方文章并未使用这个名称，因此这里按“两阶段长任务 Harness”表述。

## 四、现实中的 Harness 如何落地

- **Anthropic（Claude Agent SDK）**：通过 `query()` 函数暴露 Harness。Claude Code 采用“收集—行动—验证”的节奏：先搜索文件、阅读代码，再修改文件、执行命令，最后运行测试并检查输出。
- **OpenAI Agents SDK**：以原生 Python 表达 Agent、工具、移交和工作流逻辑。
- **Codex**：使用独立的 Agent 运行时与应用服务器技术栈。Codex App Server 提供双向 JSON-RPC API，CLI、IDE 和 Web 等客户端可以在其上构建交互。
- **LangGraph**：把 Harness 建模为显式状态图。模型节点和工具节点之间通过条件边连接：有工具调用则进入工具节点，否则走向终点。
- **CrewAI**：以角色组织多 Agent 协作，每个 Agent 拥有角色、目标和工具；流程层负责确定性骨干逻辑。
- **AutoGen**：支持顺序、并发、群聊、移交等编排方式；其 GraphFlow 图式工作流在当前开发文档中仍标为实验性能力，使用时需要关注版本变化。

![五类 Agent 框架的 Harness 设计对比](/images/extra/X6/harness-11-framework-comparison.png)

Harness 也可以理解为脚手架：它本身不直接等于最终成果，却让模型能够在更复杂的环境中完成工作。随着模型能力提升，一部分原本写死在 Harness 中的逻辑可能被简化，复杂工具也可能被通用执行接口替代。

![Harness 随模型能力共同演化的脚手架类比](/images/extra/X6/harness-12-scaffolding.png)

第一篇来源文章以 Manus 为例：其 Harness 在六个月内重写了五次，每次都在减少复杂度，把复杂工具定义逐步收敛为更通用的执行接口，把“管理 Agent”收敛为更简单的结构化移交。这说明 Harness 不是堆得越厚越好，它要随着模型能力一起演化。

这形成了模型与 Harness 的协同进化。模型会学习使用特定的工具和运行环境，Harness 也会随着模型能力变化而调整。一个面向未来的检查方式是：换用更强模型后，如果系统性能能够提升，而 Harness 不必同步变得更复杂，说明架构保留了足够的适应空间。

## 五、设计 Harness 时的七个关键决策

![设计 Harness 时的七个关键决策](/images/extra/X6/harness-13-seven-decisions.png)

1. **单 Agent 还是多 Agent**：先充分利用单 Agent。只有工具明显重叠、领域明确分离或任务可以独立并行时，再考虑拆分。第一篇来源文章给出的经验边界是：当工具超过约 10 个且存在明显重叠，或者任务领域可以清楚分离时，再认真评估多 Agent。
2. **ReAct 还是先规划后执行**：ReAct 边推理边行动，灵活但成本较高；先规划后执行把规划和执行分开，适合可以提前形成稳定计划的任务。来源文章提到，LLMCompiler 的实验中，先规划后执行比顺序 ReAct 快 3.6 倍。这个数字不能泛化到所有任务，但说明并行计划在合适场景中可能减少等待。ReAct 是常见模式，不是 Agent Loop 的唯一模式。
3. **上下文管理策略**：总结、观察掩码、结构化笔记和子智能体委托各有适用范围。来源文章引用 ACON 的结果称，优先保留推理过程而非原始工具输出，可减少 26%～54% 的 Token 消耗，同时保持 95% 以上的准确率。应优先保留对后续决策有用的信息，而不是机械保留全部历史。
4. **验证循环设计**：测试和 Lint 提供确定性反馈，模型评审能够检查语义问题但增加成本。行动前的引导和行动后的传感器需要配合。
5. **权限与安全架构**：宽松权限执行更快，但风险更高；严格审批更安全，却会增加等待时间。选择取决于部署环境。
6. **工具范围管理**：工具不是越多越好。来源文章提到，Vercel 为 v0 删除了 80% 的工具后效果反而更好，Claude Code 通过懒加载把相关上下文缩减了 95%。这些案例共同指向一个原则：只向模型暴露当前步骤需要的最小工具集，可以减少上下文负担和选择错误。
7. **Harness 的厚度**：需要决定多少逻辑写死在系统中、多少判断交给模型。薄 Harness 借助模型能力提升，显式图框架则强调可预测的控制。来源文章还提到，Anthropic 会随着模型能力提升，从 Claude Code 的 Harness 中删除已经被模型内化的规划步骤。

### Harness 即产品

使用相同模型的两个 Agent，表现可能相差很大，差异就来自 Harness。它不是已经解决的问题，也不是可以随意替换的通用外壳，而是产品工程能力的一部分：上下文怎样作为稀缺资源管理，验证怎样阻止错误累积，记忆怎样提供连续性而不放大幻觉，状态和权限怎样被可靠控制。

随着模型越来越强，Harness 可能逐渐变薄，但不会消失。即便模型能力继续提升，系统仍需要有人管理上下文、执行工具、保存状态、验证结果并约束权限。遇到 Agent 表现不稳定时，除了检查模型，也应检查它所处的 Harness。

## 六、运行时数据：Harness 的数据底座问题

记忆、上下文管理、状态管理、错误处理、护栏和安全都会产生运行时数据。这些数据通常高频、半结构化、携带上下文，并且需要回放和比较。Harness 不只是执行这些组件，还要考虑它们的数据如何被统一承载和复用。

![Agent 运行时数据的分散流向](/images/extra/X6/harness-15-runtime-data-flow.png)

常见做法是把数据分别写入 JSONL、Markdown、SQLite 等外围文件，再由 LangSmith、Phoenix 等工具二次采集。结果可能演变为 Postgres、pgvector、Redis、ClickHouse、可观测性平台和本地文件并存的割裂技术栈，增加关联与运维成本。

这一节保留自第一篇来源文章的编辑补充。该补充提到，LangChain 为 LangSmith 自研了面向 Agent Trace 数据的 SmithDB，并由此引出一个更广泛的问题：Agent 的 Context、Execution History、Task、Observability 和 Footprint 是否应直接沉淀在统一的数据底座中，而不是分别生长成互不相通的系统。

运行时数据并不只是为了“记录一下”。后续调试、回放、对比、分析、评测和训练，都可能再次消费这些数据。对外看，这是数据的二次消费能力；对内看，这是运行时数据的资产化。数据库在这里也不再只是业务结果的存放层，而可能成为智能系统运行过程的参与者。

![从结果存放到运行时数据资产化](/images/extra/X6/harness-16-runtime-data-assetization.png)

## 七、从 Harness 进入 Loop Engineering

Harness 已经包含编排循环，但 Loop Engineering 把关注点进一步放到“如何持续运行”上。

可以用程序结构作一个直观类比：

- Prompt Engineering 像一个只接收提示词的单参数函数；
- Context Engineering 像一个接收多组上下文参数的函数；
- Harness Engineering 再把工具、权限、状态和运行环境连接进来；
- Loop Engineering 让程序持续读取状态和反馈，调整下一步行动。

Loop 不只是代码里的 `while`。一个执行者看到结果后，决定下一步、采取行动、验证，再带着新信息继续调整，这就是 Loop。ReAct 是实现这种反馈式运行的常见模式之一，除此之外也可以采用先规划后执行、回合制、目标制、定时制或主动触发等形式。

Loop 解决的是一个具体问题：**一个执行者怎样持续把事情往前推。** 目标、上下文、工具、验证方法、退出条件和人工确认，都是这个循环的一部分。

## 八、从 Loop 到 Graph：讨论是怎样出现的

![多人协作与任务交接示意图](/images/extra/X6/02-graph-origin.webp)

Graph Engineering 目前还没有公认定义，但有一点可以先确定：Loop 没有被 Graph 取代。当一个 Loop 装不下执行、检查和方向判断时，把职责拆到多个执行单元是一种常见选择；分支、汇合、并行、权限隔离和局部恢复等关系，也可能让 Graph 产生价值。

7 月 18 日，OpenClaw 作者 Peter Steinberger 问：“我们还在谈 Loop，还是已经转向 Graph 了？”

![Peter Steinberger 关于 Loop 与 Graph 的提问](/images/extra/X6/03-peter-tweet.jpg)

几个小时后，Hamel Husain 发布《Loop Engineering Is Dead. Enter Graph Engineering.》，后来又在回复中说：“Nobody knows what it is.”

一个还没有公认定义的词，已经拥有了流量、阵营、架构图和教程。与其急着争论它是不是新范式，不如先看它试图解决什么工程问题。

![执行者、检查者与方向判断者的角色分离](/images/extra/X6/04-agent-role-separation.webp)

## 九、Graph Engineering 到底在设计什么

### Graph 是多个执行单元之间的关系

如果把 Loop Engineering 类比为带状态反馈的循环程序，那么 Graph Engineering 可以理解为：把多个执行单元及其控制关系、数据关系组织成一张图。部分 Agent 节点内部可能运行自己的 Loop，但 Graph 本身不等于“多个 Loop”的集合。

这些节点可能并行，也可能互相依赖；一个节点的结果会交给另一个节点处理，检查失败后还可能回到之前的节点。因此，它通常是有向图，且不一定是无环图。

![多个执行节点组成的有向图](/images/extra/X6/05-directed-graph.png)

Graph 不必是分布式程序。它既可以运行在同一进程或同一机器中，也可以跨进程、跨服务部署；“图”描述的是节点及其控制、依赖与数据关系，而不是对部署形态作出限定。

把社区讨论放在一起，大致可以看到以下共识：

- **Loop 仍然存在**：一个 Agent 仍需根据结果继续行动、验证和修正。
- **Graph 组织多个执行单元**：它描述谁先做、谁能并行、检查失败后退回哪里，以及什么时候需要人介入。
- **节点不一定都是 Agent**：节点也可以是工具、确定性程序、验证器或人；每个 Agent 节点内部仍可能运行自己的 Loop。
- **检查关口和失败路径比方框数量更重要**：如果所有箭头都只指向“继续”，那只是一条画得更复杂的流水线。

![Graph Engineering 社区讨论中的主要共识](/images/extra/X6/06-graph-consensus.jpg)

LangGraph、AutoGen 以及传统工作流系统早就在处理节点、状态、分支和重试。这轮讨论的变化在于 Agent 节点越来越自主，也越来越不确定：它可能临时改变计划、调用工具修改真实环境，甚至在运行中继续拆分任务。

### 仍然没有结论的问题

| 讨论焦点 | 相对清楚的部分 | 仍然没有结论的部分 |
| --- | --- | --- |
| Graph 和 Loop 的关系 | 多数解释中，Graph 包含多个仍在运行的 Loop | 什么规模才值得从 Loop 称为 Graph |
| Graph 是否提前画好 | 稳定步骤、权限和检查点通常需要预先约束 | 具体任务、分支乃至角色可以动态到什么程度 |
| 多个 Agent 如何协作 | 分工、并行和交接是主流重点 | 只是接力，还是让一个 Loop 校准另一个 Loop |
| Work Graph 是什么 | 可以描述一次运行中临时形成的任务和依赖 | 它还不是统一术语 |
| Agent 是否越多越好 | 可并行探索的任务可能受益 | 顺序任务中，协调成本可能超过收益 |

如果 Graph 只是“研究 Agent 做完后交给写作 Agent”，它很像把传统 Workflow 的节点换成更聪明的执行者。另一种更值得关注的结构是：一个 Agent 实现，一个独立 Agent 评审，另一个寻找反例，还有一个重新检查最初目标是否已经偏移。它们不只是接力，还会相互检查、挑战，并在必要时阻断彼此。这就是“You need loops watching loops”所表达的含义。

多个 Loop 的编排已有不少框架和实践；让一个 Loop 检查、校准甚至叫停另一个 Loop，则还没有统一做法。后者决定了 Graph 最终只是一张协作图，还是一套真正能够纠偏的系统。

## 十、先理解单个 Loop 的能力与局限

一个执行者看到结果后，决定下一步、采取行动、验证，再带着新信息继续调整，这就是单个 Loop 的基本工作方式。

![单一执行者同时承担执行、检查和方向判断](/images/extra/X6/07-single-loop.jpg)

目标、上下文、工具、验证方法、退出条件和人工确认，共同决定这个循环能否稳定运行。

![Loop 中的目标、观察、行动、验证与退出](/images/extra/X6/08-loop-components.jpg)

单个 Loop 简单、灵活、反应快，问题也来自这里：如果生产结果、检查结果和方向判断都由同一个 Loop 完成，它很容易同时成为运动员、裁判和记分员。

## 十一、把执行、检查和方向判断拆开

当一个 Loop 已经装不下所有责任时，可以把三种职责分离：有人负责把事情做出来，有人独立检查，还有人隔一段时间重新判断方向。

![执行、检查与方向三个 Loop](/images/extra/X6/09-three-loops.jpg)

- **执行 Loop** 关心怎样推动眼前工作，例如搜索、写代码、生成内容和修复问题。
- **检查 Loop** 不继续替执行者做事，而是使用测试、约束和反例判断结果是否正确。
- **方向 Loop** 看得更慢、更远：相同问题为什么反复发生，成本是否值得，用户是否接受，最初目标是否仍然合理。

![三个 Loop 的职责边界](/images/extra/X6/10-loop-responsibilities.jpg)

这三种 Loop 不一定对应三个 Agent，也不必每一步都同时运行。它们强调的是三种责任：前进、纠错和重新定向。

可以把 Loop 看成能够根据反馈继续调整的一类工作单元，把 Graph 看成执行单元之间的分工、交接、检查和控制关系。这些执行单元既可以是 Loop，也可以是工具、程序、验证器或人。

## 十二、多个 Loop 如何配合

如果把一个 Loop 看成会自己找路的执行者，Graph 关心的就不是“工位如何排列”，而是谁把什么交给谁，谁负责验收，发现错误能否退回，方向错了又由谁叫停。

![多个 Loop 的交接、检查与退回](/images/extra/X6/11-loop-handoff.jpg)

| 三层关系 | 需要说清楚什么 | 没说清楚的后果 |
| --- | --- | --- |
| **工作如何流动** | 谁负责什么；上游要交付哪些成果、证据、当前状态和未决问题 | 箭头退化为一句“我做完了”，下游只能重新推断 |
| **结果如何校准** | 谁独立检查；检查失败后是重试、退回、换路还是换人 | 评审只能提意见，却不能改变结果 |
| **系统如何停下或改方向** | 谁能阻断执行；谁能根据长期结果调整目标和规则 | 所有节点只会向前，一起走偏也停不下来 |

设计失败路径时，还要明确退回给谁、允许重试几次、何时升级给人，以及已经消耗多少预算。节点一旦能够调用工具或修改真实环境，权限也要跟随角色和阶段收紧，不能让负责检查的节点顺手修改自己正在检查的结果。

例如，实现 Loop 交出代码和测试证据；检查 Loop 不依赖它对自己的解释，而是直接运行测试、核对约束并寻找反例。检查失败后，工作真正被退回；如果发现最初目标有问题，则交给方向 Loop 或人重新判断。只有检查结果能够改变后续路径，它才不是一个普通的下一步。

传统工作流主要决定下一步运行哪个步骤。Graph Engineering 面对的节点可能自己找路、修改计划、调用工具甚至改变真实环境，因此还必须决定谁有权判断、交接哪些证据、谁能够否决，以及何时回到人工处理。

## 十三、固定边界与动态图

社区讨论中既有提前画好的固定路线，也有 Agent 在运行时临时拆出的任务图，还有根据任务难度增减角色的设想。

更实际的做法可能是：权限、验收、预算和几个必须停下来的位置提前确定；至于本次任务拆成多少子任务、走哪条支路、是否并行，则允许在边界内动态调整。

![固定外层边界与动态内部任务图](/images/extra/X6/12-dynamic-boundaries.jpg)

有些人把运行过程中临时形成的任务和依赖称为 **Work Graph**。这个词有助于理解，但目前不是统一术语，也不是已经验证的结论；Asana 也早已把 Work Graph® 用于另一套工作数据模型。

## 十四、用状态机理解 Graph，但不要混为一谈

到了 Graph Engineering，多个带状态的执行节点被连接成图。任务可以分支、汇合、回退，也可以在必要时停下。

如果检查失败后会退回、重新规划，或者再次进入已经执行过的节点，路径中就会出现环。此时可以借有限状态机（FSM）帮助理解：系统根据结果在不同状态间切换，直到满足退出条件。

![从 Prompt、Context、Harness、Loop 到状态图的程序结构类比](/images/extra/X6/13-fsm-analogy.jpg)

FSM 只是理解程序控制关系的一种类比，不能直接作为 Graph Engineering 的技术定义。图论和状态机并不是新概念，变化来自 Agent 节点本身：它们能够拆任务、改计划、调用工具，还可能修改真实环境。过去用于工作流和分布式系统的控制方法，需要被重新用于这些更自主、也更不确定的节点。

## 十五、什么时候值得使用 Graph

Graph 并不意味着 Agent 越多越好。它更适合以下任务：

- 能够拆成相对独立的部分；
- 不同部分需要不同的信息、工具或权限；
- 中间结果能够单独检查；
- 局部失败后，希望只重做局部。

如果任务很小、每一步严格依赖上一步，或者多个 Agent 会频繁修改同一份内容，一个清楚的 Loop 往往更合适。Anthropic 的多 Agent Research 系统适合广泛搜索和并行探索，但其工程文章也指出，多 Agent 系统的 Token 使用量约为普通聊天的 15 倍；Google Research、Google DeepMind 与 MIT 的实验则发现，多 Agent 在可并行任务上可能提升，在严格顺序任务上反而可能下降。

![判断任务是否适合 Graph](/images/extra/X6/14-graph-suitability.jpg)

看到一张 Graph，可以先问三个问题：

1. 它比一个 Loop 多解决了什么关系？
2. 检查结果能不能真正退回、换路或叫停？
3. 增加的协调成本，是否小于独立判断和局部恢复带来的收益？

从一个清楚的 Loop 开始。只有任务能够拆开、中间结果能够单独验收，并且确实需要不同信息、工具或权限时，Graph 才更可能带来净收益。

## 十六、从 Harness、Loop 到 Graph

Harness 解决模型怎样在真实环境中工作：它提供工具、记忆、上下文、状态、权限、错误处理和验证。

Loop 解决一个执行者怎样根据反馈持续前进：它反复观察、行动、验证和调整，直到满足退出条件。

Graph 解决多个执行单元怎样分工、交接、纠偏和停手：它组织 Agent Loop、工具、确定性程序、验证器或人工节点，并让检查结果、失败路径和方向判断真正改变系统后续行为。

![Harness、Loop 与 Graph Engineering 的关系总结](/images/extra/X6/15-loop-graph-summary.jpg)

Graph Engineering 还没有公认定义，也不必急着划定边界。Loop 没有死亡，它仍然负责让局部工作前进。执行、检查和方向判断需要拆开，是采用 Graph 的常见信号之一；分支、汇合、并行、权限隔离或局部恢复等关系，也可能让 Graph 产生价值。

真正需要设计的，是关系能否生效：交接是否包含成果和证据，检查能否改变后续路径，方向错误时谁来重新判断。

做不到这些，Graph 只是一张更复杂、更昂贵的流程图；做到这些，工程问题才会从“一个 Agent 怎样反复做事”转变为“多个 Loop 怎样协作，又怎样避免一起走偏”。

## 参考资料

> 本文主要参考自两篇技术文章：
>
> - [《深度“解剖”AI Agent Harness》](https://mp.weixin.qq.com/s/pKAr2hX4LfhUMMeev0HU1w)
> - [《聊聊 Graph Engineering —— 别让一个 Agent 既当运动员又当裁判》](https://mp.weixin.qq.com/s/Y6JmaXdc_8AWhW7OVM0EBg)
>

### Harness 相关

1. [《The Anatomy of an Agent Harness》](https://x.com/akshay_pachaar/status/2041146899319971922)
2. [《LangChain “不务正业”，居然从零造了个数据库？》](https://mp.weixin.qq.com/s?__biz=Mzk3NTE2NzU5NQ==&mid=2247491172&idx=1&sn=c52ebb2fdad4e08231bf2ff7eecf50f8)

### Graph 相关

1. [Peter Steinberger：“Are we still talking loops or did we shift to graphs yet?”](https://x.com/steipete/status/2078277297791189132)
2. [Hamel Husain：《Loop Engineering Is Dead. Enter Graph Engineering.》](https://x.com/HamelHusain/status/2078346425621237935)
3. [Hamel Husain：“Nobody knows what it is.”](https://x.com/HamelHusain/status/2079224401267224677)
4. [Akshay Pachaar：《The four types of agent loops》](https://x.com/akshay_pachaar/status/2076748259377516782)
5. [Vaibhav Sisinty：“You need loops watching loops”](https://x.com/VaibhavSisinty/status/2078646016568606961)
6. [Anthropic：《How we built our multi-agent research system》](https://www.anthropic.com/engineering/multi-agent-research-system)
7. [Google Research：《Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work》](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)

### Graph 原文延伸阅读

- [Peter Steinberger：《Are we still talking loops or did we shift to graphs yet?》](https://x.com/steipete/status/2078277297791189132)
- [社区长帖：《A graph is a map of loops + checkpoints》](https://x.com/shannholmberg/status/2079096565344739643)
- [社区长帖：《You need loops watching loops》](https://x.com/VaibhavSisinty/status/2078646016568606961)
- [Rahul：《Prompt / Context / Harness / Loop / Graph Engineering》](https://x.com/sairahul1/status/2078781824160166070)
- [ZeroZ_JQ：《从 Prompt、Context、Harness、Loop 到 Graph 的程序结构类比》](https://x.com/ZeroZ_JQ/status/2079512381294879005)
- [IntuitMachine：《关于 Loop Engineering 与 Graph Engineering 的讨论》（原文列出的链接，当前可能不可访问）](https://x.com/IntuitMachine/status/2078419526354378975)
- [LangGraph：《Graph API》](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Microsoft AutoGen：《GraphFlow》](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/graph-flow.html)
- [Anthropic：《How we built our multi-agent research system》](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Google Research：《Towards a Science of Scaling Agent Systems》](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
