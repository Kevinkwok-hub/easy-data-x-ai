<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

type Completion = {
  outcomes: string[]
  nextLabel: string
  nextLink: string
}

const completions: Record<string, Completion> = {
  'base_knowledge/F1 课程稿：AI 必知必会（一） —— 大模型的本质与边界.md': {
    outcomes: ['能解释大模型为什么会幻觉和过时', '能说明数据质量如何影响应用上限'],
    nextLabel: '进入 F2 Agent 全景图',
    nextLink: '/base_knowledge/F2 课程稿：AI 必知必会（二） —— AI Agent 全景图',
  },
  'base_knowledge/F2 课程稿：AI 必知必会（二） —— AI Agent 全景图.md': {
    outcomes: ['能画出 Agent 的核心组件', '能根据角色选择道篇或术篇路线'],
    nextLabel: '回到课程介绍选择路线',
    nextLink: '/course-intro',
  },
  'pm/P1 课程稿：AI Agent 场景识别.md': {
    outcomes: ['能用三维度判断需求是否适合 Agent', '能为高风险失败设置人工兜底'],
    nextLabel: '进入 P2 RAG 产品设计',
    nextLink: '/pm/P2 课程稿：Agentic RAG 产品设计',
  },
  'pm/P2 课程稿：Agentic RAG 产品设计.md': {
    outcomes: ['能用六段链路定位 RAG 问题', '能把产品问题转成 D2、D3 的验收证据'],
    nextLabel: '去 D3 查看工程实现',
    nextLink: '/dev/D3 课程稿：Agentic RAG 实战',
  },
  'pm/P3 课程稿：Agent 记忆系统设计.md': {
    outcomes: ['能区分短期与长期记忆', '能定义写入、遗忘、冲突与权限策略'],
    nextLabel: '去 D4 接入记忆系统',
    nextLink: '/dev/D4 课程稿：Agent 开发与记忆系统',
  },
  'pm/P4 课程稿：Skill 与 Agent 知识管理.md': {
    outcomes: ['能判断经验何时应沉淀为 Skill', '能定义 Skill 的边界与加载策略'],
    nextLabel: '去 X5 把 Skill 变成 MCP Tool',
    nextLink: '/extra/X5 从 Skill 到 MCP Tool',
  },
  'pm/P5 课程稿：综合案例与度量.md': {
    outcomes: ['能把质量、业务价值和成本放进同一度量框架', '能为上线与迭代设置门槛'],
    nextLabel: '查看 D5 课程总验收',
    nextLink: '/dev/D5 课程稿：课程总结',
  },
  'dev/D1 课程稿：大模型 API 工程化基础.md': {
    outcomes: ['能安全调用模型 API', '能处理流式输出、多工具调用和异常返回'],
    nextLabel: '进入 D2 数据层',
    nextLink: '/dev/D2 课程稿：AI 应用的数据层',
  },
  'dev/D2 课程稿：AI 应用的数据层.md': {
    outcomes: ['能完成切分、写入与多路检索', '能解释纯向量与混合检索的边界'],
    nextLabel: '进入 D3 Agentic RAG',
    nextLink: '/dev/D3 课程稿：Agentic RAG 实战',
  },
  'dev/D3 课程稿：Agentic RAG 实战.md': {
    outcomes: ['能跑通 Agent 与 seekdb 的完整链路', '能用 60 条评测验证检索、引用、拒答与调用量'],
    nextLabel: '进入 D4 记忆系统',
    nextLink: '/dev/D4 课程稿：Agent 开发与记忆系统',
  },
  'dev/D4 课程稿：Agent 开发与记忆系统.md': {
    outcomes: ['能为 Agent 接入跨会话记忆', '能验证隔离、检索和遗忘行为'],
    nextLabel: '进入 D5 课程总结',
    nextLink: '/dev/D5 课程稿：课程总结',
  },
  'dev/D5 课程稿：课程总结.md': {
    outcomes: ['能串起 D1～D4 的工程主线', '能选择一个真实问题继续完成课程项目'],
    nextLabel: '查看扩展篇',
    nextLink: '/extra/X1 探究 AI Agent 记忆系统：从遗忘曲线到永久记忆',
  },
}

const { page } = useData()
const completion = computed(() => completions[page.value.relativePath])
</script>

<template>
  <section v-if="completion" class="course-completion" aria-labelledby="course-completion-title">
    <p class="course-completion__eyebrow">本节验收</p>
    <h2 id="course-completion-title">学完后，你应该能够</h2>
    <ul>
      <li v-for="outcome in completion.outcomes" :key="outcome">{{ outcome }}</li>
    </ul>
    <a :href="withBase(completion.nextLink)">
      {{ completion.nextLabel }} <span aria-hidden="true">→</span>
    </a>
  </section>
</template>
