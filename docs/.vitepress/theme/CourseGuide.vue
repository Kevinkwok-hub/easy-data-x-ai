<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

type Guide = {
  eyebrow: string
  title: string
  goal: string
  prerequisite: string
  nextLabel: string
  nextLink: string
}

const guides: Record<string, Guide> = {
  'base_knowledge/F1 课程稿：AI 必知必会（一） —— 大模型的本质与边界.md': {
    eyebrow: '共同基础',
    title: '先建立同一套语言',
    goal: '理解模型能力边界，以及数据为什么决定 AI 应用的上限。',
    prerequisite: '可选：先读 F0 课前导读',
    nextLabel: '继续学习 F2',
    nextLink: '/base_knowledge/F2 课程稿：AI 必知必会（二） —— AI Agent 全景图',
  },
  'pm/P1 课程稿：AI Agent 场景识别.md': {
    eyebrow: '道篇起点',
    title: '从产品判断开始',
    goal: '判断一个需求是否适合做 Agent，再进入 RAG、Memory 与 Skill 设计。',
    prerequisite: '建议先完成 F1、F2',
    nextLabel: '继续学习 P2',
    nextLink: '/pm/P2 课程稿：Agentic RAG 产品设计',
  },
  'dev/D1 课程稿：大模型 API 工程化基础.md': {
    eyebrow: '术篇起点',
    title: '从可运行代码开始',
    goal: '跑通模型 API 与 Tool Use，为后续数据层和 Agentic RAG 打基础。',
    prerequisite: '建议先完成 F1、F2',
    nextLabel: '继续学习 D2',
    nextLink: '/dev/D2 课程稿：AI 应用的数据层',
  },
  'extra/X1 探究 AI Agent 记忆系统：从遗忘曲线到永久记忆.md': {
    eyebrow: '扩展篇起点',
    title: '在主线之后继续深挖',
    goal: '把 P3 的产品判断和 D4 的基础接入，推进到记忆生命周期工程。',
    prerequisite: '建议先完成 P3、D4',
    nextLabel: '阅读 X1 上篇',
    nextLink: '/extra/X1-1 记忆的生命周期工程',
  },
}

const { page } = useData()
const guide = computed(() => guides[page.value.relativePath])
</script>

<template>
  <aside v-if="guide" class="course-guide" aria-label="课程学习提示">
    <div class="course-guide__marker" aria-hidden="true"></div>
    <div class="course-guide__content">
      <p class="course-guide__eyebrow">{{ guide.eyebrow }}</p>
      <p class="course-guide__title">{{ guide.title }}</p>
      <p>{{ guide.goal }}</p>
      <div class="course-guide__footer">
        <span>{{ guide.prerequisite }}</span>
        <a :href="withBase(guide.nextLink)">{{ guide.nextLabel }} <span aria-hidden="true">→</span></a>
      </div>
    </div>
  </aside>
</template>
