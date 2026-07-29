import { defineConfig } from 'vitepress'
// https://vitepress.dev/reference/site-config

// 1. 获取环境变量并判断
// 如果环境变量 EDGEONE 等于 '1'，说明在 EdgeOne 环境，使用根路径 '/'
// 否则默认是 GitHub Pages 环境，使用仓库子路径 '/easy-data-x-ai/'
const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/easy-data-x-ai/'

export default defineConfig({
  lang: 'zh-CN',
  title: "Easy Data X AI",
  description: "面向所有 AI 爱好者的 Data 与 AI 基础知识入门教程",
  base: baseConfig,
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/datawhale-logo.png',
    nav: [
      { text: 'GitHub 仓库', link: 'https://github.com/datawhalechina/easy-data-x-ai' },
      { text: '社区在线课堂', link: 'https://open.oceanbase.com/course/760' },
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      { text: '《Easy Data X AI 课程介绍》', link: '/course-intro' },
      { text: '学习 FAQ', link: '/faq' },
      {
        text: '课前导读与公共基础',
        collapsed: true,
        items: [
          { text: 'F0：课前闲聊 —— OpenClaw 为什么越用越好用？', link: '/base_knowledge/F0 课程稿：课前闲聊 —— OpenClaw 为什么越用越好用？' },
          { text: 'F1：AI 必知必会（一）—— 大模型的本质与边界', link: '/base_knowledge/F1 课程稿：AI 必知必会（一） —— 大模型的本质与边界' },
          { text: 'F2：AI 必知必会（二）—— AI Agent 全景图', link: '/base_knowledge/F2 课程稿：AI 必知必会（二） —— AI Agent 全景图' }
        ]
      },
      {
        text: '第二章：道篇',
        collapsed: true,
        items: [
          { text: 'P1：AI Agent 场景识别', link: '/pm/P1 课程稿：AI Agent 场景识别' },
          { text: 'P2：Agentic RAG 产品设计', link: '/pm/P2 课程稿：Agentic RAG 产品设计' },
          { text: 'P3：Agent 记忆系统设计', link: '/pm/P3 课程稿：Agent 记忆系统设计' },
          { text: 'P4：Skill 与 Agent 知识管理', link: '/pm/P4 课程稿：Skill 与 Agent 知识管理' },
          { text: 'P5：综合案例与度量', link: '/pm/P5 课程稿：综合案例与度量' }
        ]
      },
      {
        text: '第三章：术篇',
        collapsed: true,
        items: [
          { text: 'D1：大模型 API 基础', link: '/dev/D1 课程稿：大模型 API 工程化基础' },
          { text: 'D2：AI 应用的数据层', link: '/dev/D2 课程稿：AI 应用的数据层' },
          { text: 'D3：Agentic RAG 实战', link: '/dev/D3 课程稿：Agentic RAG 实战' },
          { text: 'D4：Agent 开发与记忆系统', link: '/dev/D4 课程稿：Agent 开发与记忆系统' },
          { text: 'D5：课程总结', link: '/dev/D5 课程稿：课程总结' }
        ]
      },
      {
        text: '第四章：扩展篇',
        collapsed: true,
        items: [
          {
            text: 'X1：探究 AI Agent 记忆系统',
            collapsed: true,
            items: [
              { text: '系列导读', link: '/extra/X1 探究 AI Agent 记忆系统：从遗忘曲线到永久记忆' },
              { text: 'X1-1：记忆的生命周期工程', link: '/extra/X1-1 记忆的生命周期工程' },
              { text: 'X1-2：记忆的边界与信任', link: '/extra/X1-2 记忆的边界与信任' },
              { text: 'X1-3：从记忆到认知', link: '/extra/X1-3 从记忆到认知' },
              { text: '延伸：AI Memory 系统架构', link: '/extra/X1-4 AI Memory系统架构的构思与随想' },
            ]
          },
          { text: 'X2：多 Skill 与上下文工程（P4 伴读）', link: '/extra/X2 多 Skill 给上下文工程带来的麻烦：如何应对 Agent「爆上下文」' },
          { text: 'X3：混合检索与统一数据基座', link: '/extra/X3 从零到一上手混合检索：AI Native 统一数据基座实战' },
          { text: 'X4：数据湖库与多模数据降本（共建中）', link: '/extra/X4 海量 AI Agent 多模数据降本：数据湖库登场' },
          { text: 'X5：从 Skill 到 MCP Tool（P4 伴读）', link: '/extra/X5 从 Skill 到 MCP Tool' },
          { text: 'X6：Harness、Loop 与 Graph（D1/D3 进阶）', link: '/extra/X6 从 Harness 到 Loop，再到 Graph Engineering' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/easy-data-x-ai' }
    ],

    editLink: {
      pattern: 'https://github.com/datawhalechina/easy-data-x-ai/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    outline: {
      label: '本页目录',
      level: [2, 3]
    },
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },
    sidebarMenuLabel: '课程目录',
    returnToTopLabel: '返回顶部',
    skipToContentLabel: '跳转到正文',
    darkModeSwitchLabel: '切换主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    notFound: {
      title: '页面没有找到',
      quote: '链接可能已更新。你可以回到课程首页，或使用站内搜索继续学习。',
      linkLabel: '返回首页',
      linkText: '返回课程首页'
    },

    footer: {
      message: 'Built with VitePress | <a href="https://github.com/datawhalechina/easy-data-x-ai" target="_blank">GitHub 仓库</a>',
      copyright: 'Licensed under CC BY-NC-SA 4.0'
    }
  }
})
