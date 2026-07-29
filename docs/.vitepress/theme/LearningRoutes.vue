<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<template>
  <section class="learning-routes" aria-labelledby="learning-routes-title">
    <div class="learning-routes__intro">
      <p class="learning-routes__eyebrow">两条路线，一个共同起点</p>
      <h2 id="learning-routes-title">先完成 F1、F2，再按角色选择主线</h2>
      <p>道篇负责建立判断力，术篇负责把判断变成可运行、可观测的工程结果。你也可以两条路线交叉学习。</p>
    </div>
    <div class="learning-routes__tracks">
      <article>
        <span class="learning-routes__index">P</span>
        <div>
          <h3>产品与决策路线</h3>
          <p>P1 → P2 → P3 → P4 → P5</p>
          <a :href="withBase('/base_knowledge/F1 课程稿：AI 必知必会（一） —— 大模型的本质与边界')">
            从共同基础开始 <span aria-hidden="true">→</span>
          </a>
        </div>
      </article>
      <article>
        <span class="learning-routes__index">D</span>
        <div>
          <h3>开发与工程路线</h3>
          <p>D1 → D2 → D3 → D4 → D5</p>
          <a :href="withBase('/base_knowledge/F1 课程稿：AI 必知必会（一） —— 大模型的本质与边界')">
            从共同基础开始 <span aria-hidden="true">→</span>
          </a>
        </div>
      </article>
    </div>
    <nav class="learning-routes__shortcuts" aria-label="其他学习入口">
      <a :href="withBase('/faq#如何运行示例代码')">只跑示例代码</a>
      <a :href="withBase('/extra/X1 探究 AI Agent 记忆系统：从遗忘曲线到永久记忆')">进入扩展篇</a>
    </nav>
  </section>
</template>
