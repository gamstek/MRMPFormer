---
name: dev-log-writer
description: Updates dev_log.md with project overview and development timeline entries. Use after completing code generation, modifications, tests, or documentation; when user asks to sync/update dev log; or when CLAUDE.md mandates logging.
---

# 项目开发日志 Skill

向项目根目录 `dev_log.md` 追加或更新开发记录。详细模板见 [template.md](template.md)，示例见 [examples.md](examples.md)。

## 何时调用

- 完成**生成**、**修改代码**、**测试**、**文档**任务后（主对话、Task 子 Agent、外部 LLM 均适用）
- 用户要求「同步更新日志」「写 dev log」
- `CLAUDE.md` 强制触发时

## 何时不写

- 纯解释性问答、未落文件的闲聊讨论
- 用户明确说「先别记日志」

## 写入规则

### 项目概述（按需更新）

子节：`目标` / `输入` / `输出` / `方法介绍`

仅在以下情况更新概述，**不要每次任务都改**：
- 项目定位、范围、架构发生实质变化
- 首次创建 `dev_log.md`

### 开发时间线（每次必写）

- 按日期分组：`### YYYY-MM-DD`
- 条目格式：`<类型>(<作用域>): <简要描述>`
- 类型：`需求分析` / `数据建模` / `代码生成` / `调试` / `文档生成` / `重构` / `测试` / `其他`
- 作用域：模块、文件或功能范围
- 正文用 `-` 列表，建议包含：
  - 做了什么（要点）
  - 原因：为什么做（可选，决策类任务建议写）
  - 结果：产出或影响

### 用户推翻/修正

若用户修改、否定或推翻模型方案且导致方向变化，在时间线追加一条，说明原方案摘要、用户意见、最终采纳方案。

### 虚构数据

生成含学生信息等演示数据时，在时间线条目末尾追加：

> 以上学生相关数据均为虚构，仅供演示，不涉及任何真实个人隐私信息。

## 执行步骤

1. 检查 `dev_log.md` 是否存在；不存在则按 [template.md](template.md) 创建
2. 读取现有内容，定位今日日期分组（无则新建 `### YYYY-MM-DD`）
3. 判断项目概述是否需要更新；需要则修订对应子节
4. 在今日分组下追加一条时间线条目
5. 保持 Markdown 结构与既有条目风格一致

## 质量检查

- [ ] 条目格式为 `<类型>(<作用域>): <简要描述>`
- [ ] 同一任务只写一条，不重复堆砌
- [ ] 概述未被无关任务频繁改动
- [ ] 写完后文件可正常渲染（标题层级、列表缩进正确）
