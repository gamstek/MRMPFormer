---
name: code-change-logger
description: "Use when: user asks for code changes/修改代码/代码改动/新增/删除/调整/重构/修复; after any file modification via editor tools; before claiming work is done. Records structured change entries to docs/Modified.md with timestamp, type, author, scope, content, and rationale."
argument-hint: "记录代码改动到 docs/Modified.md"
user-invocable: true
disable-model-invocation: false
---

# 代码修改记录 Skill

在 `docs/Modified.md` 中记录每一次代码改动，形成可追溯的修改历史。

## 触发时机

- 用户说「记录改动」「记录修改」「记一下」「log this change」→ 立即触发
- 用户明确要求查看或维护 `docs/Modified.md`

## AI 行为规则

**每次回答末尾检查**：如果本轮回答中包含任何代码编辑操作（`replace_string_in_file`、`multi_replace_string_in_file`、`create_file`），在回答最后加一行：

> 是否登记改动人？

- 用户回复姓名 → 本轮所有编辑操作以该姓名批量写入 `docs/Modified.md`
- 用户回复「不记」「跳过」→ 本轮不写入
- 用户不回复 → 不写入，下次改代码时再问

## 不触发

- 纯只读操作（读取文件、搜索、分析、回答）
- 用户明确说「不用记」「skip logging」
- 仅创建新项目/新文件（由 `dev-log-writer` skill 覆盖）

## 执行步骤

### 1. 询问改动人（回答末尾）

在含代码编辑的回答末尾追加：

> 是否登记改动人？

### 2. 写入记录

用户回复姓名后：
1. 检查 `docs/Modified.md` 是否存在，不存在则按模板创建
2. 定位当日 `## YY-MM-DD` 标题
3. 逐条写入本轮所有编辑操作（每条编辑工具调用一条记录，改动人统一）
| **改动人** | 第 1 步获取的姓名 |
| **说明** | 改动主题 + 涉及文件及行号 + 联动文件（如有） |
| **内容** | 具体改动的精简说明（2-5 句话概括核心变更） |
| **理由** | 为什么这样改（业务需求 / 技术债务 / 性能优化 / 用户反馈等） |

### 3. 写入记录

在 `docs/Modified.md` 中按日期分组批量写入：
1. 读取文件，定位当日 `## YY-MM-DD` 标题
2. 若当日尚无记录，先创建日期标题
3. 计算当日已有记录数 N，新条目从 `N+1` 开始编号（当日递增，跨天重置为 #1）
4. 按 [条目格式](#条目格式) 逐条追加

写入后确认：文件可正常渲染，Markdown 层级正确。

## 文件模板

`docs/Modified.md` 初始内容：

```markdown
# 代码修改记录

> 本文件记录每次代码改动的详细信息，包括时间、类型、改动人、说明、内容与理由。
> 由 `code-change-logger` skill 自动维护。

---

```

## 条目格式

每条记录使用以下格式追加到对应日期分组下：

```markdown
### 修改 #N
- **改动人**: <姓名>
- **类型**: <新增|删除|调整|重构|修复|优化|其他>
- **说明**: <改动主题>；涉及 `<文件路径> L<行号>-L<行号>`；联动 `<联动文件> L<行号>`（无联动则写"无"）
- **内容**: <具体改动精简说明，2-5 句话>
- **理由**: <为什么这样改>
```

## 示例

```markdown
## 26-07-09

### 修改 #1
- **改动人**: 张三
- **类型**: 调整
- **说明**: 重构采集参数传递方式；涉及 `app/harvest/harvester.py` L42-L68；联动 `app/agent/tools.py` L15-L22
- **内容**: 将 Harvester.run() 的 6 个独立参数合并为 HarvestConfig 数据类；tools.py 中调用方同步改为构建配置对象后传入；移除 harvester.py 中冗余的默认值硬编码
- **理由**: 参数数量膨胀导致调用方可读性下降；统一为配置对象后便于扩展新来源类型

### 修改 #2
- **改动人**: 李四
- **类型**: 修复
- **说明**: 修复 PDF 解析器空指针异常；涉及 `app/document/pdf_parser.py` L88-L95；无联动
- **内容**: 在 `_extract_text()` 方法中增加 `page is None` 的提前返回判断；添加 try-except 包裹 `page.extract_text()` 调用
- **理由**: 用户反馈部分扫描版 PDF 解析时崩溃，根因是空页面对象未做防御性检查
```

## 与 dev_log.md 的区别

| | `dev_log.md` | `docs/Modified.md` |
|---|---|---|
| **粒度** | 任务级（一次完整功能） | 改动级（每次编辑操作） |
| **内容** | 功能描述 + 结果 | 具体文件行号 + 联动 + 理由 |
| **维护者** | dev-log-writer skill | code-change-logger skill |
| **受众** | 项目进度回顾 | 代码审查 + 回溯具体修改 |

两条日志互补：`dev_log.md` 看做了什么功能，`docs/Modified.md` 看改了什么代码。

## 质量检查

- [ ] 任务结束前是否一次性询问并记录了改动人？
- [ ] 时间格式为 `YY-MM-DD`
- [ ] 类型在预定义列表内
- [ ] 说明包含文件路径和行号范围
- [ ] 内容精简且覆盖核心变更
- [ ] 理由清晰，非空洞（不写"需要改"、"按要求改"）
- [ ] 每条编辑工具调用对应一条记录，不合并
- [ ] `docs/Modified.md` 格式正确，无 Markdown 渲染问题
