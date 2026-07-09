# dev_log 写入示例

## 示例 1：需求分析

```
### 2026-06-15
需求分析(项目/闭环目标): 初始设计

- 设计了 5 层检索管线架构（Query Builder → Search Provider → 去重+评分 → 页面抓取 → Claude API 提取）
- 确认 11 个查询维度，写好了 prompt 模板
- 原因：面临批量查询大量仪器自动化适配信息的任务，需要统一的自动化管线而非手工逐台搜索
- 结果：产出了架构设计文档、`prompt.txt` 初版、`instrument_loop.py` 骨架
```

## 示例 2：代码生成

```
### 2026-06-20
代码生成(instrument_loop): 实现搜索与去重主循环

- 新增 Query Builder 与 Search Provider 接口封装
- 接入 URL 去重与基础评分逻辑
- 结果：`instrument_loop.py` 可单型号端到端跑通
```

## 示例 3：用户推翻方案

```
### 2026-06-24
文档生成(日志机制): 由模型使用说明迁移至 dev_log

- 原方案：维护 `模型使用说明.md` + model-usage-recorder Skill
- 用户意见：不再维护模型使用说明，Skill 放项目级
- 最终采纳：新建 `.cursor/skills/dev-log-writer/`，统一写 `dev_log.md`
```
