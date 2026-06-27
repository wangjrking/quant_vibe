# research archive 目录

本目录用于保存不再作为 `quant/main` 主干入口，但仍有研究证据、复现或 provenance 价值的研究脚本。

## 归档原则

- 归档脚本不得作为当前标准研究入口默认调用。
- 归档前必须完成路径级引用扫描。
- 与 production / exploration / code_snapshot / research queue 有关的脚本，必须保留分组语义和证据链接。
- 归档动作必须记录原路径、目标路径、责任智能体、原因和批次。

## 推荐子目录

| 子目录 | 用途 |
| --- | --- |
| `formal_5d10d/` | formal 5D/10D 研究家族和策略库复现证据 |
| `formal_l4/` | formal L4、latest formal、formal state 相关研究家族 |
| `diagnostics_20260620_20260621/` | 早期诊断、确认、探针和锚点研究脚本 |
| `active_research_provenance/` | 当前仍与 research-only 候选 provenance 有关的脚本 |
| `strategy_oneoff/` | 已完成且不再作为入口的一次性策略研究脚本 |

## 必要 manifest 字段

实际迁移前必须建立 manifest，至少包含：

```json
{
  "actor": "commander-agent",
  "batch_id": "YYYYMMDDTHHMMSSZ",
  "group_name": "formal_5d10d",
  "reason": "历史研究归档证据链",
  "family_rule": "research_formal_5d10d_*.py",
  "family_members": [],
  "moves": [
    {
      "original_path": "quant/main/example.py",
      "archived_path": "quant/main/research/archive/formal_5d10d/example.py",
      "evidence_links": [],
      "still_referenced_by": []
    }
  ],
  "approval_scope": "审计复核后执行"
}
```

## 禁止事项

- 不得把当前标准入口移动到本目录。
- 不得用 archive 替代正式 manifest 或策略归档。
- 不得在未建立 manifest 的情况下批量移动脚本。
