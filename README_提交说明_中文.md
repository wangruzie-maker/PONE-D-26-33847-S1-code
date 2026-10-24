# 中文提交说明（详见英文 README）

完整英文说明见根目录 `README.md` 与 `code/06_retrieval_sensitivity/PROTOCOL_retrieval_sensitivity.md`。

本次仓库更新要点：

1. **情感人工探针（正文 §3.2 / Table 1）**  
   400 条分层金标准；280 仅用于核实训练适配；120 留作最终评价；分类别 Precision / Recall / F1 见 `derived_data_SI/S4_holdout_FINAL_metrics.json`；编码规则与 κ=0.74 见英文 README。

2. **检索敏感度 1,074 篇（正文 §4.4 / Table 6）**  
   同标题字段 / 媒体 / 时段；OR 扩展四词组；同 §3.1 清洗；按媒体+日期+规范化标题相对 A 去重。  
   新增：`derived_data_SI/S5_retrieval_sensitivity_BA_titles.csv`、`S5_retrieval_sensitivity_composition.csv`，以及 `code/06_retrieval_sensitivity/`。

版权限制：WiseSearch 全文库与本地模型权重仍不上传。
