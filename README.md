# 科研课堂 课程论文：碳—股风险联动 ERNN

本仓库包含课程论文实证部分的代码与结果产出。

## 目录结构

```
.
├── code/        # 实证代码（step1 ~ step8 流水线 + plot 脚本）
├── output/      # 全部产出（CSV 表 + figs/ 图表）
├── datastreamm/ # 仅保留 `数据清单.md`（原始数据集不进仓）
└── README.md
```

## 数据说明

原始数据集（EUA 期货 / datastream / 5000 数据 等，合计约 2.3GB）**不进仓库**，仅在
`datastreamm/数据清单.md` 中维护数据来源清单。复现实验时若需从原始数据重跑 step1，
请按数据清单单独同步原始 xlsx。

清洗后的主数据 `output/master_data.csv` 已入仓，可直接驱动后续 step2 ~ step8。

## 复现流程

```bash
# 准备虚拟环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # 如未生成，按 step 脚本 import 现装

# 已有 master_data.csv 时可跳过 step1，从 step2 开始
python code/step2_descriptive_and_linkage.py
python code/step2b_granger_robustness.py
python code/step3_ernn_model.py
python code/step3b_ernn_minimal_pipeline.py
python code/step4_baselines.py
python code/step5_main_experiment.py
python code/step5b_plot_results.py
python code/step6_robustness.py
python code/step6b_plot_robustness.py
python code/step7_quantile_coverage.py
python code/step8_rolling_forecast.py
python code/step8b_plot_rolling.py
```

## 文档

论文与项目方案在飞书云文档维护，本仓库不包含。
