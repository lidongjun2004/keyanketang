"""生成论文开头的研究框架流程图（图 1.1）。

老师建议 #2：放在开头、结构清晰的流程图例。
该图同时承担"任务分析"的可视化职责（老师建议 #3）：自上而下展示
研究问题 → 数据 → 联动识别 → 方法选择 → 主实验 → 稳健性 → 结论
的完整逻辑链，让读者一眼看清各环节为什么存在、如何串联。
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
FIGS_ROOT = OUTPUT_ROOT / "figs"

# 每个环节：(标题, 正文, 填充色, 边框色)
STAGES = [
    ("研究问题",
     "碳—股市场之间是否存在尾部风险联动？联动是单向还是双向？\n"
     "用 ERNN 预测尾部风险，能否优于 L-ER / QRNN / QRF / Q-GBM？",
     "#fde7e9", "#d62728"),
    ("① 数据构建（第四章）",
     "EUA 期货主力合约连续序列 + MSCI 全球能源/材料股指 + 3 个宏观控制变量\n"
     "→ 2018-01-03 ~ 2025-12-15 共 2028 个交易日的日度面板",
     "#e7f0fb", "#1f77b4"),
    ("② 联动识别：为什么值得预测（5.1 节）",
     "60 日滚动相关（弱正相关）+ Granger 因果检验\n"
     "价格层面「股→碳」单向因果；风险层面（|收益|）碳—股双向显著溢出 (p<0.001)",
     "#e7f0fb", "#1f77b4"),
    ("③ 方法选择：为什么用 ERNN（第三章）",
     "expectile 非对称平方损失处处可导、训练更稳，神经网络可拟合非线性\n"
     "对照 4 个基准：线性 L-ER / 非线性分位数 QRNN、QRF、Q-GBM",
     "#fff4e6", "#ff7f0e"),
    ("④ 主实验与评价（5.2~5.4 节）",
     "3 个方向 × 7 个 τ × 5 个模型 = 105 组实验\n"
     "双损失口径（Pinball + Expectile 平方损失）+ R² + DM 检验 + 预测区间校准",
     "#fff4e6", "#ff7f0e"),
    ("⑤ 稳健性检验（第六章）",
     "A 剔除 COVID 极端期 · B 网络结构 J 敏感性 · C 加宏观控制 · D 滚动窗口重训\n"
     "全方位检验 ERNN 优势的边界条件",
     "#eaf7ea", "#2ca02c"),
    ("⑥ 主要结论（第七章）",
     "ERNN 与线性 L-ER 基本持平（日度线性信号已很强），但显著优于其他非线性模型；\n"
     "「股→碳」方向预测优势更强；适合「周期性重训 + 较长窗口」的尾部风险预警",
     "#ede7f6", "#7b3ff2"),
]


def main():
    n = len(STAGES)
    box_height = 1.0
    gap = 0.55
    box_width = 9.4
    x_left = 0.3
    fig_height = n * (box_height + gap) + 0.6
    figure, axis = plt.subplots(figsize=(10.5, fig_height))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, fig_height)
    axis.axis("off")

    centers = []
    for index, (title, body, face, edge) in enumerate(STAGES):
        y_top = fig_height - 0.3 - index * (box_height + gap)
        y_bottom = y_top - box_height
        y_center = (y_top + y_bottom) / 2
        centers.append((x_left + box_width / 2, y_top, y_bottom))
        box = FancyBboxPatch(
            (x_left, y_bottom), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=2.0,
        )
        axis.add_patch(box)
        axis.text(x_left + 0.25, y_center + 0.27, title,
                  fontsize=12.5, fontweight="bold", color=edge, va="center", ha="left")
        axis.text(x_left + 0.25, y_center - 0.18, body,
                  fontsize=9.6, color="#222222", va="center", ha="left", linespacing=1.4)

    # 自上而下的箭头
    for index in range(n - 1):
        _, _, y_bottom_upper = centers[index]
        x_center, y_top_lower, _ = centers[index + 1]
        arrow = FancyArrowPatch(
            (x_center, y_bottom_upper), (x_center, y_top_lower),
            arrowstyle="-|>", mutation_scale=20, linewidth=2.0, color="#555555",
        )
        axis.add_patch(arrow)

    axis.text(5.0, fig_height - 0.12, "图 1.1  研究框架与技术路线",
              fontsize=14, fontweight="bold", ha="center", va="top")

    output_path = FIGS_ROOT / "fig_1_1_framework.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


if __name__ == "__main__":
    main()
