"""从已有实验记录生成论文图表；不加载模型、不训练、不重新测试。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paper_figures"
BLUE = "#32688E"
ORANGE = "#C3643C"
TEXT = "#263747"
SOURCES = set()


def read_json(relative):
    path = ROOT / relative
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(relative):
    path = ROOT / relative
    SOURCES.add(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def configure_style():
    for path in [Path("C:/Windows/Fonts/msyh.ttc"),
                 Path("C:/Windows/Fonts/simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(path)).get_name()
            break
    else:
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei"]:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        else:
            raise RuntimeError("缺少中文字体，请安装微软雅黑、黑体或 Noto Sans CJK。")
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#7A8790",
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def save_figure(fig, stem, pdf):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print("已生成：", stem, flush=True)


def architecture():
    config = read_json("models/final/experiment_config.json")
    source = ROOT / "fusion_model.py"
    SOURCES.add(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["fusion_source_sha256"]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, fill, size=10):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1, edgecolor="#82929E", facecolor=fill))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=size, linespacing=1.65)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color="#536877", shrinkA=2, shrinkB=2))

    box(.025, .70, .17, .16, "人工特征输入\n561 维", "#EDF3F8")
    box(.025, .19, .17, .16, "时间序列输入\n9 通道 × 128 点", "#EDF3F8")
    box(.25, .65, .34, .26,
        "MLP 分支\nLinear：561 → 128 → 64\nReLU + Dropout(0.3)", "#E8F0F6")
    box(.25, .075, .34, .39,
        "CNN 分支\nConv：9 → 32 → 64 → 128\n卷积核：7 / 5 / 3；BN + ReLU\n前两层后 MaxPool(2)\n全局平均池化 + Dropout(0.3)",
        "#E8F0F6", 9.2)
    box(.66, .40, .14, .20, "特征拼接\n64 + 128\n192 维", "#F8EBE3")
    box(.855, .40, .12, .20, "线性分类\n192 → 6\n六类活动", "#F8EBE3", 9.6)
    arrow((.207, .78), (.238, .78))
    arrow((.207, .27), (.238, .27))
    arrow((.602, .78), (.648, .53))
    arrow((.602, .27), (.648, .47))
    arrow((.812, .50), (.843, .50))
    ax.text(.63, .81, "64 维", ha="center", fontsize=8.5)
    ax.text(.63, .23, "128 维", ha="center", fontsize=8.5)
    ax.text(.5, 1.0, "双分支特征融合网络", ha="center", fontsize=13)
    ax.text(.5, -.02, "同一窗口的两种输入表示；两条分支从头联合训练",
            ha="center", fontsize=9, color="#5B6C78")
    fig.subplots_adjust(left=.02, right=.98, top=.92, bottom=.13)
    return fig


def cv_comparison():
    entries = []
    for name, filename in [
        ("逻辑回归", "logistic_grid_cv.csv"),
        ("SVM", "svm_grid_cv.csv"),
        ("随机森林", "forest_grid_cv.csv"),
    ]:
        best = max(read_csv("outputs/"+filename),
                   key=lambda row: float(row["cv_macro_f1"]))
        entries.append({"model": name, "mean_macro_f1": float(best["cv_macro_f1"]),
                        "fold_std_ddof0": float(best["cv_f1_std"])})
    for name, filename in [("MLP", "mlp_group_cv.csv"),
                           ("1D CNN", "cnn_group_cv.csv"),
                           ("融合模型", "fusion_group_cv.csv")]:
        rows = read_csv("outputs/"+filename)
        assert len(rows) == 5
        values = np.array([float(row["val_macro_f1"]) for row in rows])
        entries.append({"model": name, "mean_macro_f1": float(values.mean()),
                        "fold_std_ddof0": float(values.std(ddof=0))})
    entries.sort(key=lambda row: row["mean_macro_f1"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    for x, row in enumerate(entries):
        mean, std = row["mean_macro_f1"], row["fold_std_ddof0"]
        color = ORANGE if row["model"] == "融合模型" else BLUE
        ax.errorbar(x, mean, yerr=std, fmt="o", color=color,
                    markersize=7, capsize=5, elinewidth=1.4, zorder=3)
        ax.text(x, mean+std+.004, f"{mean:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(entries)), [row["model"] for row in entries])
    ax.set(xlim=(-.6, len(entries)-.4), ylim=(.84, 1.0),
           ylabel="宏平均 F1", title="训练集五折分组交叉验证")
    ax.set_xlabel("误差线为折间标准差（ddof=0），不是置信区间", labelpad=12, fontsize=9)
    ax.grid(axis="y", color="#E2E7EB", linewidth=.7)
    ax.set_axisbelow(True)
    with (OUT / "model_cv_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    return fig


def confusion_matrices(metrics):
    names = metrics["class_names"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), layout="constrained")
    image = None
    for ax, model, title in zip(axes, ["logistic", "fusion"], ["逻辑回归", "融合模型"]):
        rows = read_csv(f"outputs/final_test/{model}_confusion_matrix.csv")
        assert list(rows[0])[1:] == names
        assert [list(row.values())[0] for row in rows] == names
        matrix = np.array([[int(row[name]) for name in names] for row in rows])
        report = read_json(f"outputs/final_test/{model}_classification_report.json")
        assert matrix.shape == (6, 6)
        assert matrix.sum() == metrics["test_samples"]
        assert np.array_equal(matrix.sum(axis=1), [int(report[name]["support"]) for name in names])
        assert matrix.sum()-matrix.trace() == metrics["models"][model]["errors"]
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.1%}",
                        ha="center", va="center", fontsize=8.6,
                        color="white" if normalized[i, j] > .55 else TEXT)
        accuracy = metrics["models"][model]["accuracy"]
        ax.set_title(f"{title}（准确率 {accuracy:.2%}）", pad=12)
        ax.set_xticks(range(6), names)
        ax.set_yticks(range(6), names)
        ax.set(xlabel="预测活动", ylabel="真实活动")
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.9)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axes, fraction=.03, pad=.02)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    colorbar.set_label("占本行真实类别样本的比例", labelpad=10)
    return fig


def class_f1_comparison(metrics):
    names = metrics["class_names"]
    reports = {model: read_json(f"outputs/final_test/{model}_classification_report.json")
               for model in ["logistic", "fusion"]}
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    width = .36
    for model, offset, label, color in [
        ("logistic", -width/2, "逻辑回归", BLUE),
        ("fusion", width/2, "融合模型", ORANGE),
    ]:
        values = [reports[model][name]["f1-score"] for name in names]
        bars = ax.bar(x+offset, values, width, label=label, color=color, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=8)
    ax.set_xticks(x, names)
    ax.set(ylim=(0, 1.14), ylabel="F1", title="独立测试集各类别 F1 对比")
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(axis="y", color="#E2E7EB", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncols=2, frameon=False)
    rows = [{"activity": name,
             "logistic_f1": reports["logistic"][name]["f1-score"],
             "fusion_f1": reports["fusion"][name]["f1-score"],
             "difference_percentage_points": 100*(reports["fusion"][name]["f1-score"]
                                                  -reports["logistic"][name]["f1-score"])}
            for name in names]
    with (OUT / "class_f1_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return fig


def main():
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_json("outputs/final_test/final_test_metrics.json")
    with PdfPages(OUT / "paper_figures.pdf") as pdf:
        save_figure(architecture(), "fig01_fusion_architecture", pdf)
        save_figure(cv_comparison(), "fig02_group_cv_comparison", pdf)
        save_figure(confusion_matrices(metrics), "fig03_test_confusion_matrices", pdf)
        save_figure(class_f1_comparison(metrics), "fig04_test_class_f1", pdf)
    captions = """# 论文图注与使用说明

图1 双分支特征融合网络结构。人工特征分支输出64维表示，时序卷积分支输出128维表示，拼接为192维后完成六分类。两条分支从头联合训练，结构与冻结模型定义一致。

图2 训练集五折受试者分组交叉验证的宏平均F1。圆点为五折算术平均，误差线为五折标准差（ddof=0），不表示置信区间或显著性检验。传统模型显示训练集网格搜索选中配置的模型选择成绩；神经网络显示原始基础种子42的外层评估结果。融合模型额外两次种子复验应在正文或独立表格中报告，未混入本图。

图3 独立测试集混淆矩阵。行表示真实类别，列表示预测类别。单元格第一行为窗口数量，第二行为占该真实类别样本的比例；两幅图使用相同的0%-100%色阶。测试集共2947个窗口、9名受试者。

图4 逻辑回归与融合模型在独立测试集各活动类别上的F1。纵轴从0开始。融合模型在步行、上楼、下楼、躺卧上的F1较高，在坐与站立上的F1较低；不据此宣称所有类别均有改善。

PNG为300 dpi，可插入Word。paper_figures.pdf包含四页矢量图，便于预览及打印。model_cv_summary.csv和class_f1_summary.csv保存绘图数值。plot_manifest.json记录所有来源文件哈希。

本脚本仅读取已有记录；没有加载模型、训练、调参或重新评价测试集。论文将交叉验证和独立测试分别报告，不把两者混成一张模型排名图。
"""
    (OUT / "图注与使用说明.md").write_text(captions, encoding="utf-8")
    manifest = {
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(SOURCES)},
        "training_or_test_inference_performed": False,
    }
    (OUT / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n四张论文图已生成，另有四页矢量PDF、图注与两张数据表。")
    print("保存位置：", OUT)
    print("本步骤没有重新训练或测试模型。")


if __name__ == "__main__":
    main()
