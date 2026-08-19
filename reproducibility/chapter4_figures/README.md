# Chapter IV IEEE figures

该目录保存 Fig. 1 和 Fig. 2 的完整复现材料。正式图片不是从旧 PNG 中读取或数字化得到的；`plot_ieee_figures.py` 直接读取 `results/` 中保存的 28 个原始测试 JSON。

## 运行

需要 Python、NumPy、Matplotlib、Pillow、Poppler `pdffonts`，以及 Arial 字体。

```bash
cd reproducibility/chapter4_figures
python plot_ieee_figures.py
```

输出文件位于：

- `figures/chapter4_ieee/`：7.16 inch 宽的矢量 PDF、600 dpi PNG 和灰度预览；
- `results/chapter4_ieee_plot_data/fig1_plot_data.csv`：Fig. 1 的 7 行绘图数据；
- `results/chapter4_ieee_plot_data/fig2_plot_data.csv`：Fig. 2 的 56 行绘图数据；
- `results/chapter4_ieee_plot_data/figure_plot_data.json`：两张图的结构化绘图数据；
- `results/chapter4_ieee_plot_data/source_manifest.json`：28 个原始 JSON 的路径和 SHA-256；
- `results/chapter4_ieee_plot_data/pdf_fonts.txt`：PDF 字体嵌入检查；
- `results/chapter4_ieee_plot_data/validation_report.json`：尺寸与字号检查结果。

原始结果中，CSI 使用 20 Hz 测试序列，波束使用 5 Hz 测试序列。Fig. 2 的 C 和 F 结果由对应最佳 checkpoint 在与其他组合相同的测试集上补充评估得到。
