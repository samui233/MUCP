# Chapter IV IEEE figures

该目录保存 Fig. 1 和 Fig. 2 的完整复现材料。正式图片不是从旧 PNG 中读取或数字化得到的；`plot_ieee_figures.py` 直接聚合 `results/chapter4_sample_metrics/` 中 A–G 的逐样本 NPZ 结果。

## 运行

需要 Python、NumPy、Matplotlib、Pillow、Poppler `pdffonts`，以及 Arial 字体。

```bash
cd reproducibility/chapter4_figures
python plot_ieee_figures.py
```
绘图统一采用连通链路测试子集：一个 16 帧序列中所有帧的最优波束增益均不得低于 −100 dB。该规则只依赖原始信道增益，不依赖任一模型的预测误差，并对 A–G 使用同一个掩码：

- CSI（20 Hz）保留 2040/2133 个测试序列；
- 波束（5 Hz）保留 827/881 个测试序列。

仓库已保存生成好的掩码。若本地具有完整 prepared dataset，可重新生成：

```bash
python build_connected_link_filter.py \
  --dataset-root /path/to/simart_multimodal_v1 \
  --threshold-db -100
```


输出文件位于：

- `figures/chapter4_ieee/`：7.16 inch 宽的矢量 PDF、600 dpi PNG 和灰度预览；
- `results/chapter4_ieee_plot_data/fig1_plot_data.csv`：Fig. 1 的 7 行绘图数据；
- `results/chapter4_ieee_plot_data/fig2_plot_data.csv`：Fig. 2 的 56 行绘图数据；
- `results/chapter4_ieee_plot_data/figure_plot_data.json`：两张图的结构化绘图数据；
- `results/chapter4_sample_metrics/`：A–G 的 CSI 与波束逐样本统计；
- `results/chapter4_connected_filter/`：连通链路掩码、最低增益及筛选统计；
- `results/chapter4_ieee_plot_data/source_manifest.json`：逐样本结果和掩码的路径与 SHA-256；
- `results/chapter4_ieee_plot_data/pdf_fonts.txt`：PDF 字体嵌入检查；
- `results/chapter4_ieee_plot_data/validation_report.json`：尺寸与字号检查结果。

CSI 使用 20 Hz 测试序列，波束使用 5 Hz 测试序列。Fig. 1 的直接预测指标取同一测试序列的前两个目标点，Fig. 2 使用全部八个递归预测点。
