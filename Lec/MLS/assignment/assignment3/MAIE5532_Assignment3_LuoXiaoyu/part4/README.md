# MobileNetV2 CIFAR-10 Comprehensive Analysis

This analysis reads the existing optimization and cross-platform results.
It does not retrain the model and does not rerun QEMU or WebGPU.

## Dependencies

From the project root:

```bash
source /home/mywslubuntu/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

Matplotlib generates the report figures. `openpyxl` is only needed when
explicitly regenerating the Excel workbook.

## Generate the results

```bash
cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
python part4/part4_analysis.py
```

The command writes:

- `hardware_aware_analysis.md`
- `optimization_guidelines.md`
- JSON and CSV tables under `part4/part4_results/`
- PNG figures under `part4/part4_results/figures/`

The English report and optimization guide are written to the project root
because those locations match the final submission names. The existing
`performance_benchmarks.xlsx` is not overwritten by the default command.

To overwrite the workbook from this script:

```bash
python part4/part4_analysis.py --write-excel
```

## Convert the report to PDF

The Markdown report is intentionally generated before PDF creation. After
checking `hardware_aware_analysis.md`, convert it to the required
`hardware_aware_analysis.pdf`. Keep the figures in
`part4/part4_results/figures/` so their relative links resolve.
