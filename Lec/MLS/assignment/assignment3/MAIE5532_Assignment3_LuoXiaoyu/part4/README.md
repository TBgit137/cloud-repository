# Part 4: Comprehensive Analysis

Part 4 reads the existing Part 2 and Part 3 results. It does not retrain the
model and does not rerun QEMU or WebGPU.

## Install the Part 4 dependency

From the submission directory:

```bash
source /home/mywslubuntu/.venvs/maie5532/bin/activate
python -m pip install -r requirements.txt
```

`openpyxl` generates the required Excel workbook. Matplotlib generates the
report figures.

## Generate the results

```bash
cd /mnt/d/work/cloud-repository/Lec/MLS/assignment/assignment3/MAIE5532_Assignment3_LuoXiaoyu
python part4/part4_analysis.py
```

The command writes:

- `hardware_aware_analysis.md`
- `hardware_aware_analysis_zh-CN.md`
- `optimization_guidelines.md`
- `optimization_guidelines_zh-CN.md`
- `performance_benchmarks.xlsx`
- JSON and CSV tables under `part4/part4_results/`
- PNG figures under `part4/part4_results/figures/`

To generate everything except the Excel workbook:

```bash
python part4/part4_analysis.py --skip-excel
```

## Convert the report to PDF

The Markdown report is intentionally generated before PDF creation. After
checking `hardware_aware_analysis.md`, convert it to the required
`hardware_aware_analysis.pdf`. Keep the figures in
`part4/part4_results/figures/` so their relative links resolve.
