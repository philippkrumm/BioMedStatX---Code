# Statistical Analyzer User Guide (HowTo.md)

This guide ex* **Advanced Analyses**: Two‑Way ANOResults appearImage is saved as a temEach sheet is clearPlots can beResults export to a specified Excel file via `OutlierDetectionDialog` and `OutlierDetector` in **stats_functions.py**.saved automatically as PDF/PNG alongside Excel.y named for easy navigation.orary PNG and highlighted path shows actual branches.in separate sheets and on plots.A, Repeated Measures ANOVA, Mixed ANOVA via **AdvancedTestDialog** and:

  ```python
  StatisticalTester.perform_advanced_test(...)
  ```

All test logic, routing, and parameter handling is implemented in `StatisticalTester` within **stats_functions.py**.how to use the standalone `.exe` for Statistical Analyzer—from launching the app to importing data, running analyses, and exporting results—without needing any Python commands.

---

## 1. Launching the Application

- Locate the `StatisticalAnalyzer.exe` file.
- Double‑click to open. A Qt GUI window appears with the menus: **File**, **Analysis**, and **Help**.

---

## 2. Importing Data

1. In the **File** menu, choose **Browse** to select an Excel (`.xlsx`/`.xls`) or CSV (`.csv`) file.
2. Upon selection, the file’s sheets (for Excel) populate the **Worksheet** dropdown; CSV skips this step.
3. Click **Load** (or close the dialog) to read data.
4. Internally, the app uses:
   ```python
   DataImporter.import_data(
       file_path,            # Selected path
       sheet_name=...,       # Sheet index or name
       group_col=...,        # Column name for grouping
       value_cols=...,       # List of numeric measurement columns
       combine_columns=...   # Boolean to merge multiple columns
   ) → (samples: dict, df: DataFrame)
   ```

## 3. Selecting Groups & Measurement Columns

* **GroupSelectionDialog**: Pick which factor/column defines your groups.
* **ColumnSelectionDialog**: Choose one or more numeric columns for analysis. If multiple and **combine\_columns** is enabled, values across columns are merged per group.

---

## 4. Assumption Checking & Transformations

* Before any statistical test, the app runs:

  * **Shapiro–Wilk test** for normality
  * **Levene’s test** for homogeneity of variances
* If either assumption fails, the **TransformationDialog** appears, offering:

  * Log₁₀ transform (default)
  * Box‑Cox transform
  * Arcsine‑sqrt transform
* Under the hood, these map to:

  ```python
  log_transform(df, dv)
  boxcox_transform(df, dv)
  arcsin_sqrt_transform(df, dv)
  ```

## 5. Statistical Tests

* **Two-Group, Independent**: Student’s t‑test, Welch’s t‑test, Mann–Whitney U
* **Two-Group, Paired**: Paired t‑test, Wilcoxon signed‑rank
* **Multi-Group, Independent**: One‑way ANOVA, Welch ANOVA, Kruskal–Wallis
* **Advanced Analyses**: Two‑Way ANOVA, Repeated Measures ANOVA, Mixed ANOVA via **AdvancedTestDialog** and:

  ```python
  StatisticalTester.perform_advanced_test(...)
  ```

All test logic, routing, and parameter handling is implemented in `StatisticalTester` within **stats\_functions.py**. fileciteturn1file0

---

## 5.1. Non-parametric ANOVA Alternatives

For complex designs where parametric assumptions are violated, the software includes specialized non-parametric alternatives:

* **Non-parametric Two-Way ANOVA** - Using rank transformation + permutation tests
* **Non-parametric Repeated Measures ANOVA** - For within-subjects designs
* **Non-parametric Mixed ANOVA** - For mixed (between + within) designs

These are implemented in `NonParametricFactory` and provide robust alternatives when traditional non-parametric tests (like Kruskal-Wallis) are insufficient for complex designs.

---

## 6. Post‑Hoc Comparisons

When overall tests are significant, the software automatically performs appropriate post-hoc comparisons:

* Tukey’s HSD
* Dunn’s or Bonferroni‑corrected comparisons
* Dunnett’s test (control vs others)

Results appear in separate sheets and on plots. fileciteturn3file0

---

## 7. Decision Tree Visualization

Visualize the decision process via **Analysis → Show Decision Tree**, which calls:

```python
DecisionTreeVisualizer.visualize(results, output_path)
# or for Excel embedding:
DecisionTreeVisualizer.generate_and_save_for_excel(results)
```

Image is saved as a temporary PNG and highlighted path shows actual branches. fileciteturn2file4

---

## 8. Exporting Results to Excel

After analysis, choose **File → Export Results** (or it runs automatically):

```python
ResultsExporter.export_results_to_excel(results, output_file, analysis_log)
```

This creates a multi-sheet `.xlsx` with:

* **Summary** of tests and p‑values
* **Assumptions** (normality, variance)
* **Main Results** (statistics, effect sizes)
* **Descriptive Statistics** per group
* **Decision Tree** image
* **Raw Data** snapshots
* **Pairwise Comparisons**
* **Analysis Log** (chronological steps)

Each sheet is clearly named for easy navigation. fileciteturn1file0

---

## 9. Plotting & Customization

Plots are generated with Matplotlib (via Seaborn palettes) and include:

* Bar charts with SD/SEM error bars
* Overlayed individual data points (and connection lines for paired data)
* Violin or boxplots when selected

Use **PlotConfigDialog** to adjust:

* Titles & axis labels
* Figure dimensions
* Colors & hatches per group
* Significance annotations or custom comparisons

Plots can be saved automatically as PDF/PNG alongside Excel. fileciteturn1file5

---

## 10. Outlier Detection (Optional)

Under **Analysis → Detect Outliers**, configure and run:

* Modified Z‑Score Test
* Grubbs’ Test
* Single‑pass or iterative mode

Results export to a specified Excel file via `OutlierDetectionDialog` and `OutlierDetector` in **stats\_functions.py**. fileciteturn3file8

---

## 11. Quick Workflow

1. **Launch** `.exe`.
2. **Browse** & **Load** your data file.
3. **Select** worksheet (if Excel), group column, and measurement columns.
4. **Choose** analysis type (basic or advanced).
5. **Check** assumptions; apply transforms if needed.
6. **Review** plots and decision tree.
7. **Export** results; locate `<base>_results.xlsx`, `.pdf`, and `.png` in your working folder.

---

### Tips & Best Practices

* Ensure your group column has consistent labels.
* Use Box‑Cox or log transforms when skew is severe.
* For paired designs, confirm equal sample sizes per group.
* Consult the **Analysis Log** sheet for troubleshooting and detailed steps.

Happy analyzing!