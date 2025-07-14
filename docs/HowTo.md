

# BioMedStatX User Guide

This guide explains how to use the BioMedStatX application: from launching the program, importing data, running statistical analyses, customizing plots, and exporting results. All information is focused on the user interface, available statistics, and practical workflow—no programming or code knowledge required.

---


## 1. Launching the Application

- Locate the `BioMedStatX.exe` file in your installation directory.
- Double-click to start. A Qt-based GUI window will open with the menus: **File**, **Analysis**, and **Help**.

---



## 2. Importing Data

1. In the **File** menu, select **Browse** to choose your data file (Excel `.xlsx`/`.xls` or CSV `.csv`).
2. For Excel files, select the worksheet you want to analyze. For CSV, this step is skipped.
3. Click **Load** to import your data into the application.



## 3. Selecting Groups & Measurement Columns

- Choose which column defines your groups (e.g., treatment, condition) in the group selection dialog.
- Select one or more numeric columns for analysis. If you select multiple columns and enable "combine columns", values are merged per group.

---


arcsin_sqrt_transform(df, dv)

## 4. Assumption Checks & Data Transformations

Before any statistical test, the app automatically checks for normal distribution and equal variances. If your data does not meet these assumptions, you will be prompted to apply a transformation (log, Box–Cox, or arcsine–sqrt) to improve suitability for analysis. You can skip or accept the suggested transformation.



## 5. Statistical Analyses

BioMedStatX automatically selects the appropriate statistical test based on your data and design. Supported analyses include:

- Two-group comparisons (independent or paired)
- Multi-group comparisons (one-way, two-way, repeated measures, mixed designs)
- Parametric and non-parametric alternatives (e.g., t-tests, ANOVA, Mann–Whitney, Kruskal–Wallis)

You do not need to choose the test yourself—the software guides you and explains the result in plain language.

---



### 5.1. Non-parametric Alternatives

If your data does not meet the requirements for standard ANOVA, the app will automatically use robust non-parametric alternatives for complex designs (e.g., non-parametric two-way or repeated measures ANOVA).

---



## 6. Post‑Hoc Comparisons

If a group comparison is significant, the app automatically performs post-hoc tests (e.g., Tukey, Dunn, Bonferroni, or Dunnett) to show which groups differ. Results are clearly displayed in the results table and as annotations on the plots.

---



## 7. Decision Tree Visualization

You can visualize the statistical decision process via **Analysis → Show Decision Tree**. The app displays a graphical flowchart showing which tests were chosen and why, with the actual path highlighted. The image can be saved for documentation.

---



## 8. Exporting Results

After your analysis, you can export all results to Excel with a single click (**File → Export Results**). The exported file contains:
- A summary of all tests and p-values
- Assumption checks
- Main results and effect sizes
- Descriptive statistics for each group
- The decision tree image
- Raw data snapshots
- Pairwise comparisons
- A chronological analysis log
Each sheet is clearly named for easy navigation.

---



## 9. Plotting & Customization

BioMedStatX creates publication-ready plots for your results. Supported plot types include:
- Bar charts (with error bars)
- Violin plots
- Boxplots
- Overlayed individual data points (with lines for paired data)

### Plot Customization

Use the **Plot Settings** dialog to adjust:
- Plot title and axis labels
- Figure size (width, height, DPI)
- Colors and hatches for each group
- Error bar style (SD/SEM, caps/lines)
- Significance annotations (letters, brackets, custom comparisons)
- Legend position, font size, and title
- Background and grid style
- Data point style (jitter, strip, swarm)
- Overlay options (paired lines, custom annotations)

All changes are shown in a live preview before saving. Most options are also available for export (PDF/PNG) and are reflected in the exported plots.

**Note:** Some advanced customizations (e.g., custom color maps or font families) may require the advanced options dialog.

---



## 10. Outlier Detection (Optional)

Under **Analysis → Detect Outliers**, you can identify and flag outliers in your data using:
- Modified Z-Score Test
- Grubbs’ Test
- Single-pass or iterative mode

Results are exported to Excel for further review.

---



## 11. Window Resizing, Scrollability, and Scaling

BioMedStatX is designed to work on all common screen sizes and resolutions:
- All windows and dialogs can be resized by dragging the edges.
- Dialogs are scrollable if content does not fit on the screen.
- The layout adapts to high-DPI screens and multi-monitor setups.
- Minimum and maximum window sizes ensure usability on both small and large screens.

**Tips:**
- If you cannot see all content, maximize the window or use the scrollbars.
- On very high-resolution screens, use your operating system’s scaling settings or adjust the app’s DPI settings (see Help menu).
- For best results, keep your graphics drivers and system libraries up to date.

If you encounter display issues, please report your OS, screen resolution, and a screenshot to the developer.

---


## 12. Quick Workflow

1. **Launch** the application.
2. **Browse** and **Load** your data file.
3. **Select** worksheet (if Excel), group column, and measurement columns.
4. **Choose** analysis type (basic or advanced).
5. **Check** assumptions; apply transforms if needed.
6. **Review** plots and decision tree.
7. **Export** results; locate your results files in your working folder.

---


---


---

### Tips & Best Practices

- Ensure your group column has consistent, non-empty labels.
- Use data transformations for highly skewed data if prompted.
- For paired designs, confirm equal sample sizes per group.
- Use the **Analysis Log** sheet in the exported Excel file for troubleshooting and detailed steps.
- Preview your plot settings before exporting for best results.
- If you encounter issues with window scaling or content visibility, maximize the window or adjust your OS scaling settings.

---

Happy analyzing!