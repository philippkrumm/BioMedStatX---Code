import sys
import time 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QComboBox, QPushButton, QListWidget, 
                           QTabWidget, QGroupBox, QCheckBox, QSpinBox, QColorDialog, 
                           QMessageBox, QScrollArea, QListWidgetItem, QDialog, QDialogButtonBox,
                           QGridLayout, QLineEdit, QRadioButton, QAction, QFormLayout, QAbstractItemView, QDoubleSpinBox)
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# DISABLED: Nonparametric fallbacks are not yet supported
# from nonparametricanovas import NonParametricFactory, NonParametricRMANOVA
# New class imports:
from stats_functions import (
    DataImporter, StatisticalTester, DataVisualizer, AnalysisManager, ResultsExporter, 
    UIDialogManager, OutlierDetector, OUTLIER_IMPORTS_AVAILABLE
)
import traceback

DEFAULT_COLORS = ['#3357FF', '#FF5733', '#33FF57', '#F033FF', '#FF3366', '#33FFEC']
DEFAULT_HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', '.', '*', '']

def dict_to_long_format(samples, groups):
    """
    Converts a dictionary with groups and measurements into a DataFrame in long format.
    
    Parameters:
    -----------
    samples : dict
        Dictionary with group names as keys and lists of measurements as values
    groups : list
        List of groups to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in long format with columns 'subject', 'group', 'value'
    """
    data = []
    for group in groups:
        if group in samples:
            values = samples[group]
            for i, value in enumerate(values):
                data.append({
                    'subject': i,  # Use a unique identifier here
                    'group': group,
                    'value': value
                })
    return pd.DataFrame(data)

def no_transform(df, dv):
    """No transformation - returns the DataFrame unchanged."""
    return df

def log_transform(df, dv):
    """Logarithmic transformation of the dependent variable."""
    df2 = df.copy()
    # Min + 1 to avoid negative or zero values
    min_val = df2[dv].min()
    offset = abs(min_val) + 1 if min_val <= 0 else 0
    df2[dv] = np.log(df2[dv] + offset)
    return df2

def boxcox_transform(df, dv):
    """Box-Cox transformation of the dependent variable."""
    from scipy import stats
    df2 = df.copy()
    # For Box-Cox, all values must be positive
    min_val = df2[dv].min()
    offset = abs(min_val) + 1 if min_val <= 0 else 0
    
    # Perform Box-Cox transformation
    try:
        transformed_data, lambda_val = stats.boxcox(df2[dv] + offset)
        df2[dv] = transformed_data
        print(f"Box-Cox transformation performed with Lambda={lambda_val:.4f}")
    except Exception as e:
        print(f"Box-Cox transformation failed: {str(e)}")
        # Fallback to log transformation
        df2[dv] = np.log(df2[dv] + offset)
        print("Fallback: Logarithmic transformation was used instead")
    
    return df2

class GroupSelectionDialog(QDialog):
    """Dialog for selecting groups for a plot"""
    def __init__(self, available_groups, parent=None):
        if not available_groups:
            QMessageBox.critical(parent, "Error", "No groups available! Dialog will not open.")
            raise ValueError("No groups passed to GroupSelectionDialog.")
        super().__init__(parent)
        self.setWindowTitle("Select Groups")
        self.resize(300, 400)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoGroupSelection")
        
        # Explanation
        label = QLabel("Select the groups to be displayed in the plot:")
        label.setObjectName("lblGroupSelectionHelp")
        layout.addWidget(label)
        
        # Checkboxes for each group
        self.group_checks = {}
        group_container = QWidget()
        group_container.setObjectName("widGroupCheckboxes")
        group_layout = QVBoxLayout(group_container)
        group_layout.setObjectName("lyoGroupCheckboxes")
        
        for group in available_groups:
            check = QCheckBox(str(group))
            check.setObjectName(f"chkGroup_{str(group).replace(' ', '_')}")
            self.group_checks[group] = check
            group_layout.addWidget(check)
        
        layout.addWidget(group_container)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_groups(self):
        selected = [group for group, check in self.group_checks.items() if check.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one group!")
        return selected


class ColumnSelectionDialog(QDialog):
    """Dialog for selecting measurement columns for a dataset"""
    def __init__(self, available_columns, parent=None):
        if not available_columns:
            QMessageBox.critical(parent, "Error", "No measurement columns available! Dialog will not open.")
            raise ValueError("No measurement columns passed to ColumnSelectionDialog.")
        super().__init__(parent)
        self.setWindowTitle("Select Measurement Columns")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoColumnSelection")
        
        # Explanation
        label = QLabel("Select the columns to be used for analysis:")
        layout.addWidget(label)
        
        # NEW OPTION: Multi-dataset analysis
        self.multi_dataset_check = QCheckBox("Separate analysis per dataset with shared Excel file")
        self.multi_dataset_check.setToolTip("Analyzes each dataset separately, but combines all results in a shared Excel file")
        layout.addWidget(self.multi_dataset_check)
        
        # Checkboxes for each column
        scroll_area = QScrollArea()
        scroll_area.setObjectName("scrollColumns")
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setObjectName("widColumnContainer")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setObjectName("lyoColumnCheckboxes")
        
        self.column_checks = {}
        for column in available_columns:
            check = QCheckBox(str(column))
            check.setObjectName(f"chkColumn_{str(column).replace(' ', '_')}")
            self.column_checks[column] = check
            scroll_layout.addWidget(check)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_columns(self):
        selected = [column for column, check in self.column_checks.items() if check.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one measurement column!")
        return {
            "columns": selected,
            "multi_dataset": self.multi_dataset_check.isChecked(),
            "combine": False
        }


class PairwiseComparisonDialog(QDialog):
    """Dialog for selecting groups for pairwise comparisons"""
    def __init__(self, available_groups, parent=None):
        if not available_groups or len(available_groups) < 2:
            QMessageBox.critical(parent, "Error", "At least 2 groups are required for pairwise comparisons!")
            raise ValueError("Too few groups passed to PairwiseComparisonDialog.")
        super().__init__(parent)
        self.setWindowTitle("Pairwise Comparisons")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoPairwiseComparison")
        
        # Explanation
        label = QLabel("Select two groups between which a significance line should be displayed:")
        label.setObjectName("lblComparisonHelp")
        layout.addWidget(label)
        
        # Selection for group 1
        group1_layout = QHBoxLayout()
        group1_layout.setObjectName("lyoGroup1Selection")
        group1_label = QLabel("Group 1:")
        group1_label.setObjectName("lblGroup1")
        group1_layout.addWidget(group1_label)
        self.group1_combo = QComboBox()
        self.group1_combo.setObjectName("cboGroup1")
        self.group1_combo.addItems([str(g) for g in available_groups])
        group1_layout.addWidget(self.group1_combo)
        layout.addLayout(group1_layout)
        
        # Selection for group 2
        group2_layout = QHBoxLayout()
        group2_layout.setObjectName("lyoGroup2Selection")
        group2_label = QLabel("Group 2:")
        group2_label.setObjectName("lblGroup2")
        group2_layout.addWidget(group2_label)
        self.group2_combo = QComboBox()
        self.group2_combo.setObjectName("cboGroup2")
        self.group2_combo.addItems([str(g) for g in available_groups])
        if len(available_groups) > 1:
            self.group2_combo.setCurrentIndex(1)
        group2_layout.addWidget(self.group2_combo)
        layout.addLayout(group2_layout)
        
        # Hint text for explanation
        hint_label = QLabel("Note: Significance is automatically taken from the post-hoc tests.")
        hint_label.setObjectName("lblSignificanceHint")
        hint_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(hint_label)
        
        # Dependent samples with better description
        dependent_layout = QHBoxLayout()
        dependent_layout.setObjectName("lyoDependentOption")
        
        self.dependent_check = QCheckBox("Dependent samples (paired test)")
        self.dependent_check.setObjectName("chkDependentSamples")
        dependent_layout.addWidget(self.dependent_check)
        
        # Info button
        dependent_info = QPushButton("?")
        dependent_info.setObjectName("btnPairwiseDependentInfo")
        dependent_info.setMaximumWidth(20)
        dependent_info.clicked.connect(self.show_dependent_info)
        dependent_layout.addWidget(dependent_info)
        dependent_layout.addStretch()
        
        layout.addLayout(dependent_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_comparison(self):
        g1 = self.group1_combo.currentText()
        g2 = self.group2_combo.currentText()
        if g1 == g2:
            QMessageBox.warning(self, "Warning", "The two groups must be different!")
            return None
        return {
            'group1': g1,
            'group2': g2,
            'dependent': self.dependent_check.isChecked()
        }
    def show_dependent_info(self):
        QMessageBox.information(
            self, "Dependent samples for pairwise comparisons",
            "Select this option if the groups to be compared are dependent samples "
            "(e.g. measurements on the same subject at different time points).\n\n"
            "For dependent samples, a paired t-test (parametric) or "
            "a Wilcoxon signed-rank test (non-parametric) is performed.\n\n"
            "Note: The groups must have the same number of measurements and "
            "the order of measurements must match."
        )
        
class TwoWayAnovaDialog(QDialog):
    """Dialog for configuring a Two-Way ANOVA"""
    def __init__(self, groups, parent=None):
        if not groups or len(groups) < 2:
            QMessageBox.critical(parent, "Error", "At least 2 groups are required for a Two-Way ANOVA!")
            raise ValueError("Too few groups passed to TwoWayAnovaDialog.")
        super().__init__(parent)
        self.setWindowTitle("Configure Two-Way ANOVA")
        self.resize(500, 400)
        self.groups = groups
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoTwoWayAnova")
        
        # Explanation
        label = QLabel("Define additional factors for the Two-Way ANOVA:")
        label.setObjectName("lblTwoWayAnovaHelp")
        layout.addWidget(label)
        
        # Factor definition
        factor_group = QGroupBox("Factor definition")
        factor_group.setObjectName("grpFactorDefinition")
        factor_layout = QGridLayout(factor_group)
        factor_layout.setObjectName("lyoFactorDefinition")
        
        # Factor name
        factor_label = QLabel("Factor name:")
        factor_label.setObjectName("lblFactorName")
        factor_layout.addWidget(factor_label, 0, 0)
        
        self.factor_name = QLineEdit()
        self.factor_name.setObjectName("edtFactorName")
        self.factor_name.setPlaceholderText("e.g. Treatment, Gender, etc.")
        factor_layout.addWidget(self.factor_name, 0, 1)
        
        layout.addWidget(factor_group)
        
        # Factor values per group
        value_group = QGroupBox("Factor values per group")
        value_group.setObjectName("grpFactorValues")
        value_layout = QGridLayout(value_group)
        value_layout.setObjectName("lyoFactorValues")
        
        # Header
        group_header = QLabel("Group")
        group_header.setObjectName("lblGroupHeader")
        value_layout.addWidget(group_header, 0, 0)
        
        factor_header = QLabel("Factor value")
        factor_header.setObjectName("lblFactorHeader")
        value_layout.addWidget(factor_header, 0, 1)
        
        # Input fields for each group
        self.factor_values = {}
        for i, group in enumerate(groups):
            group_label = QLabel(str(group))
            group_label.setObjectName(f"lblGroup_{str(group).replace(' ', '_')}")
            value_layout.addWidget(group_label, i+1, 0)
            
            value_field = QLineEdit()
            value_field.setObjectName(f"edtFactorValue_{str(group).replace(' ', '_')}")
            value_field.setPlaceholderText("Value for this group")
            self.factor_values[group] = value_field
            value_layout.addWidget(value_field, i+1, 1)
        
        layout.addWidget(value_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_factor_data(self):
        factor_name = self.factor_name.text()
        if not factor_name:
            QMessageBox.warning(self, "Warning", "Please specify a factor name!")
            return None
        
        factor_data = {}
        for group, field in self.factor_values.items():
            value = field.text()
            if value:  # Only add values if a value was entered
                try:
                    # Try to convert the value to a number if possible
                    numeric_value = float(value)
                    if numeric_value.is_integer():
                        numeric_value = int(numeric_value)
                    factor_data[group] = {factor_name: numeric_value}
                except ValueError:
                    # If not a number, use the string value
                    factor_data[group] = {factor_name: value}
        if not factor_data:
            return None            
        
        return factor_data

class AdvancedTestDialog(QDialog):
    def __init__(self, parent=None, df=None, default_test=None):
        super().__init__(parent)
        self.df = df  # Store original DataFrame
        self.default_test = default_test
        self.setWindowTitle("Advanced Statistical Tests")
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        
        # 1. Test type selection
        self.testTypeGroup = QGroupBox("Select test")
        testTypeLayout = QVBoxLayout()
        self.mixedAnovaBtn = QRadioButton("Mixed ANOVA (Between + Within)")
        self.rmAnovaBtn = QRadioButton("Repeated Measures ANOVA (Within only)")
        self.twoWayAnovaBtn = QRadioButton("Two-Way ANOVA (Between only)")
        testTypeLayout.addWidget(self.mixedAnovaBtn)
        testTypeLayout.addWidget(self.rmAnovaBtn)
        testTypeLayout.addWidget(self.twoWayAnovaBtn)
        self.testTypeGroup.setLayout(testTypeLayout)
        
        # Preselection based on detected data format
        if self.df is not None:
            columns = list(self.df.columns)
            if 'Subject' in columns or 'subject' in columns:
                self.mixedAnovaBtn.setChecked(True)
            elif any('Factor' in col for col in columns):
                self.twoWayAnovaBtn.setChecked(True)
            else:
                self.mixedAnovaBtn.setChecked(True)  # Default case
            
        # 2. Column selection
        self.columnsGroup = QGroupBox("Assign variables")
        columnsLayout = QFormLayout()
        
        # Dependent variable
        self.dvCombo = QComboBox()
        if self.df is not None:
            self.dvCombo.addItems(self.df.columns)
            # Preselect value column
            value_idx = -1
            for i, col in enumerate(self.df.columns):
                if col.lower() == 'value' or 'wert' in col.lower():
                    value_idx = i
            if value_idx >= 0:
                self.dvCombo.setCurrentIndex(value_idx)
                    
        columnsLayout.addRow("Dependent variable:", self.dvCombo)
        
        # Subject ID
        self.subjectCombo = QComboBox()
        if self.df is not None:
            self.subjectCombo.addItems(self.df.columns)
            # Preselect subject column
            subject_idx = -1
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('subject', 'subjekt', 's01'):
                    subject_idx = i
            if subject_idx >= 0:
                self.subjectCombo.setCurrentIndex(subject_idx)
                
        columnsLayout.addRow("Subject/ID variable:", self.subjectCombo)
        
        # Within factors (multiple possible)
        self.withinList = QListWidget()
        if self.df is not None:
            self.withinList.addItems(self.df.columns)
            # Preselect within factors
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('timepoint', 'time', 'zeit', 'messzeitpunkt'):
                    self.withinList.item(i).setSelected(True)
                    
        self.withinList.setSelectionMode(QAbstractItemView.MultiSelection)
        columnsLayout.addRow("Within factors:", self.withinList)
        
        # Between factors (multiple possible)
        self.betweenList = QListWidget()
        if self.df is not None:
            self.betweenList.addItems(self.df.columns)
            # Preselect between factors
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('betweengrp', 'gruppe', 'group', 'factora', 'factorb'):
                    self.betweenList.item(i).setSelected(True)
                    
        self.betweenList.setSelectionMode(QAbstractItemView.MultiSelection)
        columnsLayout.addRow("Between factors:", self.betweenList)
        
        self.columnsGroup.setLayout(columnsLayout)
        
        # Connect methods for test type change
        self.mixedAnovaBtn.toggled.connect(self.update_field_visibility)
        self.rmAnovaBtn.toggled.connect(self.update_field_visibility)
        self.twoWayAnovaBtn.toggled.connect(self.update_field_visibility)
        
        # Immediately update fields based on initial test
        self.update_field_visibility()
        
        # 3. Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        # Always set a default button
        self.mixedAnovaBtn.setChecked(True)
        
        # Preselect based on detected format
        if self.default_test == "mixed_anova":
            self.mixedAnovaBtn.setChecked(True)
        elif self.default_test == "two_way_anova":
            self.twoWayAnovaBtn.setChecked(True)
        elif self.default_test == "repeated_measures_anova":
            self.rmAnovaBtn.setChecked(True)
        
        # Assemble layout
        layout.addWidget(self.testTypeGroup)
        layout.addWidget(self.columnsGroup)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def update_field_visibility(self):
        """Updates the visibility of fields depending on the selected test type"""
        is_two_way = self.twoWayAnovaBtn.isChecked()
        is_mixed = self.mixedAnovaBtn.isChecked()
        is_repeated = self.rmAnovaBtn.isChecked()
        
        # For Two-Way ANOVA we don't need Subject
        self.subjectCombo.setEnabled(not is_two_way)
        
        # Within factors only for Mixed ANOVA or Repeated Measures
        self.withinList.setEnabled(is_mixed or is_repeated)
        
        # Between factors only for Mixed ANOVA or Two-Way ANOVA
        self.betweenList.setEnabled(is_mixed or is_two_way)    
        
class PlotConfigDialog(QDialog):
    def __init__(self, groups, parent=None):
        if not groups:
            QMessageBox.critical(parent, "Error", "No groups passed for plot configuration!")
            raise ValueError("No groups for PlotConfigDialog.")
        if len(set(groups)) != len(groups):
            QMessageBox.warning(parent, "Warning", "There are duplicate group names!")
            # Optional: raise ValueError("Duplicate groups for PlotConfigDialog.")
        super().__init__(parent)
        self.setWindowTitle("Configure Plot")
        self.groups = groups
        
        # IMPORTANT: Explicitly initialize all status variables
        self.comparisons = []  # Empty list as initial state
        self.colors = None     # No preselected colors
        self.hatches = None    # No preselected hatches
        
        # Initialize UI elements
        layout = QVBoxLayout(self)
        
        # Comparisons list with clear button
        comp_group = QGroupBox("Pairwise Comparisons")
        comp_layout = QVBoxLayout()
        
        self.comp_list = QListWidget()
        comp_layout.addWidget(self.comp_list)
        
        # IMPORTANT: Add button for explicit reset
        reset_button = QPushButton("Reset comparisons")
        reset_button.clicked.connect(self.reset_comparisons)
        comp_layout.addWidget(reset_button)
        
        # Plot title and axis labels
        title_group = QGroupBox("Plot Labels")
        title_group.setObjectName("grpPlotLabels")
        title_layout = QGridLayout(title_group)
        title_layout.setObjectName("lyoPlotLabels")

        # Plot title
        title_label = QLabel("Plot Title:")
        title_label.setObjectName("lblPlotTitle")
        title_layout.addWidget(title_label, 0, 0)
        self.title_edit = QLineEdit("")
        self.title_edit.setObjectName("edtPlotTitle")
        self.title_edit.setPlaceholderText("Optional")
        title_layout.addWidget(self.title_edit, 0, 1)

        # Add file name
        file_label = QLabel("File Name:")
        file_label.setObjectName("lblFileName")
        title_layout.addWidget(file_label, 1, 0)
        self.file_name_edit = QLineEdit("")
        self.file_name_edit.setObjectName("edtFileName")
        self.file_name_edit.setPlaceholderText("Default: automatically generated from group names")
        title_layout.addWidget(self.file_name_edit, 1, 1)

        # X-axis
        x_label = QLabel("X-Axis Label:")
        x_label.setObjectName("lblXAxis")
        title_layout.addWidget(x_label, 2, 0)  # Changed here: 2, 0 instead of 1, 0
        self.x_label_edit = QLineEdit("")
        self.x_label_edit.setObjectName("edtXAxis")
        self.x_label_edit.setPlaceholderText("Optional")
        title_layout.addWidget(self.x_label_edit, 2, 1)  # Changed here: 2, 1 instead of 1, 1

        # Y-axis
        y_label = QLabel("Y-Axis Label:")
        y_label.setObjectName("lblYAxis")
        title_layout.addWidget(y_label, 3, 0)  # Changed here: 3, 0 instead of 2, 0
        self.y_label_edit = QLineEdit("")
        self.y_label_edit.setObjectName("edtYAxis")
        self.y_label_edit.setPlaceholderText("Optional")
        title_layout.addWidget(self.y_label_edit, 3, 1)  # Changed here: 3, 1 instead of 2, 1
        
        layout.addWidget(title_group)
        
        # Plot dimensions
        dim_group = QGroupBox("Plot Dimensions")
        dim_group.setObjectName("grpPlotDimensions")
        dim_layout = QHBoxLayout(dim_group)
        dim_layout.setObjectName("lyoPlotDimensions")
        
        width_label = QLabel("Width:")
        width_label.setObjectName("lblWidth")
        dim_layout.addWidget(width_label)
        self.width_spin = QSpinBox()
        self.width_spin.setObjectName("spnWidth")
        self.width_spin.setRange(4, 20)
        self.width_spin.setValue(12)
        dim_layout.addWidget(self.width_spin)
        
        height_label = QLabel("Height:")
        height_label.setObjectName("lblHeight")
        dim_layout.addWidget(height_label)
        self.height_spin = QSpinBox()
        self.height_spin.setObjectName("spnHeight")
        self.height_spin.setRange(4, 20)
        self.height_spin.setValue(10)
        dim_layout.addWidget(self.height_spin)
        
        layout.addWidget(dim_group)
        
        # Color and hatch settings
        appearance_group = QGroupBox("Column Design")
        appearance_group.setObjectName("grpColumnAppearance")
        appearance_layout = QGridLayout(appearance_group)
        appearance_layout.setObjectName("lyoColumnAppearance")
        
        # Header
        grp_header = QLabel("Group")
        grp_header.setObjectName("lblGroupHeader")
        appearance_layout.addWidget(grp_header, 0, 0)
        color_header = QLabel("Color")
        color_header.setObjectName("lblColorHeader")
        appearance_layout.addWidget(color_header, 0, 1)
        hatch_header = QLabel("Hatch")
        hatch_header.setObjectName("lblHatchHeader")
        appearance_layout.addWidget(hatch_header, 0, 2)
        
        self.color_buttons = {}
        self.hatch_combos = {}
        
        for i, group in enumerate(groups):
            # Group name
            group_label = QLabel(str(group))
            group_label.setObjectName(f"lblGroup_{str(group).replace(' ', '_')}")
            appearance_layout.addWidget(group_label, i+1, 0)
            
            # Color selection
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            color_button = QPushButton()
            color_button.setObjectName(f"btnColor_{str(group).replace(' ', '_')}")
            color_button.setStyleSheet(f"background-color: {color};")
            color_button.setFixedSize(30, 30)
            color_button.clicked.connect(lambda checked, g=group: self.select_color(g))
            self.color_buttons[group] = color_button
            appearance_layout.addWidget(color_button, i+1, 1)
            
            # Hatch selection
            hatch_combo = QComboBox()
            hatch_combo.setObjectName(f"cboHatch_{str(group).replace(' ', '_')}")
            hatch_combo.addItems(DEFAULT_HATCHES)
            hatch_combo.setCurrentText(DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)])
            self.hatch_combos[group] = hatch_combo
            appearance_layout.addWidget(hatch_combo, i+1, 2)
        
        layout.addWidget(appearance_group)
        
        # Statistical options
        stats_group = QGroupBox("Statistical Options")
        stats_group.setObjectName("grpStatOptions")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setObjectName("lyoStatOptions")
        
        # Improved description for dependent samples
        dependent_layout = QHBoxLayout()
        self.dependent_check = QCheckBox("Dependent samples (paired tests)")
        self.dependent_check.setObjectName("chkDependentSamples")
        dependent_layout.addWidget(self.dependent_check)
        
        # Add an info button with tooltip
        dependent_info = QPushButton("?")
        dependent_info.setObjectName("btnDependentInfo")
        dependent_info.setMaximumWidth(20)
        dependent_info.setToolTip(
            "Dependent samples are measurements on the same subject at different time points or "
            "under different conditions. Examples: pre/post measurements, repeated measurements.\n\n"
            "Important: For dependent tests, all groups must have the same number of measurements and "
            "the order of the values must match."
        )
        dependent_info.clicked.connect(self.show_dependent_help)
        dependent_layout.addWidget(dependent_info)
        dependent_layout.addStretch()
        
        stats_layout.addLayout(dependent_layout)
        
        # Options for dependent visualization
        self.dependent_visualization_options = QWidget()
        dependent_vis_layout = QVBoxLayout(self.dependent_visualization_options)
        dependent_vis_layout.setObjectName("lyoDependentVisOptions")
        
        self.show_individual_lines = QCheckBox("Show individual connection lines")
        self.show_individual_lines.setObjectName("chkShowIndividualLines")
        self.show_individual_lines.setChecked(True)
        self.show_individual_lines.setToolTip(
            "If enabled, lines are drawn connecting the values of the same subject. "
            "This is useful to visualize individual changes."
        )
        
        dependent_vis_layout.addWidget(self.show_individual_lines)
        # Hide initially
        self.dependent_visualization_options.setVisible(False)
        
        # Insert below the dependent-check
        stats_layout.addWidget(self.dependent_visualization_options)

        # -- Options for dependent tests: select subject ID and within factor
        self.dependent_data_options = QWidget()
        dep_data_layout = QFormLayout(self.dependent_data_options)

        # Placeholder text for missing column data
        self.subject_info_label = QLabel(
            "Note: After closing this dialog, subject ID and within factor will be required. "
            "You can select these later in the advanced test options or in the data import dialog."
        )
        self.subject_info_label.setWordWrap(True)
        self.subject_info_label.setStyleSheet("color: #666; font-style: italic;")
        dep_data_layout.addRow(self.subject_info_label)

        # Hide initially
        self.dependent_data_options.setVisible(False)
        stats_layout.addWidget(self.dependent_data_options)
        
        # Error bar type
        error_bar_layout = QHBoxLayout()
        error_bar_layout.setObjectName("lyoErrorBarType")
        error_bar_label = QLabel("Error bar type:")
        error_bar_label.setObjectName("lblErrorBarType")
        error_bar_layout.addWidget(error_bar_label)

        self.error_type_sd = QRadioButton("Standard deviation (SD)")
        self.error_type_sd.setObjectName("radErrorTypeSD")
        self.error_type_sd.setChecked(True)
        self.error_type_sem = QRadioButton("Standard error (SEM)")
        self.error_type_sem.setObjectName("radErrorTypeSEM")
        
        error_bar_layout.addWidget(self.error_type_sd)
        error_bar_layout.addWidget(self.error_type_sem)
        stats_layout.addLayout(error_bar_layout)
        
        # Options to create or skip plots
        self.create_plot_check = QCheckBox("Create plot")
        self.create_plot_check.setObjectName("chkCreatePlot")
        self.create_plot_check.setChecked(False)
        stats_layout.addWidget(self.create_plot_check)
        
        # List of comparisons
        comparisons_label = QLabel("Specific comparisons:")
        comparisons_label.setObjectName("lblComparisons")
        stats_layout.addWidget(comparisons_label)
        
        self.comparisons_list = QListWidget()
        self.comparisons_list.setObjectName("lstComparisons")
        stats_layout.addWidget(self.comparisons_list)
        
        # Buttons for comparisons
        compare_buttons = QHBoxLayout()
        compare_buttons.setObjectName("lyoCompareButtons")
        
        add_compare_button = QPushButton("Add comparison")
        add_compare_button.setObjectName("btnAddComparison")
        add_compare_button.clicked.connect(self.add_comparison)
        
        remove_compare_button = QPushButton("Remove comparison")
        remove_compare_button.setObjectName("btnRemoveComparison")
        remove_compare_button.clicked.connect(self.remove_comparison)
        
        compare_buttons.addWidget(add_compare_button)
        compare_buttons.addWidget(remove_compare_button)
        stats_layout.addLayout(compare_buttons)
        
        layout.addWidget(stats_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Save comparisons
        self.comparisons = []
        
    def toggle_dependent_options(self, checked):
        """Show or hide dependent options (visualization + data selection)."""
        self.dependent_visualization_options.setVisible(checked)
        self.dependent_data_options.setVisible(checked)
            
    def show_dependent_help(self):
        QMessageBox.information(
            self, "Dependent samples",
            "<b>When to choose dependent samples?</b><br><br>"
            "Dependent (or paired) samples are present when you:<br>"
            "• Measure the same subjects/individuals multiple times (e.g. before/after a treatment)<br>"
            "• Perform measurements on naturally paired pairs (e.g. twin studies)<br>"
            "• Take multiple measurements on the same material under different conditions<br><br>"
            "<b>Requirements for dependent tests:</b><br>"
            "• All groups must have the <i>same number</i> of measurements<br>"
            "• The values must be in the <i>same order</i> (i.e. measurement 1 from group A "
            "belongs to the same subject as measurement 1 from group B)<br>"
            "• For more than 2 groups, a special repeated measures ANOVA is performed<br><br>"
            "<b>Visualization:</b><br>"
            "For dependent samples, an additional line plot is created "
            "showing the individual changes."
        )
        
    def set_title(self, title): self.title_edit.setText(title)
    def set_x_label(self, x): self.x_label_edit.setText(x)
    def set_y_label(self, y): self.y_label_edit.setText(y)
    def set_groups(self, groups): self.groups = groups
    def set_width(self, w): self.width_spin.setValue(w)
    def set_height(self, h): self.height_spin.setValue(h)
    def set_color(self, group, color):
        if group in self.color_buttons:
            self.color_buttons[group].setStyleSheet(f"background-color: {color};")
        else:
            QMessageBox.warning(self, "Warning", f"Group {group} not in color table!")
    def set_hatch(self, group, hatch):
        if group in self.hatch_combos:
            idx = self.hatch_combos[group].findText(hatch)
            if idx >= 0:
                self.hatch_combos[group].setCurrentIndex(idx)
            else:
                QMessageBox.warning(self, "Warning", f"Hatch '{hatch}' not available for group {group}!")
        else:
            QMessageBox.warning(self, "Warning", f"Group {group} not in hatch table!")
    def set_dependent(self, dep): self.dependent_check.setChecked(dep)
    def set_create_plot(self, create): self.create_plot_check.setChecked(create)
    def set_comparisons(self, comps):
        self.comparisons = comps
        self.comparisons_list.clear()
        for comp in comps:
            self.comparisons_list.addItem(f"{comp['group1']} vs {comp['group2']} ({comp['test_type']})")
    def set_file_name(self, name): self.file_name_edit.setText(name)

    def select_color(self, group):
        """Opens a color dialog for a group"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_buttons[group].setStyleSheet(f"background-color: {color.name()};")

    def add_comparison(self):
        dialog = PairwiseComparisonDialog(self.groups, self)
        if dialog.exec_() == QDialog.Accepted:
            comp = dialog.get_comparison()
            if comp['group1'] == comp['group2']:
                QMessageBox.warning(self, "Error", "The two groups must be different.")
                return
                    # Set a default value for the test type, will be replaced later by post-hoc results
            comp['test_type'] = "From post-hoc test"

            self.comparisons.append(comp)
            self.comparisons_list.addItem(f"{comp['group1']} vs {comp['group2']} ({comp['test_type']})")
    def remove_comparison(self):
        current_row = self.comparisons_list.currentRow()
        if current_row >= 0:
            self.comparisons_list.takeItem(current_row)
            self.comparisons.pop(current_row)
              
    def get_config(self):
        if not self.groups:
            QMessageBox.critical(self, "Error", "Plot without groups is not possible!")
            return None
        try:
            colors_dict = {}
            for g in self.groups:
                if g in self.color_buttons:
                    try:
                        # Try to extract color from stylesheet (safer)
                        style = self.color_buttons[g].styleSheet()
                        if "background-color:" in style:
                            color = style.split("background-color:")[1].split(";")[0].strip()
                            colors_dict[g] = color
                        else:
                            colors_dict[g] = DEFAULT_COLORS[0]  # Fallback
                    except Exception:
                        colors_dict[g] = DEFAULT_COLORS[0]  # Second fallback
                        
            hatches_dict = {}
            for g in self.groups:
                if g in self.hatch_combos:
                    try:
                        hatches_dict[g] = self.hatch_combos[g].currentText()
                    except Exception:
                        hatches_dict[g] = ""  # Fallback
                    
            config = {
                'title': self.title_edit.text() if self.title_edit.text() else None,
                'x_label': self.x_label_edit.text() if self.x_label_edit.text() else None,
                'y_label': self.y_label_edit.text() if self.y_label_edit.text() else None,
                'file_name': self.file_name_edit.text() if self.file_name_edit.text() else None,
                'groups': self.groups,
                'width': self.width_spin.value(),
                'height': self.height_spin.value(),
                'colors': colors_dict,
                'hatches': hatches_dict,
                'create_plot': self.create_plot_check.isChecked(),
                'comparisons': self.comparisons.copy() if self.comparisons else [],
                'error_type': 'se' if self.error_type_sem.isChecked() else 'sd',
                'dependent': self.dependent_check.isChecked(),
                'show_individual_lines': self.show_individual_lines.isChecked() if self.dependent_check.isChecked() else False,
                'needs_subject_selection': self.dependent_check.isChecked()
            }
            return config
        
        except Exception as e:
            import traceback
            print(f"Error in get_config: {str(e)}")
            traceback.print_exc()
            # Return a minimal config to prevent crashes
            return {
                'groups': self.groups,
                'title': None,
                'x_label': None,
                'y_label': None,
                'width': 12,
                'height': 10,
                'colors': {g: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, g in enumerate(self.groups)},
                'hatches': {g: "" for g in self.groups},
                'dependent': False,
                'create_plot': True,
                'comparisons': [],
                'error_type': 'sd'
            }
    
    def reset_comparisons(self):
        """Explicit method to reset comparisons"""
        self.comparisons = []
        self.comparisons_list.clear()
        
class TransformationDialog(QDialog):
    def __init__(self, test_info=None, parent=None):
        super().__init__(parent)
        self.test_info = test_info
        self.transformation = None
        self.setWindowTitle("Select Data Transformation")
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # Info text
        info = QLabel("The data do not meet the requirements for parametric tests.\n"
                     "Please select a transformation:")
        layout.addWidget(info)
        
        # Transformation options
        self.log10RB = QRadioButton("Log10 transformation (for positive, right-skewed data)")
        self.boxcoxRB = QRadioButton("Box-Cox transformation (automatic lambda optimization)")
        self.arcsinRB = QRadioButton("Arcsin square root transformation (for percentages/proportions)")
        self.noTransformRB = QRadioButton("No transformation (use non-parametric test)")
        
        # Default selection
        self.log10RB.setChecked(True)
        
        # Add options to layout
        layout.addWidget(self.log10RB)
        layout.addWidget(self.boxcoxRB)
        layout.addWidget(self.arcsinRB)
        layout.addWidget(self.noTransformRB)
        
        # Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def get_transformation(self):
        if self.noTransformRB.isChecked():
            return None
        elif self.log10RB.isChecked():
            return "log10"
        elif self.boxcoxRB.isChecked():
            return "boxcox"
        elif self.arcsinRB.isChecked():
            return "arcsin_sqrt"
        return None

class OutlierDetectionDialog(QDialog):
    """Dialog for configuring outlier detection analysis"""
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Outlier Detection")
        self.resize(600, 500)  # Make it slightly wider
        self.df = df
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Check if outlier detection is available
        if not OUTLIER_IMPORTS_AVAILABLE:
            warning_label = QLabel("Warning: Outlier detection is not available.\n"
                                "Please install required packages: outliers, pingouin, openpyxl")
            warning_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(warning_label)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(self.reject)
            layout.addWidget(button_box)
            return
        
        # Info section
        info_label = QLabel("Select data columns and parameters for outlier detection:")
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)
        
        # Data selection section
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)
        
        # Group column selection
        self.group_col_combo = QComboBox()
        self.group_col_combo.addItems(self.df.columns)
        data_layout.addRow("Group Column:", self.group_col_combo)
        
        # Analysis mode selection
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.single_dataset_radio = QRadioButton("Single dataset analysis")
        self.multi_dataset_radio = QRadioButton("Multi-dataset analysis")
        self.single_dataset_radio.setChecked(True)
        
        mode_layout.addWidget(self.single_dataset_radio)
        mode_layout.addWidget(self.multi_dataset_radio)
        
        # Connect radio buttons to update UI
        self.single_dataset_radio.toggled.connect(self.update_dataset_selection)
        self.multi_dataset_radio.toggled.connect(self.update_dataset_selection)
        
        layout.addWidget(mode_group)
        
        # Single dataset selection
        self.single_dataset_group = QGroupBox("Single Dataset")
        single_layout = QFormLayout(self.single_dataset_group)
        
        self.value_col_combo = QComboBox()
        numeric_columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        self.value_col_combo.addItems(numeric_columns)
        single_layout.addRow("Value Column:", self.value_col_combo)
        
        layout.addWidget(self.single_dataset_group)
        
        # Multi-dataset selection
        self.multi_dataset_group = QGroupBox("Multiple Datasets")
        multi_layout = QVBoxLayout(self.multi_dataset_group)
        
        multi_info = QLabel("Select all value columns to analyze for outliers:")
        multi_layout.addWidget(multi_info)
        
        # Scrollable list for dataset selection
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.dataset_checkboxes = {}
        # REMOVED: CNRQ-specific filtering - now shows ALL numeric columns
        for col in numeric_columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(False)  # REMOVED: Auto-select based on CNRQ
            self.dataset_checkboxes[col] = checkbox
            scroll_layout.addWidget(checkbox)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(150)
        multi_layout.addWidget(scroll_area)
        
        # SIMPLIFIED: Select all/none buttons only (removed CNRQ button)
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        
        select_all_btn.clicked.connect(lambda: self.set_all_datasets(True))
        select_none_btn.clicked.connect(lambda: self.set_all_datasets(False))
        # REMOVED: select_cnrq_btn and its connection
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(select_none_btn)
        # REMOVED: button_layout.addWidget(select_cnrq_btn)
        multi_layout.addLayout(button_layout)
        
        layout.addWidget(self.multi_dataset_group)
        layout.addWidget(data_group)
        
        # Test parameters section (unchanged)
        params_group = QGroupBox("Test Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Test mode
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Test Mode:")
        mode_label.setStyleSheet("font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        self.single_mode = QRadioButton("Single-pass detection (detect one outlier per test)")
        self.iterative_mode = QRadioButton("Iterative detection (remove outliers until none found)")
        self.iterative_mode.setChecked(True)  # Default to iterative
        
        mode_layout.addWidget(self.single_mode)
        mode_layout.addWidget(self.iterative_mode)
        
        params_layout.addLayout(mode_layout)
        
        # Test types
        test_layout = QVBoxLayout()
        test_label = QLabel("Tests to Perform:")
        test_label.setStyleSheet("font-weight: bold;")
        test_layout.addWidget(test_label)

        self.modz_check = QCheckBox("Modified Z-Score Test (robust detection using median)")
        self.grubbs_check = QCheckBox("Grubbs' Test (for normally distributed data)")
        self.modz_check.setChecked(True)
        self.grubbs_check.setChecked(False)

        test_layout.addWidget(self.modz_check)
        test_layout.addWidget(self.grubbs_check)
        
        params_layout.addLayout(test_layout)
        
        layout.addWidget(params_group)
        
        # Output section
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        
        # File path selection
        file_layout = QHBoxLayout()
        file_label = QLabel("Output File:")
        self.file_path_label = QLabel("outlier_analysis_results.xlsx")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_file)
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(browse_button)
        
        output_layout.addLayout(file_layout)
        layout.addWidget(output_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initial UI update
        self.update_dataset_selection()
        
    def update_dataset_selection(self):
        """Update UI based on selected analysis mode"""
        is_single = self.single_dataset_radio.isChecked()
        self.single_dataset_group.setVisible(is_single)
        self.multi_dataset_group.setVisible(not is_single)
        
    def set_all_datasets(self, checked):
        """Select or deselect all dataset checkboxes"""
        for checkbox in self.dataset_checkboxes.values():
            checkbox.setChecked(checked)
    
    def select_cnrq_datasets(self):
        """Select only CNRQ columns"""
        for col, checkbox in self.dataset_checkboxes.items():
            checkbox.setChecked('CNRQ' in col.upper())
    
    def browse_output_file(self):
        """Browse for output file location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Outlier Analysis Results", 
            "outlier_analysis_results.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
            self.file_path_label.setText(file_path)
    
    def get_config(self):
        """Get the configuration from the dialog"""
        if not OUTLIER_IMPORTS_AVAILABLE:
            return None
            
        config = {
            'group_column': self.group_col_combo.currentText(),
            'iterate': self.iterative_mode.isChecked(),
            'run_modz': self.modz_check.isChecked(),
            'run_grubbs': self.grubbs_check.isChecked(),
            'output_file': self.file_path_label.text(),
            'is_multi_dataset': self.multi_dataset_radio.isChecked()
        }
        
        if config['is_multi_dataset']:
            # Get selected datasets
            selected_datasets = [col for col, checkbox in self.dataset_checkboxes.items() 
                            if checkbox.isChecked()]
            if not selected_datasets:
                QMessageBox.warning(self, "No Datasets Selected", 
                                "Please select at least one dataset column for analysis.")
                return None
            config['dataset_columns'] = selected_datasets
        else:
            config['value_column'] = self.value_col_combo.currentText()
            
        return config


class StatisticalAnalyzerApp(QMainWindow):
    """Main application for statistical analysis of data from Excel/CSV files."""
    
    def __init__(self):
        """Initializes the application with all UI elements."""
        super().__init__()
        self.setWindowTitle("Statistical Data Analyzer")
        self.setGeometry(100, 50, 1600, 1300)
        
        # Data attributes
        self.file_path = None
        self.df = None
        self.samples = None
        self.sheet_names = []
        self.available_groups = []
        self.numeric_columns = []
        self.plot_configs = []
        
        # Initialize UI elements
        self.init_ui()
        
        # Status for combined columns
        self.selected_columns = []
        self.combine_columns = False
        
        # Add menu bar
        self.create_menu()
               
    def create_menu(self):
        """Creates the menu bar with help options"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        dependent_help_action = QAction('Dependent Samples', self)
        dependent_help_action.triggered.connect(self.show_dependent_samples_help)
        help_menu.addAction(dependent_help_action)
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Advanced tests in help menu
        advanced_action = QAction('Advanced Tests', self)
        advanced_action.triggered.connect(self.show_advanced_tests_help)
        help_menu.addAction(advanced_action)

        # Analysis menu (create new or use existing)
        analysis_menu = menubar.addMenu('&Analysis')

        # Action for advanced tests
        advanced_test_action = QAction('Run Advanced Tests', self)
        advanced_test_action.triggered.connect(self.open_advanced_tests)
        analysis_menu.addAction(advanced_test_action)
        
        # Action for outlier detection
        outlier_action = QAction('Detect Outliers', self)
        outlier_action.triggered.connect(self.run_outlier_detection)
        analysis_menu.addAction(outlier_action)

        
    def show_about(self):
        QMessageBox.information(
            self,
            "About Statistical Data Analyzer",
            "<h2>Statistical Data Analyzer</h2>"
              "<p>Version 1.0</p>"
              "<p>An application for statistical analysis and visualization of data.</p>"
              "<p>© 2023</p>"
        )
        
    def init_ui(self):
        """Initializes all UI elements of the application."""
        # Main widget and layout
        central_widget = QWidget()
        central_widget.setObjectName("widMainContainer")
        main_layout = QVBoxLayout(central_widget)
        main_layout.setObjectName("lyoMainLayout")
        
        # File selection
        file_section = QGroupBox("Data Source")
        file_section.setObjectName("grpDataSource")
        file_layout = QHBoxLayout(file_section)
        file_layout.setObjectName("lyoFileSection")
        
        file_label = QLabel("Excel file:")
        file_label.setObjectName("lblFileLabel")
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setObjectName("lblFilePath")
        browse_button = QPushButton("Browse...")
        browse_button.setObjectName("btnBrowse")
        browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(browse_button)
        
        main_layout.addWidget(file_section)
        
        # Excel sheet and column selection
        data_section = QGroupBox("Data Configuration")
        data_section.setObjectName("grpDataConfig")
        data_layout = QVBoxLayout(data_section)
        data_layout.setObjectName("lyoDataSection")
        
        # Excel sheet
        sheet_layout = QHBoxLayout()
        sheet_layout.setObjectName("lyoSheetSection")
        sheet_label = QLabel("Worksheet:")
        sheet_label.setObjectName("lblSheetLabel")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setObjectName("cboWorksheet")
        self.sheet_combo.currentIndexChanged.connect(self.load_sheet)
        sheet_layout.addWidget(sheet_label)
        sheet_layout.addWidget(self.sheet_combo, 1)
        data_layout.addLayout(sheet_layout)
        
        # Group column
        group_col_layout = QHBoxLayout()
        group_col_layout.setObjectName("lyoGroupColSection")
        group_col_label = QLabel("Group column:")
        group_col_label.setObjectName("lblGroupColLabel")
        self.group_col_combo = QComboBox()
        self.group_col_combo.setObjectName("cboGroupColumn")
        self.group_col_combo.currentIndexChanged.connect(self.update_available_groups)
        group_col_layout.addWidget(group_col_label)
        group_col_layout.addWidget(self.group_col_combo, 1)
        data_layout.addLayout(group_col_layout)
        
        # Value column(s)
        value_cols_layout = QHBoxLayout()
        value_cols_layout.setObjectName("lyoValueColSection")
        value_cols_label = QLabel("Value column(s):")
        value_cols_label.setObjectName("lblValueColLabel")
        self.value_cols_combo = QComboBox()
        self.value_cols_combo.setObjectName("cboValueColumn")
        self.value_cols_combo.currentIndexChanged.connect(self.update_samples)
        value_cols_layout.addWidget(value_cols_label)
        value_cols_layout.addWidget(self.value_cols_combo, 1)
        
        # Button to select multiple columns
        self.select_columns_button = QPushButton("Multiple columns...")
        self.select_columns_button.setObjectName("btnSelectColumns")
        self.select_columns_button.clicked.connect(self.select_multiple_columns)
        value_cols_layout.addWidget(self.select_columns_button)
        
        data_layout.addLayout(value_cols_layout)
        
        # Mark for combined columns
        self.combine_columns_label = QLabel("No combined columns selected")
        self.combine_columns_label.setObjectName("lblCombineStatus")
        self.combine_columns_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.combine_columns_label)
        
        main_layout.addWidget(data_section)
        
        # Available groups and plot management
        groups_and_plots = QHBoxLayout()
        groups_and_plots.setObjectName("lyoGroupsAndPlots")
        
        # Available groups
        groups_section = QGroupBox("Available Groups")
        groups_section.setObjectName("grpAvailableGroups")
        groups_layout = QVBoxLayout(groups_section)
        groups_layout.setObjectName("lyoGroupsSection")
        
        self.groups_list = QListWidget()
        self.groups_list.setObjectName("lstAvailableGroups")
        groups_layout.addWidget(self.groups_list)
        
        # Buttons for group selection
        group_buttons = QHBoxLayout()
        group_buttons.setObjectName("lyoGroupButtons")
        select_groups_button = QPushButton("Select groups for plot")
        select_groups_button.setObjectName("btnSelectGroups")
        select_groups_button.clicked.connect(self.select_groups_for_plot)
        
        group_buttons.addWidget(select_groups_button)
        groups_layout.addLayout(group_buttons)
        
        groups_and_plots.addWidget(groups_section)
        
        # Plot configurations
        plots_section = QGroupBox("Plot Configurations")
        plots_section.setObjectName("grpPlotConfigs")
        plots_layout = QVBoxLayout(plots_section)
        plots_layout.setObjectName("lyoPlotsSection")
        
        self.plots_list = QListWidget()
        self.plots_list.setObjectName("lstPlotConfigurations")
        self.plots_list.itemDoubleClicked.connect(self.edit_plot_config)
        plots_layout.addWidget(self.plots_list)
        
        # Plot buttons
        plot_buttons = QHBoxLayout()
        plot_buttons.setObjectName("lyoPlotButtons")
        remove_plot_button = QPushButton("Remove plot")
        remove_plot_button.setObjectName("btnRemovePlot")
        remove_plot_button.clicked.connect(self.remove_plot)
        preview_plot_button = QPushButton("Plot preview")
        preview_plot_button.setObjectName("btnPreviewPlot")
        preview_plot_button.clicked.connect(self.preview_selected_plot)
        
        plot_buttons.addWidget(remove_plot_button)
        plot_buttons.addWidget(preview_plot_button)
        plots_layout.addLayout(plot_buttons)
        
        groups_and_plots.addWidget(plots_section)
        main_layout.addLayout(groups_and_plots)
        
        # Plot preview
        preview_section = QGroupBox("Plot Preview")
        preview_section.setObjectName("grpPlotPreview")
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setObjectName("lyoPreviewSection")
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("canvasPlotPreview")
        preview_layout.addWidget(self.canvas)
        
        main_layout.addWidget(preview_section)

        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setObjectName("lyoActionButtons")
        analyze_button = QPushButton("Create and analyze all plots")
        analyze_button.setObjectName("btnAnalyzeAll")
        analyze_button.clicked.connect(self.run_all_analyses)
        analyze_selected_button = QPushButton("Create selected plot")
        analyze_selected_button.setObjectName("btnAnalyzeSelected")
        analyze_selected_button.clicked.connect(self.run_selected_analysis)
        multi_analyze_button = QPushButton("Start multi-dataset analysis")
        multi_analyze_button.setObjectName("btnMultiDatasetAnalyze")
        multi_analyze_button.clicked.connect(self.run_multi_dataset_analysis)
        outlier_button = QPushButton("Detect outliers")
        outlier_button.setObjectName("btnDetectOutliers")
        outlier_button.clicked.connect(self.run_outlier_detection)
        actions_layout.addWidget(analyze_button)
        actions_layout.addWidget(analyze_selected_button)
        actions_layout.addWidget(multi_analyze_button)
        actions_layout.addWidget(outlier_button)
        
        main_layout.addLayout(actions_layout)
        
        self.setCentralWidget(central_widget)
    
    def browse_file(self):
        """Opens a dialog to select an Excel or CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Excel file", "", 
            "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            # Check if file exists and has a supported format
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Error", f"The file {file_path} does not exist.")
                return
                
            if not file_path.endswith(('.xlsx', '.xls', '.csv')):
                QMessageBox.critical(self, "Error", "Only Excel and CSV files are supported.")
                return
                
            self.file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.load_file()
    
    def load_file(self):
        """Loads the selected file and prepares it for analysis."""
        if not self.file_path:
            return
            
        try:
            if self.file_path.endswith('.csv'):
                # CSV file
                self.df = pd.read_csv(self.file_path)
                self.sheet_combo.clear()
                self.sheet_combo.setEnabled(False)
            else:
                # Excel file
                excel = pd.ExcelFile(self.file_path)
                self.sheet_names = excel.sheet_names
                
                self.sheet_combo.clear()
                self.sheet_combo.addItems(self.sheet_names)
                self.sheet_combo.setEnabled(True)
                
                # Load first worksheet by default
                if self.sheet_names:
                    self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_names[0])
            
            # Reset selection
            self.selected_columns = []
            self.combine_columns = False
            self.combine_columns_label.setText("No combined columns selected")
            
            self.update_column_lists()
            
        except Exception as e:
            self.df = None
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_sheet(self, index):
        """Loads a specific worksheet from an Excel file."""
        if index < 0 or not self.file_path or not self.file_path.endswith(('.xlsx', '.xls')):
            return
            
        try:
            sheet_name = self.sheet_combo.itemText(index)
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            self.update_column_lists()
        except Exception as e:
            self.df = None
            QMessageBox.critical(self, "Error", f"Error loading worksheet: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_column_lists(self):
        """Updates the combo boxes for group and value columns."""
        if self.df is None:
            return
            
        try:
            # Group columns
            self.group_col_combo.clear()
            self.group_col_combo.addItems(self.df.columns)
            
            # Value columns (only numeric columns)
            self.value_cols_combo.clear()
            self.numeric_columns = []
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.numeric_columns.append(col)
            
            self.value_cols_combo.addItems(self.numeric_columns)
            
            # Update available groups after loading new columns
            self.update_available_groups()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error updating column lists: {str(e)}")
    
    def select_multiple_columns(self):
        """Opens a dialog to select multiple value columns."""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
            
        if not self.numeric_columns:
            QMessageBox.warning(self, "Warning", "No numeric columns available.")
            return
            
        dialog = ColumnSelectionDialog(self.numeric_columns, self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_selected_columns()
            selected_columns = result["columns"]
            multi_dataset = result["multi_dataset"]
            
            if not selected_columns:
                QMessageBox.warning(self, "Warning", "No columns selected.")
                return
                
            self.selected_columns = selected_columns
            self.combine_columns = False  # Always set to False
            self.multi_dataset_analysis = multi_dataset
            
            # Update display
            if len(selected_columns) == 1:
                # If only one column, select it in the combobox
                index = self.value_cols_combo.findText(selected_columns[0])
                if index >= 0:
                    self.value_cols_combo.setCurrentIndex(index)
                self.combine_columns_label.setText("No combined columns selected")
            else:
                # For multiple columns, show info
                if multi_dataset:
                    self.combine_columns_label.setText(f"Multi-dataset analysis: {', '.join(selected_columns)}")
                    # ... rest of code ...
                else:
                    self.combine_columns_label.setText(f"Selected: {', '.join(selected_columns)}")
            
            # Update available groups
            self.update_available_groups()
    
    def update_available_groups(self):
        """Updates the list of available groups based on the selected group column."""
        if self.df is None:
            return
            
        if not self.group_col_combo.currentText():
            return
            
        group_col = self.group_col_combo.currentText()
        
        try:
            # Extract groups from the column
            self.available_groups = sorted(self.df[group_col].unique())
            
            # Show list of available groups
            self.groups_list.clear()
            for group in self.available_groups:
                self.groups_list.addItem(str(group))
            
            # Reload data
            self.update_samples()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error loading groups: {str(e)}")
    
    def update_samples(self):
        """Updates the samples based on the current selection of group column and value columns."""
        if self.df is None:
            # No data loaded
            self.samples = None
            return
            
        if not self.group_col_combo.currentText():
            # No group column selected
            QMessageBox.warning(self, "Warning", "Please select a group column.")
            self.samples = None
            return
            
        group_col = self.group_col_combo.currentText()
        
        try:
            # Determine columns to use
            if len(self.selected_columns) > 1:
                value_cols = self.selected_columns
                # Check if all columns actually exist
                for col in value_cols:
                    if col not in self.df.columns:
                        QMessageBox.warning(self, "Warning", 
                            f"The value column '{col}' was not found. Please select valid columns.")
                        self.samples = None
                        return
            else:
                # If no explicit selection, use current combobox selection
                current_value_col = self.value_cols_combo.currentText()
                if not current_value_col:
                    QMessageBox.warning(self, "Warning", "Please select at least one value column.")
                    self.samples = None
                    return
                    
                value_cols = [current_value_col]
            
            # Import data with the DataImporter class
            sheet_name = self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0
            self.samples, _ = DataImporter.import_data(
                self.file_path,
                sheet_name=sheet_name,
                group_col=group_col,
                value_cols=value_cols,
                combine_columns=self.combine_columns
            )
            
            # New validation for possible dependent samples
            self.validate_dependent_samples_possibility()
            
        except Exception as e:
            self.samples = None
            QMessageBox.warning(self, "Warning", f"Error importing data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def validate_dependent_samples_possibility(self):
        """Checks if the loaded data might be suitable for dependent tests"""
        if not self.samples or len(self.samples) < 2:
            return
            
        # Check if all groups have the same number of measurements
        group_sizes = [len(values) for values in self.samples.values()]
        equal_sizes = len(set(group_sizes)) == 1
        
        if not equal_sizes:
            # Discreet hint message at the bottom of the screen or as status
            self.statusBar().showMessage(
                "Note: Groups have different sizes - dependent tests may be unsuitable", 
                10000  # Show for 10 seconds
            )        
    
    def open_advanced_tests(self):
        """Opens the dialog for advanced statistical tests"""
        import traceback  # Add this line
        
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        # Phase 1: Select test and parameters
        data_format = self.detect_data_format(self.df)
        dialog = AdvancedTestDialog(parent=self, df=self.df, default_test=data_format)
        
        if dialog.exec_() == QDialog.Accepted:
            # Read test parameters
            test_type = None
            if dialog.mixedAnovaBtn.isChecked():
                test_type = "mixed_anova"
            elif dialog.rmAnovaBtn.isChecked():
                test_type = "repeated_measures_anova"
            elif dialog.twoWayAnovaBtn.isChecked():
                test_type = "two_way_anova"
                
            # Validate inputs based on test type
            between = [item.text() for item in dialog.betweenList.selectedItems()]
            within = [item.text() for item in dialog.withinList.selectedItems()]
            
            if test_type == "mixed_anova" and (not between or not within):
                QMessageBox.warning(self, "Error", "Mixed ANOVA requires both between and within factors!")
                return
            elif test_type == "repeated_measures_anova" and not within:
                QMessageBox.warning(self, "Error", "Repeated Measures ANOVA requires at least one within factor!")
                return
            elif test_type == "two_way_anova" and len(between) < 2:
                QMessageBox.warning(self, "Error", "Two-Way ANOVA requires at least two between factors!")
                return
    
            # NEW: Dialog to select save path for Excel file
            excel_path, _ = QFileDialog.getSaveFileName(
                self, "Save Excel results file", "",
                "Excel files (*.xlsx);;All files (*.*)"
            )
            if not excel_path:  # User cancelled
                return
                
            # Add .xlsx at the end if not present
            if not excel_path.lower().endswith('.xlsx'):
                excel_path += '.xlsx'
    
            # Phase 2: Check prerequisites and ask for transformation if needed
            df_copy = self.df.copy()
            
            try:
                # Prepare test and check prerequisites
                test_preparation = StatisticalTester.prepare_advanced_test(
                    df=df_copy,
                    test=test_type,
                    dv=dialog.dvCombo.currentText(),
                    subject=dialog.subjectCombo.currentText(),
                    between=between,
                    within=within
                )
                
                # Check if an error occurred
                if "error" in test_preparation:
                    QMessageBox.critical(self, "Error", f"Error preparing the test: {test_preparation['error']}")
                    return
                
                # Transformation needed? If yes, show SEPARATE WINDOW
                transformation = None
                # Use the UIDialogManager to maintain consistent dialog state
                transformation = UIDialogManager.select_transformation_dialog(
                    parent=self, 
                    progress_text="for Advanced Test",
                    column_name=dialog.dvCombo.currentText()
                )
                if transformation is None:
                    # Dialog cancelled
                    return
                
                # Phase 3: Perform test with specified Excel path
                results = StatisticalTester.perform_advanced_test(
                    df=self.df,
                    test=test_type,
                    dv=dialog.dvCombo.currentText(),
                    subject=dialog.subjectCombo.currentText(),
                    between=between,
                    within=within,
                    alpha=0.05,  # Use default value
                    force_parametric=False,  # Never force
                    manual_transform=transformation,
                    file_name=excel_path,  # NEW: Specific Excel file path
                    skip_excel=False  # Always create Excel file
                )

                print("DEBUG: After RM ANOVA execution")
                print("DEBUG: Current results structure:", results.keys() if isinstance(results, dict) else type(results))
                if isinstance(results, dict) and 'error' in results and results['error']:
                    print("DEBUG: Error detected:", results['error'])
                    traceback.print_exc()

                QMessageBox.information(self, "Test completed", 
                    f"The advanced test was successfully performed.\nResults were saved in:\n{excel_path}")
            except Exception as e:
                # Add this except block to handle errors
                QMessageBox.critical(self, "Error", f"Error performing the test: {str(e)}")
                traceback.print_exc()

    def detect_data_format(self, df):
        """Detects the data format based on column names"""
        columns = set(df.columns)
        columns_lower = {col.lower() for col in columns}  # Convert to lowercase for more robust detection
        format_type = "unknown"
        
        # Check for Subject column - required for both RM and Mixed ANOVA
        has_subject = 'subject' in columns_lower or any(col.startswith('s') and col[1:].isdigit() for col in columns_lower)
        
        # Check for Timepoint column - indicates a within-subjects factor
        has_timepoint = 'timepoint' in columns_lower or 'time' in columns_lower or 'zeit' in columns_lower
        
        # Check for obvious between-subjects factor columns
        has_between = 'group' in columns_lower or 'gruppe' in columns_lower or 'treatment' in columns_lower or 'condition' in columns_lower
        
        # Repeated Measures ANOVA: Subject + Timepoint without obvious between factors
        if has_subject and has_timepoint and not has_between:
            format_type = "repeated_measures_anova"
        # Mixed ANOVA: Subject + Timepoint + Between factors
        elif has_subject and has_timepoint and has_between:
            format_type = "mixed_anova"
        # Two-Way ANOVA format (FactorA, FactorB, Value)
        elif ('factora' in columns_lower and 'factorb' in columns_lower) or \
            any('factor' in col.lower() for col in columns):
            format_type = "two_way_anova"
        
        print(f"Detected data format: {format_type}")
        return format_type
    
    def select_groups_for_plot(self):
        """Opens a dialog to select groups for a plot."""
        if not self.available_groups:
            QMessageBox.warning(self, "Error", "No groups available. Please load data first.")
            return
        
        dialog = GroupSelectionDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_groups = dialog.get_selected_groups()
            if selected_groups:
                self.configure_plot(selected_groups)
            else:
                QMessageBox.warning(self, "Error", "No groups selected.")
    
    def configure_plot(self, groups):
        """Opens a dialog to configure a plot with the selected groups."""
        dialog = PlotConfigDialog(groups, self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            self.plot_configs.append(config)
            
            # Display in the plot list
            if config['file_name']:
                plot_item_text = f"Plot: {config['file_name']}"
            else:
                title_text = config['title'] if config['title'] else f"Plot with {', '.join(config['groups'])}"
                plot_item_text = f"Plot: {title_text}"
            if not config['create_plot']:
                plot_item_text += " (analysis only)"
            
            self.plots_list.addItem(plot_item_text)
            
            # Create preview
            self.preview_plot(len(self.plot_configs) - 1)
    
    def edit_plot_config(self, item):
        """Edits the configuration of a selected plot."""
        index = self.plots_list.row(item)
        
        if index < 0 or index >= len(self.plot_configs):
            return
        
        config = self.plot_configs[index]
        
        dialog = PlotConfigDialog(config['groups'], self)
        
        # Set all properties of the dialog
        dialog.set_title(config.get('title', ''))
        dialog.set_x_label(config.get('x_label', ''))
        dialog.set_y_label(config.get('y_label', ''))
        dialog.set_file_name(config.get('file_name', ''))
        dialog.set_width(config.get('width', 12))
        dialog.set_height(config.get('height', 10))
        
        # Dependent samples
        dialog.dependent_check.setChecked(config.get('dependent', False))
        
        # Create plot or skip
        dialog.create_plot_check.setChecked(config.get('create_plot', True))
        
        # Error bar type
        if config.get('error_type', 'sd') == 'se':
            dialog.error_type_sem.setChecked(True)
        else:
            dialog.error_type_sd.setChecked(True)
        
        # Set colors and hatches
        for group in config['groups']:
            if group in config['colors']:
                dialog.color_buttons[group].setStyleSheet(f"background-color: {config['colors'][group]};")
            if group in config['hatches'] and dialog.hatch_combos[group].findText(config['hatches'][group]) >= 0:
                dialog.hatch_combos[group].setCurrentText(config['hatches'][group])
        
        # Set comparisons
        dialog.comparisons = config.get('comparisons', [])
        dialog.comparisons_list.clear()
        for comp in dialog.comparisons:
            dialog.comparisons_list.addItem(f"{comp['group1']} vs {comp['group2']} ({comp.get('test_type', 'Unknown')})")
        
        # Two-Way ANOVA additional factors
        if 'additional_factors' in config and config['additional_factors']:
            dialog.additional_factors = config['additional_factors']
        
        if dialog.exec_() == QDialog.Accepted:
            # Update the configuration with new values
            self.plot_configs[index] = dialog.get_config()
            
            # Update display in the list
            if self.plot_configs[index].get('file_name'):
                plot_item_text = f"Plot: {self.plot_configs[index]['file_name']}"
            else:
                title_text = self.plot_configs[index].get('title') if self.plot_configs[index].get('title') else f"Plot with {', '.join(self.plot_configs[index]['groups'])}"
                plot_item_text = f"Plot: {title_text}"
                
            if not self.plot_configs[index].get('create_plot', True):
                plot_item_text += " (analysis only)"
                
            item.setText(plot_item_text);
            
            # Update the preview
            self.preview_plot(index)
    
    def get_analysis_params(self):
        # Implement this method to return analysis parameters
        return {
            'file_path': self.file_path,
                       'group_col': self.group_col_combo.currentText(),
            # Add more parameters here
        }
    
    def display_results(self, results):
        # Error handling
        if 'error' in results and results['error'] is not None and results['error']:
            QMessageBox.critical(self, "Error", f"Analysis error: {results['error']}")
            return
    
        analysis_log = ""
        
        # Add main results of the ANOVA
        analysis_log += f"Test: {results.get('test', 'Unknown')}\n"
        
        # Safe formatting for p-value
        if 'p_value' in results and results['p_value'] is not None:
            analysis_log += f"p-value: {results['p_value']:.4f}\n"
        else:
            analysis_log += "p-value: Not available\n"
        
        # Safe formatting for test statistic
        if 'statistic' in results and results['statistic'] is not None:
            analysis_log += f"Test statistic: {results['statistic']:.4f}\n"
        else:
            analysis_log += "Test statistic: Not available\n"
        
        # Add status of post-hoc tests
        if results.get('posthoc_status') == 'not_performed':
            analysis_log += f"\nPost-hoc tests: Not performed\n"
            analysis_log += f"Reason: {results.get('posthoc_reason', 'Unknown')}\n"
        
        # Pairwise comparisons, if available
        if 'pairwise_comparisons' in results and results['pairwise_comparisons']:
            analysis_log += "\nPairwise comparisons:\n"
            for comp in results["pairwise_comparisons"]:
                analysis_log += (f"{comp['group1']} vs {comp['group2']}: "
                            f"p = {comp['p_value']:.4g}, "
                            f"significant: {'yes' if comp['significant'] else 'no'}\n")
        else:
            analysis_log += "\nNo pairwise comparisons were performed or calculated.\n"
    
        # Debug output
        print("Analysis results:", analysis_log)
        QMessageBox.information(self, "Analysis Results", analysis_log)
    
    def direct_group_comparison(self):
        """Performs a direct comparison between two selected groups."""
        if not self.available_groups:
            QMessageBox.warning(self, "Error", "No groups available. Please load data first.")
            return
        
        dialog = PairwiseComparisonDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            comp = dialog.get_comparison()
            if comp['group1'] == comp['group2']:
                QMessageBox.warning(self, "Error", "The two groups must be different!")
                return
            
            self.run_direct_comparison(comp)
    
    def run_direct_comparison(self, comp):
        """Performs a direct statistical comparison between two groups."""
        if not self.samples:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
        
        try:
            # Extract the two groups
            group1 = comp['group1']
            group2 = comp['group2']
            
            # Check if both groups are present in the data
            if group1 not in self.samples or group2 not in self.samples:
                QMessageBox.warning(self, "Error", f"One or both groups ({group1}, {group2}) not found in the data.")
                return
            
            # Create transformed samples for statistical tests
            groups = [group1, group2]
            transformed_samples, test_recommendation, _ = StatisticalTester.check_normality_and_variance(groups, self.samples)
            
            # Determine test type
            if comp['test_type'] == "t-Test (parametric)":
                test_recommendation = "parametric"
            elif comp['test_type'] == "Mann-Whitney-U (non-parametric)":
                test_recommendation = "non_parametric"
            
            # Perform test and save results
            results = StatisticalTester.perform_statistical_test(groups, transformed_samples, self.samples, 
                                dependent=comp['dependent'], 
                                test_recommendation=test_recommendation)
            
            # Visualization
            colors = DEFAULT_COLORS[:2]  # Use the first two default colors
            hatches = DEFAULT_HATCHES[:2]  # Use the first two default hatches
            
            # Extract pairwise_comparisons from results
            pairwise_comparisons = results.get('pairwise_comparisons', None)
            
            DataVisualizer.plot_bar(groups, self.samples, width=8, height=6, 
                colors=colors, hatches=hatches, 
                compare=[(group1, group2)],
                test_recommendation=test_recommendation,
                pairwise_results=pairwise_comparisons)  # Pass pairwise_comparisons
    
            # Show results
            self.display_results(results)
            
            QMessageBox.information(self, "Success", 
                                f"Direct comparison between {group1} and {group2} was performed and visualized.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing the comparison: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def remove_plot(self):
        """Removes the selected plot from the list of configurations."""
        current_row = self.plots_list.currentRow()
        if current_row >= 0:
            self.plots_list.takeItem(current_row)
            self.plot_configs.pop(current_row)
            
            # Clear the preview if no plot is left
            if len(self.plot_configs) == 0:
                self.figure.clear()
                self.canvas.draw()
    
    def preview_selected_plot(self):
        """Creates a preview of the selected plot."""
        current_row = self.plots_list.currentRow()
        if current_row >= 0:
            self.preview_plot(current_row)
        else:
            QMessageBox.warning(self, "Error", "Please select a plot from the list.")
    
    def preview_plot(self, plot_idx):
        """Creates a preview of a plot based on its configuration."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        if plot_idx < 0 or plot_idx >= len(self.plot_configs):
            return
        
        plot_config = self.plot_configs[plot_idx]
        
        try:
            # Clear the figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Prepare data - with error checking
            plot_samples = {}
            # Check if 'groups' exists in plot_config
            if 'groups' not in plot_config:
                QMessageBox.warning(self, "Error", "Configuration contains no groups.")
                return
                
            for group in plot_config.get('groups', []):
                if self.samples and group in self.samples:
                    plot_samples[group] = self.samples[group]
            
            if not plot_samples:
                QMessageBox.warning(self, "Warning", "No data found for the selected groups.")
                return
            
            # Prepare data for DataFrame
            plot_data = []
            for group, values in plot_samples.items():
                for value in values:
                    plot_data.append({'Group': group, 'Value': value})
            
            df = pd.DataFrame(plot_data)
            
            # Create color list and hatch list
            colors = [plot_config['colors'].get(group, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) 
                    for i, group in enumerate(plot_config['groups'])]
            hatches = [plot_config['hatches'].get(group, DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)]) 
                    for i, group in enumerate(plot_config['groups'])]
            
            # Error type from configuration
            error_type = plot_config.get('error_type', 'sd')
            
            # Bar plot
            bars = sns.barplot(x='Group', y='Value', data=df, ax=ax, errorbar=error_type, 
                  palette=['#E0E0E0', '#A0A0A0', '#707070', '#505050', '#303030', '#000000'], 
                  capsize=0.1, edgecolor='black', linewidth=1)
            
            # Remove top and right spines
            sns.despine(ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Enable scientific notation for y-axis
            from matplotlib.ticker import ScalarFormatter
            y_formatter = ScalarFormatter(useOffset=False)
            y_formatter.set_scientific(True)
            y_formatter.set_powerlimits((-3, 4))  # Scientific notation for values outside 10^-3 to 10^4
            ax.yaxis.set_major_formatter(y_formatter)
            
            # Apply hatches
            for i, patch in enumerate(bars.patches):
                group_idx = i % len(plot_config['groups'])
                patch.set_hatch(hatches[group_idx])
                patch.set_edgecolor('black')  # Black border for all bars
                patch.set_linewidth(1)        # Border width
            
            # Add data points (jitter)
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
            
            # For the jitter part in preview_plot:
            for i, group in enumerate(plot_config['groups']):
                if group in plot_samples:
                    values = plot_samples[group]
                    if not values:  # Check if values are present
                        continue
                        
                    x_pos = i
                    
                    # Safer method to calculate bar width
                    try:
                        # Check if bars and patches exist
                        if hasattr(bars, 'patches') and bars.patches and i < len(bars.patches):
                            bar_width = bars.patches[i].get_width() * 0.8  # Use 80% of bar width
                        else:
                            bar_width = 0.8  # Default if patches not available
                            
                        # Jitter for x-position - distribute evenly over bar width
                        if len(values) > 1:
                            jitter = np.linspace(-bar_width/2, bar_width/2, len(values))
                        else:
                            jitter = [0]  # Center if only one value
                            
                        # Add some randomness to avoid perfect lines
                        jitter += np.random.normal(0, 0.01, size=len(values))
                    except Exception as e:
                        # Fallback if there are problems with jitter calculation
                        print(f"Error in jitter calculation: {str(e)}")
                        jitter = np.random.normal(0, 0.1, size=len(values))  # Simple jitter
                    
                    for j, (val, jit) in enumerate(zip(values, jitter)):
                        marker_idx = j % len(markers)
                        ax.scatter(x_pos + jit, val, color='black', 
                                marker=markers[marker_idx], s=40, zorder=3, alpha=0.8)
            
            # Format plot
            if plot_config.get('title'):
                ax.set_title(plot_config['title'])
            
            # Axis labels, only if specified
            if plot_config.get('x_label'):
                ax.set_xlabel(plot_config['x_label'])
            if plot_config.get('y_label'):
                ax.set_ylabel(plot_config['y_label'])
            
            ax.tick_params(axis='x', rotation=45)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error creating preview: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def run_selected_analysis(self):
        """Runs the analysis for the selected plot only."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        current_row = self.plots_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Error", "Please select a plot from the list.")
            return
            
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
            
        try:
            plot_config = self.plot_configs[current_row]
            self.run_single_analysis(plot_config, output_dir)
            QMessageBox.information(self, "Success", "Plot was successfully created and saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_all_analyses(self):
        """Runs all configured analyses in sequence."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        if not self.plot_configs:
            QMessageBox.warning(self, "Warning", "No plot configurations available.")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
        
        # Run all analyses
        success_count = 0
        for i, plot_config in enumerate(self.plot_configs):
            try:
                self.run_single_analysis(plot_config, output_dir)
                success_count += 1
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error in plot {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        QMessageBox.information(self, "Success", f"{success_count} of {len(self.plot_configs)} plots were successfully created and saved.")
    
    def run_single_analysis(self, plot_config, output_dir=None):
        """Runs the analysis for a specific plot configuration.
        
        Args:
            plot_config (dict): Configuration for the plot
            output_dir (str, optional): Output directory. Default is None.
        """
        if self.samples is None:
            raise ValueError("No data loaded.")
        
        # Check if all groups have data
        for group in plot_config['groups']:
            if group not in self.samples or not self.samples[group]:
                QMessageBox.warning(self, "Warning", f"Group '{group}' has no data or does not exist.")
                return
            
        # Determine columns to use for analysis
        if len(self.selected_columns) > 1:
            value_cols = self.selected_columns
        else:
            value_cols = [self.value_cols_combo.currentText()]
        
        # Prepare parameters for analyze()
        kwargs = {
            'file_path': self.file_path,
            'group_col': self.group_col_combo.currentText(),
            'groups': plot_config['groups'],
            'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
            'width': plot_config['width'],
            'height': plot_config['height'],
            'dependent': plot_config['dependent'],
            'combine_columns': self.combine_columns,
            'skip_plots': not plot_config.get('create_plot', True),
            'x_label': plot_config.get('x_label'),
            'y_label': plot_config.get('y_label'),
            'title': plot_config.get('title'),
            'error_type': plot_config.get('error_type', 'sd'),
            'file_name': plot_config.get('file_name', f"{plot_config['groups'][0]}_vs_{plot_config['groups'][1]}"),
            'show_individual_lines': plot_config.get('show_individual_lines', True),
            'value_cols': value_cols,
        }
        
        # Add additional_factors for Two-Way ANOVA if present
        if 'additional_factors' in plot_config and plot_config['additional_factors']:
            kwargs['additional_factors'] = plot_config['additional_factors']
            
        # Add compare parameter if comparisons are present
        if 'comparisons' in plot_config and plot_config['comparisons']:
            # Create list of tuples for compare parameter
            compare_list = [(comp['group1'], comp['group2']) for comp in plot_config['comparisons']]
            kwargs['compare'] = compare_list
        
        # Create color and hatch lists based on configuration
        colors = [plot_config['colors'].get(group, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) 
                 for i, group in enumerate(plot_config['groups'])]
        hatches = [plot_config['hatches'].get(group, DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)]) 
                  for i, group in enumerate(plot_config['groups'])]
        
        kwargs['colors'] = colors
        kwargs['hatches'] = hatches
        # Validation for dependent samples
        if plot_config['dependent']:
            validation = StatisticalTester.validate_dependent_data(self.samples, plot_config['groups'])
            if not validation["valid"]:
                error_msg = "Problem with dependent samples:\n\n" + "\n".join(validation["messages"])
                
                # Option to continue or abort
                reply = QMessageBox.warning(
                    self, 
                    "Validation problem for dependent samples", 
                    error_msg + "\n\nDo you want to continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    QMessageBox.information(
                        self, "Analysis aborted", 
                        "You should uncheck the 'Dependent samples' box or "
                        "ensure your data meet the requirements."
                    )
                    return
                    
                # If user wants to continue, add a warning to kwargs
                kwargs['force_continue'] = True
                
        # Save original working directory
        original_cwd = os.getcwd()
        
        try:
            # If output_dir is specified, change to it
            if output_dir:
                os.chdir(output_dir)
                print(f"Changed output directory to: {output_dir}")
                
            # Ask for subject ID and within factor for dependent tests if needed
            if plot_config.get('dependent', False) and plot_config.get('needs_subject_selection', False):
                from PyQt5.QtWidgets import QDialog, QFormLayout, QComboBox, QDialogButtonBox, QVBoxLayout, QLabel
                
                # Dialog to select subject and within factor
                subject_dialog = QDialog(self)
                subject_dialog.setWindowTitle("Select Subject ID and Within Factor")
                subject_layout = QVBoxLayout(subject_dialog)
                
                # Explanation
                info_label = QLabel("The following assignments are required for dependent tests:")
                subject_layout.addWidget(info_label)
                
                form_layout = QFormLayout()
                
                # Subject ID
                subject_combo = QComboBox()
                # Add all columns from the DataFrame
                subject_combo.addItems(self.df.columns)
                # Preselect if available 
                for i, col in enumerate(self.df.columns):
                    if col.lower() in ('subject', 'subjekt', 'id', 'patient'):
                        subject_combo.setCurrentIndex(i)
                        break
                form_layout.addRow("Subject ID column:", subject_combo)
                
                # Within factor
                within_combo = QComboBox()
                within_combo.addItems(self.df.columns)
                # Preselect if available
                for i, col in enumerate(self.df.columns):
                    if col.lower() in ('timepoint', 'zeit', 'messung', 'time', 'within'):
                        within_combo.setCurrentIndex(i)
                        break
                form_layout.addRow("Within factor column:", within_combo)
                
                subject_layout.addLayout(form_layout)
                
                # OK/Cancel buttons
                button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                button_box.accepted.connect(subject_dialog.accept)
                button_box.rejected.connect(subject_dialog.reject)
                subject_layout.addWidget(button_box)
                
                if subject_dialog.exec_() == QDialog.Accepted:
                    # Take selected values into kwargs
                    kwargs['subject_column'] = subject_combo.currentText()
                    kwargs['within_column'] = within_combo.currentText()
                else:
                    # Abort analysis if cancelled
                    return
            
            # Run the analysis
            results = AnalysisManager.analyze(**kwargs)
            
            if 'error' in results and results['error'] is not None and results['error']:
                QMessageBox.critical(self, "Error", f"Error during analysis: {results['error']}")
                return
                
            # Check for file path info in results
            if '_file_paths' in results:
                file_paths = results['_file_paths']
                files_info = []
                
                for file_type, path in file_paths.items():
                    if os.path.exists(path):
                        files_info.append(f"{file_type.upper()}: {path}")
                
                if files_info:
                    QMessageBox.information(self, "Success", 
                                          f"Analysis successful! The following files were created:\n\n" + 
                                          "\n".join(files_info))
                else:
                    QMessageBox.warning(self, "Warning", 
                                       f"The analysis was performed, but no output files could be found.")
            else:
                # Fallback to old method of file detection
                group_name = "_".join(plot_config['groups'])
                pdf_file = os.path.join(os.getcwd(), f"{group_name}.pdf")
                png_file = os.path.join(os.getcwd(), f"{group_name}.png")
                excel_file = os.path.join(os.getcwd(), f"{group_name}_results.xlsx")
                
                files_info = []
                if os.path.exists(pdf_file):
                    files_info.append(f"PDF: {pdf_file}")
                if os.path.exists(png_file):
                    files_info.append(f"PNG: {png_file}")
                if os.path.exists(excel_file):
                    files_info.append(f"Excel: {excel_file}")
                
                if files_info:
                    QMessageBox.information(self, "Success", 
                                          f"Analysis successful! The following files were created:\n\n" + 
                                          "\n".join(files_info))
                else:
                    QMessageBox.warning(self, "Warning", 
                                       f"The analysis was performed, but no output files were found in the directory:\n{os.getcwd()}")
        
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
        finally:
            # Cleanup
            print(f"Switching back to original directory: {original_cwd}")
            os.chdir(original_cwd)
            
    # Neue Methode für die Anzeige einer Hilfefunktion zu abhängigen Stichproben
    def show_dependent_samples_help(self):
        QMessageBox.information(
            self,
            "Help for Dependent Samples",
            "<h3>When are samples dependent?</h3>"
            "<p>Dependent samples arise when:</p>"
            "<ul>"
            "<li>Measurements are taken on the <b>same subject</b> at different time points</li>"
            "<li>Measurements are naturally paired (e.g. left and right eye)</li>"
            "<li>Experiments are conducted with repeated measurements</li>"
            "</ul>"
            "<h3>Data structure for dependent tests</h3>"
            "<p>For dependent tests, each group must:</p>"
            "<ul>"
            "<li>Contain the <b>same number</b> of measurements</li>"
            "<li>Have measurements in <b>matching order</b></li>"
            "</ul>"
            "<p>Example: Measurement 1 in group A and measurement 1 in group B must be from the same subject</p>"
            "<h3>Available tests</h3>"
            "<ul>"
            "<li><b>Two groups:</b> Paired t-test or Wilcoxon signed-rank test</li>"
            "<li><b>More than two groups:</b> Repeated Measures ANOVA or Friedman test</li>"
            "</ul>"
        )
    
    def show_advanced_tests_help(self):
        """Shows an overview of ANOVA types and between/within factors."""
        QMessageBox.information(
            self,
            "Advanced Tests",
            "<h3>Types of ANOVA</h3>"
            "<ul>"
            "<li><b>One-way ANOVA:</b> Compare means of ≥ 2 independent groups</li>"
            "<li><b>Welch-ANOVA:</b> One-way ANOVA with unequal variances</li>"
            "<li><b>Kruskal-Wallis:</b> Non-parametric alternative to one-way ANOVA</li>"
            "<li><b>Two-way ANOVA:</b> Two between factors (e.g. gender × treatment)</li>"
            "<li><b>Repeated Measures ANOVA:</b> ≥ 1 within factor (repeated measurements on the same subject)</li>"
            "<li><b>Mixed ANOVA:</b> Combination of between and within factors</li>"
            "</ul>"
            "<h3>Between vs. Within Factors</h3>"
            "<p><b>Between factor:</b> Groups are independent (different subjects)</p>"
            "<p><b>Within factor:</b> Same subjects are measured multiple times (e.g. before/after)</p>"
        )
    
    def run_multi_dataset_analysis(self):
        """Runs separate analyses for multiple datasets, 
        with individual plot configuration and a shared Excel file."""
        print("DEBUG MULTI: ENTERED run_multi_dataset_analysis()")
        print("DEBUG MULTI:   self.multi_dataset_analysis =", getattr(self, "multi_dataset_analysis", None))
        print("DEBUG MULTI:   self.selected_columns =", getattr(self, "selected_columns", None))
        
        if not hasattr(self, 'multi_dataset_analysis') or not self.multi_dataset_analysis:
            QMessageBox.warning(self, "Warning", "Please select multi-dataset analysis in the column selection dialog first.")
            return
            
        if len(self.selected_columns) <= 1:
            QMessageBox.warning(self, "Warning", "Multi-dataset analysis requires multiple selected datasets.")
            return
        print("DEBUG MULTI: Passed all pre-checks.  → proceed with multi-dataset loop")
        print("DEBUG MULTI:   selected_columns =", self.selected_columns)
        print("DEBUG MULTI:   available_groups =", self.available_groups)
        try:
            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select output directory for multi-dataset analysis")
            if not output_dir:
                print("No output directory selected")
                return

            # Select groups for analysis
            print("Opening group dialog...")
            dialog = GroupSelectionDialog(self.available_groups, self)
            if dialog.exec_() != QDialog.Accepted:
                print("Group dialog cancelled")
                return
                
            selected_groups = dialog.get_selected_groups()
            if not selected_groups:
                QMessageBox.warning(self, "Warning", "Please select at least one group.")
                return

            all_results = {}
            plot_configs = {}

            # Remember current working directory
            original_cwd = os.getcwd()
            
            try:
                os.chdir(output_dir)
                print(f"Changed to output directory: {output_dir}")

                # 1. For each value column (e.g. gene): plot configuration & analysis
                for i, column in enumerate(self.selected_columns):
                    print(f"Configuring plot for column {column} ({i+1}/{len(self.selected_columns)})")
                    print(f"DEBUG MULTI:   ► iterating dataset #{i+1}: '{column}'")
                    print(f"DEBUG MULTI:      → plot_configs keys so far: {list(plot_configs.keys())}")
                    # --- Plot configuration per dataset ---
                    config_dialog = PlotConfigDialog(selected_groups, self)
                    config_dialog.setWindowTitle(f"Configure plot for '{column}' ({i+1}/{len(self.selected_columns)})")
                    config_dialog.set_title(column)
                    config_dialog.set_file_name(f"{column}_analysis")
                    
                    if config_dialog.exec_() != QDialog.Accepted:
                        print(f"Configuration for {column} cancelled")
                        continue
                        
                    plot_config = config_dialog.get_config()
                    plot_configs[column] = plot_config
                    print(f"Configuration for {column} saved")

                if not plot_configs:
                    QMessageBox.warning(self, "Aborted", "No datasets were configured.")
                    return

                # Progress dialog for analysis
                print("Creating progress dialog...")
                progress = QMessageBox()
                progress.setWindowTitle("Analysis running...")
                progress.setText("The multi-dataset analysis is running. Please wait...")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()

                # 2. Analyze all configured datasets in sequence
                for i, (column, plot_config) in enumerate(plot_configs.items()):
                    progress_text = f"({i+1}/{len(plot_configs)})"
                    progress.setText(f"Analyzing dataset {i+1}/{len(plot_configs)}: {column}")
                    QApplication.processEvents()
                    
                    print(f"Starting analysis for {column}...")
                    try:
                        # --- Prepare values for analyze() ---
                        kwargs = {
                            'file_path': self.file_path,
                            'group_col': self.group_col_combo.currentText(),
                            'groups': plot_config['groups'],
                            'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
                            'value_cols': [column],
                            'combine_columns': False,
                            'width': plot_config.get('width', 12),
                            'height': plot_config.get('height', 10),
                            'dependent': plot_config.get('dependent', False),
                            'skip_plots': not plot_config.get('create_plot', True),
                            # *** NEW: skip_excel=True prevents individual Excel files for each dataset
                            'skip_excel': True,
                            'x_label': plot_config.get('x_label'),
                            'y_label': plot_config.get('y_label'),
                            'title': plot_config.get('title', column),
                            'error_type': plot_config.get('error_type', 'sd'),
                            'file_name': plot_config.get('file_name', f"{column}_analysis"),
                            'dataset_name': column,
                            'dialog_column': column,
                            'dialog_progress': progress_text
                        }

                        # Two-Way ANOVA factors
                        if 'additional_factors' in plot_config and plot_config['additional_factors']:
                            kwargs['additional_factors'] = plot_config['additional_factors']
                            
                        # Specific comparisons
                        if 'comparisons' in plot_config and plot_config['comparisons']:
                            compare_list = [(comp['group1'], comp['group2']) for comp in plot_config['comparisons']]
                            kwargs['compare'] = compare_list
                            
                        # Colors and hatches
                        colors = [plot_config['colors'].get(group, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                                for j, group in enumerate(plot_config['groups'])]
                        kwargs['colors'] = colors
                        
                    # Run analysis (without individual Excel export)
                        print(f"Calling AnalysisManager.analyze() for {column} with skip_excel=True...")
                        start_time = time.time()  # Track execution time
                        results = AnalysisManager.analyze(**kwargs)
                        analysis_time = time.time() - start_time
                        
                        # Validate results structure
                        print(f"Analysis for {column} completed in {analysis_time:.2f} seconds")
                        print(f"Result type: {type(results)}")
                        print(f"Keys in results: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
                        
                        # Check for critical keys
                        if isinstance(results, dict):
                            if 'error' in results and results['error']:
                                print(f"WARNING: Error found in results: {results['error']}")
                            if 'test' in results:
                                print(f"Test performed: {results['test']}")
                            if 'p_value' in results:
                                print(f"p-value: {results['p_value']}")
                        
                        # Store results
                        all_results[column] = results
                        print(f"✓ Results for '{column}' successfully stored in all_results dictionary")
                        print(f"Analysis for {column} completed")

                    except Exception as e:
                        QMessageBox.critical(self, "Error",
                                        f"Error analyzing {column}: {str(e)}")
                        traceback.print_exc()
                        
                print("DEBUG MULTI: Finished analyzing all columns.")
                print("DEBUG MULTI: all_results keys:", list(all_results.keys()))
                print("DEBUG MULTI: all_results contents:")
                for key, value in all_results.items():
                    print(f"  {key}: {type(value)} with keys {list(value.keys()) if isinstance(value, dict) else 'not a dict'}")

                if all_results:
                    print("DEBUG MULTI: About to call export_multi_dataset_results()")
                    excel_path = os.path.join(output_dir, "All_Datasets_Analysis.xlsx")
                    print(f"DEBUG MULTI: Excel path will be: {excel_path}")
                    ResultsExporter.export_multi_dataset_results(all_results, excel_path)
                    print("DEBUG MULTI: export_multi_dataset_results() completed successfully")
                    
                    # Show success message with file path
                    QMessageBox.information(self, "Multi-Dataset Analysis Complete", 
                        f"Analysis completed successfully!\n\n"
                        f"Combined results saved to:\n{excel_path}\n\n"
                        f"Analyzed {len(all_results)} datasets: {', '.join(all_results.keys())}")
                else:
                    print("DEBUG MULTI: all_results is empty, skipping export")
                    QMessageBox.warning(self, "No Results", "No analysis results were generated.")

                # Ensure the progress dialog is closed
                try:
                    progress.close()
                    progress = None  # Delete reference
                except Exception as e:
                    print(f"Error closing progress dialog: {str(e)}")
                    
            except Exception as e:
                print(f"ERROR in main flow of multi-dataset analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Critical error", 
                                f"An unexpected error occurred: {str(e)}")
            finally:
                print(f"Switching back to original directory: {original_cwd}")
                os.chdir(original_cwd)
        except Exception as e:
            print(f"CRITICAL ERROR in run_multi_dataset_analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Critical error", 
                            f"A serious error occurred: {str(e)}")
            
    def configure_two_way_anova(self):
        """Configure Two-Way ANOVA"""
        # IMPORTANT: Use a local variable for the status
        current_factors = None

        # Create and execute dialog
        dialog = TwoWayAnovaDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            # Store status only in local variable
            current_factors = dialog.get_factor_data()

            # Use status only for this action, not as an instance variable
            if current_factors:
                # Immediately run analysis with the factors
                self.run_analysis_with_factors(current_factors)

    def run_analysis_with_factors(self, factors):
        """Run analysis with the given factors"""
        try:
            # Prepare parameters for the analysis
            params = self.get_analysis_params()

            # Explicitly add factors for this call
            params['additional_factors'] = factors

            # Run analysis
            results = AnalysisManager.analyze(**params)

            # Show results
            self.display_results(results)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error during analysis with factors: {str(e)}")

    def run_pairwise_comparison(self):
        """Perform pairwise comparison"""
        dialog = PairwiseComparisonDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            comparison_data = dialog.get_comparison()
            if comparison_data:
                # Use the existing run_direct_comparison
                self.run_direct_comparison(comparison_data)

    def run_outlier_detection(self):
        """Run outlier detection analysis"""
        # Check if data is loaded
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded. Please load a file first.")
            return
        
        try:
            # Create and show the outlier detection dialog
            dialog = OutlierDetectionDialog(self.df, self)
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_config()
                if config is None:
                    return
                
                # Create progress dialog
                progress = QMessageBox()
                progress.setWindowTitle("Outlier Detection")
                progress.setText("Running outlier detection analysis...")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()
                
                try:
                    if config['is_multi_dataset']:
                        # Multi-dataset analysis
                        results = OutlierDetector.run_multi_dataset_outlier_detection(
                            df=self.df,
                            group_col=config['group_column'],
                            dataset_columns=config['dataset_columns'],
                            # alpha parameter removed
                            iterate=config['iterate'],
                            run_grubbs=config['run_grubbs'], 
                            run_modz=config['run_modz'],
                            output_path=config['output_file']
                        )
                        
                        # Show multi-dataset results
                        summary = f"Multi-dataset outlier detection completed!\n\n"
                        summary += f"Analyzed {len(config['dataset_columns'])} datasets:\n"
                        summary += f"{', '.join(config['dataset_columns'])}\n\n"
                        summary += f"Results saved to: {config['output_file']}"
                        
                    else:
                        # Single dataset analysis (existing code)
                        detector = OutlierDetector(
                            df=self.df.copy(),
                            group_col=config['group_column'],
                            value_col=config['value_column']
                        )

                        if config['run_modz']:
                            detector.run_mod_z_score(threshold=3.5, iterate=config['iterate'])
                        
                        if config['run_grubbs']:
                            detector.run_grubbs_test(alpha=0.05, iterate=config['iterate'])
                        
                        detector.save_results(config['output_file'])
                        
                        outlier_count = 0
                        if 'ModZ_Outlier' in detector.df.columns:
                            outlier_count += detector.df['ModZ_Outlier'].sum()
                        if 'Grubbs_Outlier' in detector.df.columns:
                            outlier_count += detector.df['Grubbs_Outlier'].sum()
                        
                        summary = f"Outlier detection completed!\n\n"
                        summary += f"Dataset: {config['value_column']}\n"
                        summary += f"Total outliers found: {outlier_count}\n\n"
                        summary += f"Results saved to: {config['output_file']}"
                    
                    progress.close()
                    QMessageBox.information(self, "Outlier Detection Complete", summary)
                    
                except Exception as e:
                    progress.close()
                    QMessageBox.critical(self, "Error", f"Error during outlier detection: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening outlier detection dialog: {str(e)}")
            import traceback
            traceback.print_exc()

def resource_path(relative_path):
    """ Funktion, die den Pfad zur Ressource findet – auch wenn es eine .exe ist """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


if __name__ == "__main__":
    try:
        # Apply stylesheet if available
        try:
            with open(resource_path("StyleSheet.qss"), "r", encoding="utf-8") as f:
                stylesheet = f.read()
                print("Stylesheet loaded successfully")
        except:
            stylesheet = ""
            print("No stylesheet found")
        
        app = QApplication(sys.argv)
        app.setStyleSheet(stylesheet)
        window = StatisticalAnalyzerApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()