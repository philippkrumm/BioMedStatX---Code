from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QDialogButtonBox, QWidget, QScrollArea
from PyQt5.QtCore import Qt

class ComparisonSelectionDialog(QDialog):
    def __init__(self, comparisons, parent=None, checked_by_default=True):
        super().__init__(parent)
        self.setWindowTitle("Select Group Comparisons for Post-hoc Test")
        self.selected = set()
        self.comparisons = comparisons
        self.checkboxes = []
        self.checked_by_default = checked_by_default
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Select the group comparisons you want to include in the post-hoc test:")
        layout.addWidget(label)

        # Scroll area for many comparisons
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for comp in self.comparisons:
            text = f"{comp[0]}  vs  {comp[1]}"
            cb = QCheckBox(text)
            cb.setChecked(self.checked_by_default)
            self.checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_comparisons(self):
        selected = []
        for cb, comp in zip(self.checkboxes, self.comparisons):
            if cb.isChecked():
                selected.append(comp)
        return selected
