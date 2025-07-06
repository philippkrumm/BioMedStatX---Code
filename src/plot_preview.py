"""
Preview Widget für Plot-Darstellung
Zeigt Live-Vorschau der Plot-Einstellungen
"""

import sys
import os
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Import der DataVisualizer Klasse
try:
    from stats_functions import DataVisualizer
except ImportError:
    # Fallback wenn Import nicht funktioniert
    print("Warning: Could not import DataVisualizer from stats_functions")
    DataVisualizer = None

class PlotPreviewWidget(FigureCanvasQTAgg):
    """
    Widget zur Live-Vorschau von Plots basierend auf Konfiguration.
    Verwendet die zentrale plot_from_config Methode für konsistente Darstellung.
    """
    
    def __init__(self, parent=None, figsize=(5, 4), dpi=100):
        """
        Initialisiert das Preview Widget mit Live-Vorschau.
        """
        self.fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Initialer subplot
        self.ax = self.fig.add_subplot(111)

        # Daten-Container
        self.groups = []
        self.samples = {}

        # Standard-Konfiguration für alle Appearance-Optionen
        self.default_config = {
            'plot_type': 'Bar',
            'colors': {},
            'hatches': {},
            'alpha': 0.8,
            'error_type': 'sd',
            'error_style': 'caps',
            'bar_edge_color': 'black',
            'bar_linewidth': 0.5,
            'grid': False,
            'grid_style': 'none',
            'minor_ticks': False,
            'despine': True,
            'axis_thickness': 0.7,
            'font_family': 'Arial',
            'show_error_bars': True,
            'show_points': True,
            'point_style': 'jitter',
            'point_size': 80,
            'jitter_strength': 0.3,
            'show_significance_letters': True,
            'show_legend': True,
            'capsize': 0.05,
            'x_label': '',
            'y_label': '',
            'title': '',
            'fontsize_axis': 12,
            'fontsize_title': 14,
            'fontsize_ticks': 10,
            'grid_alpha': 0.3,
            'show_title': True,
            'width': 8,
            'height': 6,
            'dpi': 300,
            'theme': 'default',
            # Raincloud specific defaults
            'group_spacing': 0.90,
            'point_offset': 0.2,
            'point_jitter': 0.05,
            'violin_width': 0.8,
            'box_width': 0.2,
            'frame_thickness': 0.7
        }

        # Initiale leere Darstellung
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Zeigt Platzhalter-Text wenn keine Daten vorhanden"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No data available\nfor preview', 
                     ha='center', va='center', transform=self.ax.transAxes,
                     fontsize=12, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.draw()
    
    def set_data(self, groups, samples):
        """
        Setzt die aktuellen Daten für die Vorschau.
        
        Parameters:
        -----------
        groups : list
            Liste der Gruppennamen
        samples : dict
            Dictionary mit Gruppennamen als Keys und Datenwerten als Values
        """
        self.groups = groups if groups else []
        self.samples = samples if samples else {}
        
        # Validierung der Daten
        if not self.groups or not self.samples:
            self._show_placeholder()
            return
        
        # Prüfe ob alle Gruppen Daten haben
        valid_groups = []
        valid_samples = {}
        
        for group in self.groups:
            if group in self.samples and len(self.samples[group]) > 0:
                valid_groups.append(group)
                valid_samples[group] = self.samples[group]
        
        self.groups = valid_groups
        self.samples = valid_samples
        
        if not self.groups:
            self._show_placeholder()
    
    def update_plot(self, config=None):
        """
        Aktualisiert die Plot-Darstellung basierend auf der Konfiguration.
        
        Parameters:
        -----------
        config : dict, optional
            Konfigurationsdictionary mit Plot-Parametern.
            Wenn None, wird Standard-Konfiguration verwendet.
        """
        if not self.groups or not self.samples:
            self._show_placeholder()
            return
        
        if DataVisualizer is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'DataVisualizer nicht verfügbar', 
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=12, color='red')
            self.ax.axis('off')
            self.draw()
            return
        
        # Verwende Standard-Config wenn keine übergeben
        if config is None:
            config = self.default_config.copy()
        else:
            # Merge mit Standard-Config für fehlende Keys
            merged_config = self.default_config.copy()
            merged_config.update(config)
            config = merged_config
            
        # Ensure colors are set if not provided
        if not config.get('colors') and self.groups:
            # Only set default colors if absolutely no colors are provided
            # This should rarely happen since configs should always include colors
            DEFAULT_COLORS = ['#FF69B4', '#32CD32', '#FFD700', '#00BFFF', '#DA70D6', '#D8BFD8']
            colors_dict = {}
            for i, group in enumerate(self.groups):
                colors_dict[group] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            config['colors'] = colors_dict
            print(f"DEBUG: PlotPreviewWidget set fallback colors: {colors_dict}")
        else:
            print(f"DEBUG: PlotPreviewWidget using provided colors: {config.get('colors', {})}")
        
        try:
            # Clear and redraw
            self.ax.clear()
            
            # Force matplotlib to refresh font cache wenn font_family geändert wurde
            if 'font_family' in config:
                import matplotlib.pyplot as plt
                
                # WICHTIG: Setze Font sofort und explizit ohne _rebuild
                font_family = config['font_family']
                plt.rcParams['font.family'] = font_family
                
                # Erzwinge Update durch rcParams reset
                try:
                    plt.rcdefaults()
                    plt.rcParams['font.family'] = font_family
                except Exception as e:
                    print(f"Font cache update error: {e}")
                    
                # Setze auch für diese Figure explizit
                try:
                    self.fig.suptitle('', fontfamily=font_family)  # Dummy title to force font load
                except:
                    pass
            
            # Verwende die zentrale Dispatcher-Methode
            DataVisualizer.plot_from_config(self.ax, self.groups, self.samples, config)
            
            # Force immediate redraw
            self.draw_idle()
            self.flush_events()
            
        except Exception as e:
            print(f"Error in plot update: {str(e)}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Fehler beim Zeichnen:\n{str(e)}', 
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='red')
            self.ax.axis('off')
            self.draw()
    
    def get_current_config(self):
        """
        Gibt die aktuelle Konfiguration zurück.
        
        Returns:
        --------
        dict
            Aktuelle Konfiguration
        """
        return self.default_config.copy()
    
    def set_default_config(self, config):
        """
        Setzt eine neue Standard-Konfiguration.
        
        Parameters:
        -----------
        config : dict
            Neue Standard-Konfiguration
        """
        if config:
            self.default_config.update(config)


# Test-Funktion für eigenständige Verwendung
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    app = QApplication(sys.argv)
    
    # Test-Daten
    test_groups = ['Group A', 'Group B', 'Group C']
    test_samples = {
        'Group A': np.random.normal(10, 2, 50),
        'Group B': np.random.normal(12, 3, 45),
        'Group C': np.random.normal(8, 1.5, 55)
    }
    
    # Hauptfenster
    main_window = QMainWindow()
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    
    # Preview Widget
    preview = PlotPreviewWidget()
    preview.set_data(test_groups, test_samples)
    preview.update_plot()
    
    layout.addWidget(preview)
    main_window.setCentralWidget(central_widget)
    main_window.setWindowTitle("Plot Preview Test")
    main_window.resize(600, 400)
    main_window.show()
    
    sys.exit(app.exec_())
