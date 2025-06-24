# -*- mode: python ; coding: utf-8 -*-
import os
this_dir = os.getcwd()

from PyInstaller.utils.hooks import collect_data_files

pyqt5_plugins = collect_data_files('PyQt5', subdir='Qt5/plugins')

a = Analysis(
    ['statistical_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
        (os.path.join(this_dir, 'stats_functions.py'), '.'),
        (os.path.join(this_dir, 'StyleSheet.qss'), '.'),
        (os.path.join(this_dir, 'decisiontreevisualizer.py'), '.')
    ] + pyqt5_plugins,
    hiddenimports=[
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'scipy.stats', 'scipy.stats.mstats',
        'statsmodels.stats.multicomp',
        'seaborn',
        'xlsxwriter',  # Wichtig für Excel-Export
        'pandas',
        'numpy',
        'scikit_posthocs',  # Für Dunn-Test
        'pingouin',  # Für Welch-ANOVA
        'matplotlib.ticker',  # Für ScalarFormatter
        'statsmodels.formula.api',  # Für ols (Two-Way ANOVA)
        'statsmodels.api',  # Für sm.stats.anova_lm
        'networkx'  # Add this line for network graph visualization
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StatisticalAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # True for debugging, can be False for final release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Institutslogo.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StatisticalAnalyzer',
)
app = BUNDLE(
    coll,
    name='StatisticalAnalyzer.app',
    icon='Institutslogo.ico',
    bundle_identifier=None,
)