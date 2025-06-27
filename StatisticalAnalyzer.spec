# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

this_dir = os.getcwd()
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
        'xlsxwriter',
        'pandas',
        'numpy',
        'scikit_posthocs',
        'pingouin',
        'matplotlib.ticker',
        'statsmodels.formula.api',
        'statsmodels.api',
        'networkx'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,  # <--- Das ist der entscheidende Fix!
    name='StatisticalAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # False, wenn kein Terminalfenster erscheinen soll
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Institutslogo.ico',
    onefile=True
)