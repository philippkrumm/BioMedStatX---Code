# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE

this_dir = os.getcwd()
pyqt5_plugins = collect_data_files('PyQt5', subdir='Qt5/plugins')

a = Analysis(
    ['statistical_analyzer.py'],
    pathex=[this_dir],
    binaries=[],
    datas=[
        (os.path.join(this_dir, 'stats_functions.py'), '.'),
        (os.path.join(this_dir, 'StyleSheet.qss'), '.'),
        (os.path.join(this_dir, 'decisiontreevisualizer.py'), '.'),
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
        'networkx',
    ],
    excludes=['PySide6'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,           # Include binaries in the executable
    a.datas,              # Include data files in the executable
    [],
    name='StatisticalAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Institutslogo.ico',
    onefile=True,
)