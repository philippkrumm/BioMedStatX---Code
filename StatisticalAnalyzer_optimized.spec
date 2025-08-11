# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Platform-specific icon handling
if sys.platform.startswith('win'):
    icon_file = 'assets/Institutslogo.ico'
elif sys.platform.startswith('darwin'):
    icon_file = 'assets/Institutslogo.ico'  # macOS can handle .ico files
else:
    icon_file = None  # Linux doesn't use icons in the same way

a = Analysis(
    ['src/statistical_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/StyleSheet.qss', 'assets'),
        ('assets/Institutslogo.ico', 'assets'),
        ('assets/StatisticalAnalyzer_Excel_Template.xlsx', 'assets'),
        ('assets/HowToScreenshots', 'assets/HowToScreenshots'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_agg',
        'scikit_posthocs',
        'networkx',
        'pingouin',
        'statsmodels',
        'statsmodels.stats.multitest',
        'statsmodels.stats.multicomp',
        'statsmodels.formula.api',
        'statsmodels.stats.anova',
        'seaborn',
        'xlsxwriter',
        'openpyxl',
        'scipy.stats',
        'numpy',
        'pandas',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Web-related (definitiv nicht gebraucht für Desktop-App)
        'tornado',
        'flask',
        'django',
        'bottle',
        'cherrypy',
        'twisted',
        
        # Jupyter/IPython (definitiv nicht gebraucht für standalone App)
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'jupyterlab',
        'notebook',
        'ipython_genutils',
        'ipykernel',
        'ipywidgets',
        
        # Dokumentation/Testing (nur die wirklich sicheren)
        'sphinx',
        'pytest',
        'nose',
        
        # Alternative GUI frameworks (wir nutzen PyQt5)
        'tkinter',
        'Tkinter',
        '_tkinter',
        'PySide2',
        'PySide6', 
        'PyQt6',
        'wxPython',
        
        # Compiler/Development tools (nur die wirklich sicheren)
        'wheel',
        'pip',
        
        # Database engines (nur die externen, nicht sqlite3)
        'pymongo',
        'redis',
        'psycopg2',
        'MySQLdb',
        
        # Note: distutils was removed from excludes as it's required by setuptools/PyInstaller
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='BioMedStatX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BioMedStatX',  # Changed from 'BioMedStatX_dist' for cleaner folder name
)