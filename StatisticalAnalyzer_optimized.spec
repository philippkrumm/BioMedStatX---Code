# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Get absolute path to the script directory
script_dir = os.getcwd()  # Use current working directory instead of __file__

# Platform-specific icon handling with absolute paths
if sys.platform.startswith('win'):
    icon_file = os.path.join(script_dir, 'assets', 'Institutslogo.ico')
elif sys.platform.startswith('darwin'):
    icon_file = os.path.join(script_dir, 'assets', 'Institutslogo.ico')  # macOS can handle .ico files
else:
    icon_file = None  # Linux doesn't use icons in the same way

# Verify icon file exists
if icon_file and not os.path.exists(icon_file):
    print(f"WARNING: Icon file not found at {icon_file}")
    icon_file = None
else:
    print(f"Using icon file: {icon_file}")

a = Analysis(
    [os.path.join(script_dir, 'src', 'statistical_analyzer.py')],
    pathex=[script_dir],
    binaries=[],
    datas=[
        (os.path.join(script_dir, 'assets', 'StyleSheet.qss'), 'assets'),
        (os.path.join(script_dir, 'assets', 'Institutslogo.ico'), 'assets'),
        (os.path.join(script_dir, 'assets', 'StatisticalAnalyzer_Excel_Template.xlsx'), 'assets'),
        (os.path.join(script_dir, 'assets', 'HowToScreenshots'), 'assets/HowToScreenshots'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_qt5agg',
    'mpl_toolkits',
    'mpl_toolkits.axes_grid1',
    'mpl_toolkits.axes_grid1.inset_locator',
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
    console=True,  # Console window enabled for debugging
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