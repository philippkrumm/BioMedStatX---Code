# PyInstaller Build-Anleitung für BioMedStatX

Diese .spec-Datei ist plattformunabhängig konfiguriert und funktioniert sowohl auf macOS als auch auf Windows.

## Voraussetzungen

### Für alle Plattformen:
1. Python 3.9+ installiert
2. Virtuelles Environment erstellt und aktiviert
3. Alle Dependencies installiert:
```bash
pip install PyQt5 numpy pandas scipy matplotlib seaborn openpyxl statsmodels pingouin psutil xlsxwriter networkx scikit-posthocs pyinstaller
```

## Build-Befehle

### macOS / Linux:
```bash
cd /pfad/zum/BioMedStatX---Code
rm -rf dist build  # Alte Builds löschen (optional)
pyinstaller StatisticalAnalyzer_optimized.spec
```

### Windows:
```cmd
cd C:\pfad\zum\BioMedStatX---Code
rmdir /s dist build  # Alte Builds löschen (optional)
pyinstaller StatisticalAnalyzer_optimized.spec
```

## Ergebnis

Nach dem Build finden Sie die ausführbare Anwendung in:
- **Ordner:** `dist/BioMedStatX/`
- **Ausführbare Datei:** 
  - macOS/Linux: `BioMedStatX`
  - Windows: `BioMedStatX.exe`

## Besonderheiten der Konfiguration

### Plattformspezifische Icons:
- Die .spec-Datei erkennt automatisch die Plattform
- Windows und macOS verwenden `assets/Institutslogo.ico`
- Linux verwendet kein Icon (da unterschiedlich gehandhabt)

### Enthaltene Assets:
- `assets/StyleSheet.qss` - UI-Styling
- `assets/Institutslogo.ico` - Programm-Icon
- `assets/StatisticalAnalyzer_Excel_Template.xlsx` - Excel-Template
- `assets/HowToScreenshots/` - Kompletter Screenshots-Ordner

### Ausgeschlossene Module:
- Web-Frameworks (Flask, Django, etc.)
- Jupyter/IPython-Komponenten
- Alternative GUI-Frameworks (Tkinter, PySide, etc.)
- Development-Tools
- Externe Datenbank-Engines

### Hidden Imports:
Alle wichtigen Module sind explizit aufgeführt:
- Matplotlib backends
- Scipy/NumPy/Pandas
- Statsmodels
- Pingouin
- Scikit-posthocs
- NetworkX
- etc.

## Tipps für Windows

1. **Visual C++ Redistributable:** Windows-Benutzer benötigen eventuell die Microsoft Visual C++ Redistributable.

2. **Windows Defender:** Der erste Start kann langsam sein, da Windows Defender die Anwendung scannt.

3. **Pfade:** Verwenden Sie Backslashes (`\`) in Windows-Pfaden.

## Tipps für macOS

1. **Codesigning-Warnung:** Die Warnung bezüglich Code-Signing kann ignoriert werden (nur für App Store relevant).

2. **Gatekeeper:** Bei der ersten Ausführung möglicherweise Rechtsklick → "Öffnen" erforderlich.

## Größe optimieren

Die aktuelle Konfiguration ist bereits für Größe optimiert durch:
- Ausschluss nicht benötigter Module
- UPX deaktiviert (kann zu Problemen führen)
- Keine Debug-Symbole

Die finale Anwendung ist ca. 500MB - 1GB groß, was für eine wissenschaftliche Anwendung mit vielen Bibliotheken normal ist.
