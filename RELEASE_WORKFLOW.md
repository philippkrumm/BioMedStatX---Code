# Release Workflow für BioMedStatX

## Vorbereitung eines neuen Releases

### 1. Version Update
1. Öffnen Sie `src/updater.py`
2. Aktualisieren Sie die `CURRENT_VERSION` Variable:
   ```python
   CURRENT_VERSION = "1.0.2"  # Neue Version
   ```

3. Aktualisieren Sie den Titel in `src/statistical_analyzer.py`:
   ```python
   self.setWindowTitle("BioMedStatX v1.0.2 - Comprehensive Statistical Analysis Tool")
   ```

### 2. PyInstaller Build für verschiedene Plattformen

#### Windows (.exe)
```bash
# In der Hauptordner
pyinstaller --onefile --windowed --name BioMedStatX-v1.0.2-Windows --icon=assets/Institutslogo.ico src/statistical_analyzer.py
```

#### macOS (.app + .dmg) 
```bash
# .app erstellen
pyinstaller --onefile --windowed --name BioMedStatX-v1.0.2-macOS --icon=assets/Institutslogo.ico src/statistical_analyzer.py

# .dmg erstellen (optional, mit create-dmg)
create-dmg --volname "BioMedStatX v1.0.2" --window-pos 200 120 --window-size 800 400 --icon-size 100 --app-drop-link 600 185 "BioMedStatX-v1.0.2-macOS.dmg" "dist/BioMedStatX-v1.0.2-macOS.app"
```

#### Linux (.AppImage)
```bash
# AppImage erstellen (erfordert AppImage-Tools)
pyinstaller --onefile --name BioMedStatX-v1.0.2-Linux src/statistical_analyzer.py
# Dann mit appimagetool zu .AppImage konvertieren
```

### 3. GitHub Release erstellen

1. **Tag erstellen:**
   ```bash
   git tag -a v1.0.2 -m "Release version 1.0.2 - Added auto-updater functionality"
   git push origin v1.0.2
   ```

2. **GitHub Release Page:**
   - Gehen Sie zu: https://github.com/philippkrumm/BioMedStatX---Code/releases/new
   - Tag: `v1.0.2`
   - Title: `BioMedStatX v1.0.2`
   - Description: 
     ```markdown
     ## What's New in v1.0.2
     
     ### ✨ New Features
     - **Auto-Update System**: The application now automatically checks for updates on startup
     - **Manual Update Check**: Check for updates anytime via Help → Check for Updates...
     - **Cross-Platform Support**: Automatic installer detection for Windows, macOS, and Linux
     
     ### 🐛 Bug Fixes
     - Fixed issue with...
     - Improved stability when...
     
     ### 📥 Download
     Choose the appropriate version for your operating system:
     - **Windows**: Download `BioMedStatX-v1.0.2-Windows.exe`
     - **macOS**: Download `BioMedStatX-v1.0.2-macOS.dmg`  
     - **Linux**: Download `BioMedStatX-v1.0.2-Linux.AppImage`
     
     ### 🔧 Installation
     1. Download the appropriate file for your system
     2. Run the installer/executable
     3. The application will automatically check for future updates
     ```

3. **Assets hochladen:**
   - `BioMedStatX-v1.0.2-Windows.exe`
   - `BioMedStatX-v1.0.2-macOS.dmg` (oder .pkg)
   - `BioMedStatX-v1.0.2-Linux.AppImage`

## 4. Testing des Update-Systems

### Lokales Testing:
1. Bauen Sie eine Version mit einer älteren Versionsnummer (z.B. 1.0.0)
2. Erstellen Sie ein Test-Release auf GitHub
3. Starten Sie die alte Version und prüfen Sie, ob das Update erkannt wird

### Test-Script:
```python
# test_updater.py
import sys
sys.path.append('src')
from updater import UpdateChecker

checker = UpdateChecker(current_version="1.0.0")  # Test mit alter Version
checker.update_available.connect(lambda info: print(f"Update gefunden: {info}"))
checker.no_update.connect(lambda: print("Kein Update verfügbar"))
checker.error_occurred.connect(lambda err: print(f"Fehler: {err}"))
checker.start()
```

## 5. Automatisierung (Optional)

### GitHub Actions für automatische Builds:
```yaml
# .github/workflows/release.yml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, macos-latest, ubuntu-latest]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r docs/requirements_production.txt
        pip install pyinstaller
    
    - name: Build executable
      run: |
        pyinstaller --onefile --windowed --name BioMedStatX-${{ github.ref_name }}-${{ matrix.os }} --icon=assets/Institutslogo.ico src/statistical_analyzer.py
    
    - name: Upload Release Asset
      uses: actions/upload-release-asset@v1
      with:
        upload_url: ${{ github.event.release.upload_url }}
        asset_path: ./dist/BioMedStatX-${{ github.ref_name }}-${{ matrix.os }}
        asset_name: BioMedStatX-${{ github.ref_name }}-${{ matrix.os }}
        asset_content_type: application/octet-stream
```

## 6. Rollback-Plan

Falls ein Update Probleme verursacht:
1. **Hotfix Release**: Erstellen Sie schnell eine Patch-Version (z.B. 1.0.3)
2. **Download-Links**: Stellen Sie sicher, dass die vorherige stabile Version verfügbar bleibt
3. **Kommunikation**: Informieren Sie Benutzer über bekannte Probleme

## 7. Best Practices

- **Semantic Versioning**: Verwenden Sie Major.Minor.Patch (z.B. 1.0.1)
- **Testing**: Testen Sie Updates immer auf allen Zielplattformen
- **Release Notes**: Dokumentieren Sie alle Änderungen ausführlich
- **Backwards Compatibility**: Stellen Sie sicher, dass Benutzerdaten kompatibel bleiben
- **Rollback**: Haben Sie immer einen Plan für den Fall, dass ein Update fehlschlägt

## Update-Benachrichtigungen für Benutzer

Das System informiert Benutzer über:
- ✅ Neue verfügbare Updates
- ✅ Download-Fortschritt
- ✅ Installationsanweisungen
- ✅ Release Notes
- ✅ Automatische/Stille Prüfung beim Start