#!/usr/bin/env python3
"""
Simple test script to check if PlotConfigDialog works correctly
"""

import sys
from PyQt5.QtWidgets import QApplication

# Add the current directory to the path so we can import our modules
sys.path.append('.')

from statistical_analyzer import PlotConfigDialog

def test_plot_config_dialog():
    """Test if PlotConfigDialog can be created and get_config works"""
    app = QApplication([])
    
    try:
        # Create a dialog with test groups
        test_groups = ['Group1', 'Group2', 'Group3']
        dialog = PlotConfigDialog(test_groups)
        
        # Test if get_config works without errors
        config = dialog.get_config()
        
        print("✓ PlotConfigDialog created successfully")
        print("✓ get_config() executed without errors")
        print(f"✓ Config contains {len(config)} keys")
        print(f"✓ Groups in config: {config.get('groups', [])}")
        print(f"✓ Error type: {config.get('error_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        app.quit()

if __name__ == "__main__":
    success = test_plot_config_dialog()
    sys.exit(0 if success else 1)
