#!/usr/bin/env python3
"""
Final verification test for BioMedStatX GUI improvements
Tests all implemented features to ensure they work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_color_picker_flexibility():
    """Test that the ColorButton allows any color selection"""
    print("Testing ColorButton flexibility...")
    
    try:
        from plot_aesthetics_dialog import ColorButton
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QColor
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Test ColorButton with custom color
        button = ColorButton("#FF0000")
        assert button.get_color() == "#FF0000", "Initial color not set correctly"
        
        # Test setting a different color
        button.set_color("#00FF00")
        assert button.get_color() == "#00FF00", "Color not updated correctly"
        
        print("✓ ColorButton allows any color selection")
        return True
    except Exception as e:
        print(f"✗ ColorButton test failed: {e}")
        return False

def test_automatic_figure_sizing():
    """Test the automatic figure size adaptation"""
    print("Testing automatic figure size adaptation...")
    
    try:
        from stats_functions import DataVisualizer
        
        # Test with 2 groups (should not change size)
        groups_2 = ['A', 'B']
        width, height = DataVisualizer._auto_adjust_figure_size(8, 6, groups_2, 'Bar')
        assert width == 8 and height == 6, "Size should not change for 2 groups"
        
        # Test with 5 groups
        groups_5 = ['A', 'B', 'C', 'D', 'E']
        width, height = DataVisualizer._auto_adjust_figure_size(8, 6, groups_5, 'Bar')
        expected_width = max(8, 6 + 5 * 1.0)  # 11
        assert width == expected_width, f"Expected width {expected_width}, got {width}"
        
        # Test with 10 groups (should add extra scaling)
        groups_10 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        width, height = DataVisualizer._auto_adjust_figure_size(8, 6, groups_10, 'Bar')
        expected_width = max(8, 6 + 10 * 1.0) + (10 - 6) * 0.5  # 16 + 2 = 18
        assert width == expected_width, f"Expected width {expected_width}, got {width}"
        
        # Test Raincloud plot (height scaling)
        width, height = DataVisualizer._auto_adjust_figure_size(8, 6, groups_10, 'Raincloud')
        expected_height = max(6, 4 + 10 * 1.2) + (10 - 6) * 0.8  # 16 + 3.2 = 19.2
        assert height == expected_height, f"Expected height {expected_height}, got {height}"
        
        print("✓ Automatic figure sizing works correctly")
        return True
    except Exception as e:
        print(f"✗ Figure sizing test failed: {e}")
        return False

def test_grayscale_defaults():
    """Test that default colors are grayscale"""
    print("Testing grayscale default colors...")
    
    try:
        from plot_aesthetics_dialog import PlotAestheticsDialog
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create dialog with test groups
        groups = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
        samples = {group: [1, 2, 3] for group in groups}
        
        dialog = PlotAestheticsDialog(groups, samples)
        config = dialog.get_config()
        
        # Check that colors are present and are grayscale
        assert 'colors' in config, "Colors not found in config"
        
        grayscale_colors = [
            '#2C2C2C', '#4A4A4A', '#686868', '#868686', '#A4A4A4', '#C2C2C2',
            '#3C3C3C', '#5A5A5A', '#787878', '#969696'
        ]
        
        for group in groups:
            color = config['colors'].get(group)
            assert color in grayscale_colors, f"Color {color} for group {group} is not grayscale"
        
        print("✓ Default colors are grayscale")
        return True
    except Exception as e:
        print(f"✗ Grayscale defaults test failed: {e}")
        return False

def test_group_support():
    """Test support for up to 10 groups"""
    print("Testing support for 10 groups...")
    
    try:
        from plot_aesthetics_dialog import PlotAestheticsDialog
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create dialog with 10 groups
        groups = [f'Group{i+1}' for i in range(10)]
        samples = {group: [1, 2, 3, 4, 5] for group in groups}
        
        dialog = PlotAestheticsDialog(groups, samples)
        config = dialog.get_config()
        
        # Check that all 10 groups have colors
        assert len(config['colors']) == 10, f"Expected 10 colors, got {len(config['colors'])}"
        
        # Check that figure size is adapted for 10 groups
        assert config['width'] > 8, "Width should be increased for 10 groups"
        
        print("✓ Support for 10 groups works correctly")
        return True
    except Exception as e:
        print(f"✗ 10 groups support test failed: {e}")
        return False

def test_window_size():
    """Test that the dialog window is properly sized"""
    print("Testing dialog window size...")
    
    try:
        from plot_aesthetics_dialog import PlotAestheticsDialog
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        groups = ['Group1', 'Group2']
        samples = {group: [1, 2, 3] for group in groups}
        
        dialog = PlotAestheticsDialog(groups, samples)
        
        # Check window size
        size = dialog.size()
        assert size.width() == 1600, f"Expected width 1600, got {size.width()}"
        assert size.height() == 900, f"Expected height 900, got {size.height()}"
        
        print("✓ Dialog window size is correct (1600x900)")
        return True
    except Exception as e:
        print(f"✗ Window size test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=== BioMedStatX Final Verification Tests ===\n")
    
    tests = [
        test_color_picker_flexibility,
        test_automatic_figure_sizing,
        test_grayscale_defaults,
        test_group_support,
        test_window_size
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The BioMedStatX GUI improvements are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
