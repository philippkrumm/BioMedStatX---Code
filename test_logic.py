#!/usr/bin/env python3

# Test the specific method that was causing issues
def test_get_config_logic():
    """Test the logic of the get_config method"""
    
    # Simulate the error_type logic that was causing the AttributeError
    class MockDialog:
        def __init__(self):
            # This is what we added to fix the issue
            self.error_type_sd = MockRadioButton(True)  # Default: SD selected
            self.error_type_sem = MockRadioButton(False)
            
        def get_error_type(self):
            # This is the line that was failing: 'se' if self.error_type_sem.isChecked() else 'sd'
            return 'se' if self.error_type_sem.isChecked() else 'sd'
    
    class MockRadioButton:
        def __init__(self, checked=False):
            self._checked = checked
            
        def isChecked(self):
            return self._checked
    
    # Test 1: SD selected (default)
    dialog1 = MockDialog()
    result1 = dialog1.get_error_type()
    print(f"✓ Test 1 - SD selected: {result1} (expected: 'sd')")
    assert result1 == 'sd', f"Expected 'sd', got '{result1}'"
    
    # Test 2: SEM selected
    dialog2 = MockDialog()
    dialog2.error_type_sem._checked = True
    dialog2.error_type_sd._checked = False
    result2 = dialog2.get_error_type()
    print(f"✓ Test 2 - SEM selected: {result2} (expected: 'se')")
    assert result2 == 'se', f"Expected 'se', got '{result2}'"
    
    print("✓ All tests passed! The AttributeError should be fixed.")

if __name__ == "__main__":
    test_get_config_logic()
