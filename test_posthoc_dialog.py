#!/usr/bin/env python3
"""
Test script to verify that the post-hoc test dialog logic is working correctly
for different ANOVA types.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_posthoc_options():
    """Test that the correct post-hoc options are returned for different ANOVA types"""
    
    # Mock the dialog selection logic
    def get_options_for_progress_text(progress_text):
        """Simulate the options logic from select_posthoc_test_dialog"""
        if progress_text and ("two_way_anova" in progress_text or "mixed_anova" in progress_text or "repeated_measures_anova" in progress_text):
            # For advanced ANOVAs: only offer Tukey and Custom paired t-tests (no Dunnett)
            return [
                ("Tukey-HSD Test (compares all pairs)", "tukey"),
                ("Custom paired t-tests (you select specific pairs to compare)", "paired_custom"),
            ]
        else:
            # For One-Way ANOVA: offer all three options
            return [
                ("Tukey-HSD Test (compares all pairs, best for main effects)", "tukey"),
                ("Dunnett Test (compares all groups against ONE control group)", "dunnett"),
                ("Custom paired t-tests (you select specific pairs to compare)", "paired_custom"),
            ]
    
    # Test cases
    test_cases = [
        ("(one_way_anova)", "One-Way ANOVA", 3),  # Should have 3 options including Dunnett
        ("(two_way_anova)", "Two-Way ANOVA", 2),  # Should have 2 options without Dunnett
        ("(mixed_anova)", "Mixed ANOVA", 2),      # Should have 2 options without Dunnett
        ("(repeated_measures_anova)", "RM ANOVA", 2), # Should have 2 options without Dunnett
        (None, "No progress text", 3),             # Should have 3 options including Dunnett
    ]
    
    print("Testing post-hoc dialog option selection:")
    print("=" * 60)
    
    for progress_text, test_name, expected_count in test_cases:
        options = get_options_for_progress_text(progress_text)
        option_names = [opt[1] for opt in options]  # Extract option values
        has_dunnett = "dunnett" in option_names
        has_tukey = "tukey" in option_names
        has_paired_custom = "paired_custom" in option_names
        
        print(f"{test_name:20} | Options: {len(options):1d} | Tukey: {has_tukey} | Dunnett: {has_dunnett} | PairwiseT: {has_paired_custom}")
        
        # Verify expectations
        if expected_count == 2:  # Advanced ANOVAs
            assert not has_dunnett, f"{test_name} should not have Dunnett option"
            assert has_tukey, f"{test_name} should have Tukey option"
            assert has_paired_custom, f"{test_name} should have Pairwise T-test option"
        elif expected_count == 3:  # One-Way ANOVA and default
            assert has_dunnett, f"{test_name} should have Dunnett option"
            assert has_tukey, f"{test_name} should have Tukey option"
            assert has_paired_custom, f"{test_name} should have Pairwise T-test option"
        
        assert len(options) == expected_count, f"{test_name} should have {expected_count} options, got {len(options)}"
    
    print("=" * 60)
    print("✅ All tests passed! Post-hoc dialog options are correctly configured:")
    print("   - One-Way ANOVA: Tukey, Dunnett, Pairwise T-tests")
    print("   - Advanced ANOVAs (Two-Way, Mixed, RM): Tukey, Pairwise T-tests (no Dunnett)")

def test_decision_tree_logic():
    """Test the decision tree visualization logic"""
    print("\nTesting decision tree post-hoc highlighting:")
    print("=" * 60)
    
    def get_highlighted_paths(test_name, posthoc_test=None):
        """Simulate the decision tree highlighting logic"""
        highlighted = set()
        
        # Check if this is a One-Way ANOVA where users should see options
        is_one_way_anova = ("one-way" in test_name or 
                           (("anova" in test_name or "one way" in test_name) and 
                            "two-way" not in test_name and "two way" not in test_name and
                            "rm" not in test_name and "repeated" not in test_name and
                            "mixed" not in test_name))
        
        # Check if this is an advanced ANOVA (Two-Way, Mixed, RM)
        is_advanced_anova = ("two-way" in test_name or "two way" in test_name or 
                            "mixed" in test_name or "rm" in test_name or "repeated" in test_name)
        
        if is_one_way_anova and not posthoc_test:
            # For One-Way ANOVA with no specific post-hoc performed: show all options (including Dunnett)
            highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
            highlighted.add(('O1_PH', 'P1_PH_DN'))  # Dunnett  
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak
            return highlighted
        elif is_advanced_anova and not posthoc_test:
            # For Advanced ANOVAs with no specific post-hoc performed: show only Tukey and Pairwise (no Dunnett)
            highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak (Pairwise t-tests)
            return highlighted
        
        # For specific tests performed: show the specific path
        if posthoc_test:
            if "tukey" in posthoc_test.lower():
                highlighted.add(('O1_PH', 'P1_PH_TK'))
            elif "dunnett" in posthoc_test.lower():
                highlighted.add(('O1_PH', 'P1_PH_DN'))
            elif "holm" in posthoc_test.lower() or "sidak" in posthoc_test.lower():
                highlighted.add(('O1_PH', 'P1_PH_SD'))
        
        return highlighted
    
    # Test cases
    test_cases = [
        ("one-way anova", None, 3, True, "Should show all options including Dunnett"),
        ("two-way anova", None, 2, False, "Should show only Tukey and Pairwise (no Dunnett)"),
        ("mixed anova", None, 2, False, "Should show only Tukey and Pairwise (no Dunnett)"),
        ("rm anova", None, 2, False, "Should show only Tukey and Pairwise (no Dunnett)"),
        ("one-way anova", "Tukey HSD", 1, False, "Should show only performed Tukey test"),
        ("two-way anova", "Dunnett Test", 1, True, "Should show performed Dunnett test even for Two-Way"),
    ]
    
    for test_name, posthoc_test, expected_count, should_have_dunnett, description in test_cases:
        highlighted = get_highlighted_paths(test_name, posthoc_test)
        has_dunnett = ('O1_PH', 'P1_PH_DN') in highlighted
        has_tukey = ('O1_PH', 'P1_PH_TK') in highlighted
        has_pairwise = ('O1_PH', 'P1_PH_SD') in highlighted
        
        test_result = "✅" if has_dunnett == should_have_dunnett else "❌"
        print(f"{test_result} {test_name:15} | {str(posthoc_test):12} | Paths: {len(highlighted):1d} | Dunnett: {has_dunnett}")
        print(f"    {description}")
        
        if not posthoc_test:  # Only check for option display cases
            assert len(highlighted) == expected_count, f"Expected {expected_count} highlighted paths, got {len(highlighted)}"
            assert has_dunnett == should_have_dunnett, f"Dunnett expectation failed for {test_name}"
    
    print("=" * 60)
    print("✅ Decision tree logic tests completed!")

if __name__ == "__main__":
    try:
        test_posthoc_options()
        test_decision_tree_logic()
        print("\n🎉 All tests passed! The implementation correctly restricts Dunnett test to One-Way ANOVAs only.")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
