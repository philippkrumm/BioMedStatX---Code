"""
Comprehensive Test Script for Plot Generation
=============================================

This script tests all plot types (Bar, Box, Violin, Raincloud) with various parameter combinations
to ensure robust plot generation and identify potential issues.

The script simulates user interactions and generates plots with:
- Different color schemes
- Various hatching patterns
- Different typography settings
- Multiple styling options
- Random parameter variations

Usage: python test_plot_generation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stats_functions import DataVisualizer
    print("✓ Successfully imported DataVisualizer")
except ImportError as e:
    print(f"✗ Failed to import DataVisualizer: {e}")
    sys.exit(1)

class PlotTestSuite:
    """Comprehensive test suite for plot generation"""
    
    def __init__(self, output_dir="test_plots_output"):
        """Initialize the test suite"""
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Test data configurations
        self.test_groups = ['Control', 'Treatment_A', 'Treatment_B', 'Treatment_C']
        self.small_groups = ['Group_1', 'Group_2']
        self.large_groups = ['A', 'B', 'C', 'D', 'E', 'F']
        
        # Generate test datasets
        self.datasets = self.generate_test_datasets()
        
        # Plot types to test
        self.plot_types = ['Bar', 'Box', 'Violin', 'Raincloud']
        
        # Color schemes for testing
        self.color_schemes = [
            # Vibrant colors
            {'Control': '#FF6B6B', 'Treatment_A': '#4ECDC4', 'Treatment_B': '#45B7D1', 'Treatment_C': '#96CEB4'},
            # Pastel colors
            {'Control': '#FFB3BA', 'Treatment_A': '#BAFFC9', 'Treatment_B': '#BAE1FF', 'Treatment_C': '#FFFFBA'},
            # Professional colors
            {'Control': '#2C3E50', 'Treatment_A': '#E74C3C', 'Treatment_B': '#3498DB', 'Treatment_C': '#27AE60'},
            # Monochrome
            {'Control': '#2C2C2C', 'Treatment_A': '#4A4A4A', 'Treatment_B': '#686868', 'Treatment_C': '#868686'},
        ]
        
        # Hatch patterns for testing
        self.hatch_schemes = [
            {'Control': '', 'Treatment_A': '///', 'Treatment_B': '\\\\\\', 'Treatment_C': '|||'},
            {'Control': '...', 'Treatment_A': '+++', 'Treatment_B': 'xxx', 'Treatment_C': '***'},
            {'Control': '', 'Treatment_A': '', 'Treatment_B': '', 'Treatment_C': ''},  # No hatches
        ]
        
        # Typography variations (use fonts available on Windows)
        self.font_families = ['Arial', 'Times New Roman', 'Calibri', 'Segoe UI']
        self.font_sizes = [
            {'title': 16, 'axis': 14, 'ticks': 12, 'legend': 10},
            {'title': 12, 'axis': 10, 'ticks': 9, 'legend': 8},
            {'title': 20, 'axis': 16, 'ticks': 14, 'legend': 12},
        ]
        
        # Style variations
        self.style_variations = [
            {'theme': 'default', 'grid': False, 'despine': True, 'alpha': 0.8},
            {'theme': 'seaborn', 'grid': True, 'despine': False, 'alpha': 0.7},
            {'theme': 'minimal', 'grid': False, 'despine': True, 'alpha': 0.9},
        ]
        
        # Results tracking
        self.test_results = []
        self.failed_tests = []
        
    def create_output_directory(self):
        """Create output directory for test plots"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}")
        else:
            print(f"✓ Using existing output directory: {self.output_dir}")
    
    def generate_test_datasets(self):
        """Generate various test datasets with different characteristics"""
        np.random.seed(42)  # For reproducible results
        
        datasets = {}
        
        # Dataset 1: Normal distribution with different means
        datasets['normal_different_means'] = {
            'Control': np.random.normal(10, 2, 50),
            'Treatment_A': np.random.normal(12, 2, 45),
            'Treatment_B': np.random.normal(15, 2, 48),
            'Treatment_C': np.random.normal(8, 2, 52)
        }
        
        # Dataset 2: Different variances
        datasets['different_variances'] = {
            'Control': np.random.normal(10, 1, 50),
            'Treatment_A': np.random.normal(10, 3, 45),
            'Treatment_B': np.random.normal(10, 0.5, 48),
            'Treatment_C': np.random.normal(10, 2, 52)
        }
        
        # Dataset 3: Skewed distributions
        datasets['skewed_data'] = {
            'Control': np.random.exponential(2, 50),
            'Treatment_A': np.random.lognormal(1, 0.5, 45),
            'Treatment_B': np.random.gamma(2, 2, 48),
            'Treatment_C': np.random.weibull(1.5, 52) * 5
        }
        
        # Dataset 4: Small sample sizes
        datasets['small_samples'] = {
            'Group_1': np.random.normal(10, 2, 8),
            'Group_2': np.random.normal(12, 3, 6)
        }
        
        # Dataset 5: Large number of groups
        datasets['many_groups'] = {
            'A': np.random.normal(10, 2, 30),
            'B': np.random.normal(11, 2, 32),
            'C': np.random.normal(9, 2, 28),
            'D': np.random.normal(13, 2, 35),
            'E': np.random.normal(8, 2, 25),
            'F': np.random.normal(14, 2, 40)
        }
        
        # Dataset 6: Extreme values
        datasets['with_outliers'] = {
            'Control': np.concatenate([np.random.normal(10, 2, 45), [25, 30]]),  # Add outliers
            'Treatment_A': np.concatenate([np.random.normal(12, 2, 40), [-5, 35]]),
            'Treatment_B': np.random.normal(15, 2, 48),
            'Treatment_C': np.random.normal(8, 2, 52)
        }
        
        print(f"✓ Generated {len(datasets)} test datasets")
        return datasets
    
    def get_random_config(self, groups, plot_type):
        """Generate a random configuration for testing"""
        # Randomly select color and hatch schemes
        colors = random.choice(self.color_schemes)
        hatches = random.choice(self.hatch_schemes)
        
        # Adjust for different group sets
        if set(groups) != set(self.test_groups):
            # Map colors to the actual groups
            color_values = list(colors.values())
            hatch_values = list(hatches.values())
            
            colors = {group: color_values[i % len(color_values)] for i, group in enumerate(groups)}
            hatches = {group: hatch_values[i % len(hatch_values)] for i, group in enumerate(groups)}
        
        # Random typography
        font_family = random.choice(self.font_families)
        font_config = random.choice(self.font_sizes)
        
        # Random style
        style = random.choice(self.style_variations)
        
        # Random plot-specific parameters
        config = {
            'plot_type': plot_type,
            'colors': colors,
            'hatches': hatches,
            'alpha': style['alpha'],
            'theme': style['theme'],
            'grid_style': 'major' if style['grid'] else 'none',
            'spine_style': 'minimal' if style['despine'] else 'box',
            'font_family': font_family,
            'title_size': font_config['title'],
            'x_label_size': font_config['axis'],
            'y_label_size': font_config['axis'],
            'tick_label_size': font_config['ticks'],
            'show_legend': random.choice([True, False]),
            'show_points': random.choice([True, False]),
            'point_size': random.randint(40, 120),
            'show_significance_letters': random.choice([True, False]),
            'width': random.choice([6, 8, 10, 12]),
            'height': random.choice([4, 6, 8]),
            'dpi': random.choice([150, 200, 300]),
            'title': f'{plot_type} Plot Test - {random.choice(["Experiment A", "Study B", "Analysis C"])}',
            'x_label': random.choice(['Groups', 'Conditions', 'Treatments', '']),
            'y_label': random.choice(['Values', 'Measurements', 'Response', 'Outcome']),
        }
        
        # Plot-specific parameters
        if plot_type == 'Bar':
            config.update({
                'show_error_bars': random.choice([True, False]),
                'error_type': random.choice(['sd', 'se']),
                'bar_edge_color': random.choice(['black', 'white', 'gray']),
                'bar_linewidth': random.uniform(0.5, 2.0),
            })
        elif plot_type == 'Box':
            config.update({
                'box_width': random.uniform(0.5, 0.9),
                'edge_color': random.choice(['black', 'gray', 'darkblue']),
                'edge_width': random.uniform(0.5, 1.5),
            })
        elif plot_type == 'Violin':
            config.update({
                'violin_width': random.uniform(0.6, 0.9),
                'edge_color': random.choice(['black', 'gray', 'darkred']),
                'edge_width': random.uniform(0.3, 1.0),
            })
        elif plot_type == 'Raincloud':
            config.update({
                'violin_width': random.uniform(0.6, 0.8),
                'box_width': random.uniform(0.1, 0.3),
                'point_offset': random.uniform(0.1, 0.3),
                'group_spacing': random.uniform(0.7, 1.0),
            })
        
        return config
    
    def test_single_plot(self, dataset_name, groups, samples, plot_type, variation_num):
        """Test a single plot configuration"""
        test_name = f"{dataset_name}_{plot_type}_v{variation_num}"
        print(f"  Testing: {test_name}")
        
        try:
            # Generate random configuration
            config = self.get_random_config(groups, plot_type)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(config['width'], config['height']))
            
            # Set random seed for consistent test results
            np.random.seed(42 + variation_num)
            
            # Call the plot function through the central dispatcher
            DataVisualizer.plot_from_config(ax, groups, samples, config)
            
            # Save the plot
            output_file = os.path.join(self.output_dir, f"{test_name}.png")
            fig.savefig(output_file, dpi=config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            # Record success
            self.test_results.append({
                'test_name': test_name,
                'status': 'SUCCESS',
                'dataset': dataset_name,
                'plot_type': plot_type,
                'variation': variation_num,
                'config': config,
                'output_file': output_file
            })
            
            print(f"    ✓ SUCCESS: {test_name}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ✗ FAILED: {test_name} - {error_msg}")
            
            # Record failure
            self.failed_tests.append({
                'test_name': test_name,
                'status': 'FAILED',
                'dataset': dataset_name,
                'plot_type': plot_type,
                'variation': variation_num,
                'error': error_msg,
                'config': config if 'config' in locals() else None
            })
            
            # Close any open figures
            plt.close('all')
            return False
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests for all plot types and configurations"""
        print("\n" + "="*60)
        print("STARTING COMPREHENSIVE PLOT GENERATION TESTS")
        print("="*60)
        
        start_time = time.time()
        total_tests = 0
        successful_tests = 0
        
        # Test each dataset with each plot type
        for dataset_name, samples in self.datasets.items():
            print(f"\n📊 Testing dataset: {dataset_name}")
            groups = list(samples.keys())
            
            # Test each plot type
            for plot_type in self.plot_types:
                print(f"  🎨 Plot type: {plot_type}")
                
                # Create 3 variations of each plot type
                for variation in range(1, 4):
                    total_tests += 1
                    success = self.test_single_plot(dataset_name, groups, samples, plot_type, variation)
                    if success:
                        successful_tests += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests run: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        # Report failures
        if self.failed_tests:
            print("\n❌ FAILED TESTS:")
            for failure in self.failed_tests:
                print(f"  - {failure['test_name']}: {failure['error']}")
        else:
            print("\n✅ ALL TESTS PASSED!")
        
        return successful_tests, len(self.failed_tests)
    
    def generate_test_report(self):
        """Generate a detailed test report"""
        report_file = os.path.join(self.output_dir, "test_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("PLOT GENERATION TEST REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total tests: {len(self.test_results) + len(self.failed_tests)}\n")
            f.write(f"Successful: {len(self.test_results)}\n")
            f.write(f"Failed: {len(self.failed_tests)}\n\n")
            
            # Successful tests by plot type
            f.write("SUCCESS BREAKDOWN BY PLOT TYPE:\n")
            f.write("-" * 30 + "\n")
            for plot_type in self.plot_types:
                count = len([t for t in self.test_results if t['plot_type'] == plot_type])
                f.write(f"{plot_type}: {count} successful tests\n")
            f.write("\n")
            
            # Failed tests
            if self.failed_tests:
                f.write("FAILED TESTS:\n")
                f.write("-" * 15 + "\n")
                for failure in self.failed_tests:
                    f.write(f"Test: {failure['test_name']}\n")
                    f.write(f"Error: {failure['error']}\n")
                    f.write(f"Dataset: {failure['dataset']}\n")
                    f.write(f"Plot Type: {failure['plot_type']}\n\n")
            
            # Configuration examples
            f.write("EXAMPLE CONFIGURATIONS:\n")
            f.write("-" * 25 + "\n")
            for i, test in enumerate(self.test_results[:3]):  # Show first 3 successful configs
                f.write(f"Example {i+1} ({test['test_name']}):\n")
                for key, value in test['config'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"📄 Test report saved to: {report_file}")
    
    def test_edge_cases(self):
        """Test edge cases and potential problem scenarios"""
        print("\n🔍 Testing edge cases...")
        
        edge_cases = [
            # Empty groups (should be handled gracefully)
            {
                'name': 'empty_group',
                'groups': ['A', 'B'],
                'samples': {'A': [], 'B': [1, 2, 3]}
            },
            # Single value groups
            {
                'name': 'single_values',
                'groups': ['A', 'B'],
                'samples': {'A': [5], 'B': [7]}
            },
            # Very large values
            {
                'name': 'large_values',
                'groups': ['A', 'B'],
                'samples': {'A': [1e6, 1.1e6, 1.2e6], 'B': [2e6, 2.1e6, 2.2e6]}
            },
            # Very small values
            {
                'name': 'small_values',
                'groups': ['A', 'B'],
                'samples': {'A': [1e-6, 1.1e-6, 1.2e-6], 'B': [2e-6, 2.1e-6, 2.2e-6]}
            },
        ]
        
        edge_case_results = []
        
        for case in edge_cases:
            for plot_type in self.plot_types:
                try:
                    config = self.get_random_config(case['groups'], plot_type)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    DataVisualizer.plot_from_config(ax, case['groups'], case['samples'], config)
                    
                    output_file = os.path.join(self.output_dir, f"edge_case_{case['name']}_{plot_type}.png")
                    fig.savefig(output_file, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    
                    edge_case_results.append(f"✓ {case['name']}_{plot_type}")
                    
                except Exception as e:
                    edge_case_results.append(f"✗ {case['name']}_{plot_type}: {str(e)}")
                    plt.close('all')
        
        print(f"Edge case results ({len(edge_case_results)} tests):")
        for result in edge_case_results:
            print(f"  {result}")


def main():
    """Main function to run the comprehensive plot tests"""
    print("🚀 Starting comprehensive plot generation test suite...")
    
    # Create test suite
    test_suite = PlotTestSuite()
    
    # Run comprehensive tests
    successful, failed = test_suite.run_comprehensive_tests()
    
    # Test edge cases
    test_suite.test_edge_cases()
    
    # Generate detailed report
    test_suite.generate_test_report()
    
    print(f"\n🎯 Final Results:")
    print(f"   ✅ {successful} tests passed")
    print(f"   ❌ {failed} tests failed")
    print(f"   📁 Output: {os.path.abspath(test_suite.output_dir)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Plot generation system is working correctly.")
    else:
        print(f"\n⚠️  {failed} tests failed. Check the output and error messages above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
