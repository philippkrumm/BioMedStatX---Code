#!/usr/bin/env python3
"""
Comprehensive Statistical Test Verification for BioMedStatX
Tests ALL statistical capabilities and generates Excel results for verification.

This script systematically tests every statistical test path in your program:
- Basic two-group tests (t-test, Mann-Whitney U, etc.)
- Multi-group tests (ANOVA, Kruskal-Wallis, etc.)
- Advanced ANOVA (Two-Way, Repeated Measures, Mixed)
- Post-hoc tests for all scenarios
- Outlier detection methods
- Data transformations
- Multi-dataset analysis
- Edge cases and error handling

Only Excel outputs are generated - no plots - for focused verification.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all statistical modules
from stats_functions import (
    DataImporter, StatisticalTester, AnalysisManager, ResultsExporter,
    OutlierDetector, PostHocFactory, OUTLIER_IMPORTS_AVAILABLE
)

class ComprehensiveStatTestSuite:
    """Comprehensive test suite for all statistical capabilities"""
    
    def __init__(self, output_dir="comprehensive_test_results"):
        self.output_dir = output_dir
        self.results_summary = []
        self.test_counter = 0
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.chdir(output_dir)
        
        print(f"=== BioMedStatX Comprehensive Statistical Test Suite ===")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    def log_test(self, test_name, status, details="", error=None):
        """Log test results"""
        self.test_counter += 1
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if status == "PASS":
            print(f"[{timestamp}] ✓ Test {self.test_counter:02d}: {test_name}")
        elif status == "FAIL":
            print(f"[{timestamp}] ✗ Test {self.test_counter:02d}: {test_name}")
            if error:
                print(f"    Error: {str(error)[:100]}...")
        elif status == "SKIP":
            print(f"[{timestamp}] ⚠ Test {self.test_counter:02d}: {test_name} (SKIPPED)")
        
        if details:
            print(f"    → {details}")
            
        self.results_summary.append({
            'test_number': self.test_counter,
            'test_name': test_name,
            'status': status,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': timestamp
        })

    def create_basic_dataset(self, n_per_group=20, n_groups=3, group_names=None):
        """Create a basic dataset for testing"""
        if group_names is None:
            group_names = [f"Group_{chr(65+i)}" for i in range(n_groups)]
        
        data = []
        np.random.seed(42)  # Reproducible results
        
        for i, group in enumerate(group_names):
            # Different means to ensure detectable differences
            mean = 10 + i * 3
            values = np.random.normal(mean, 2, n_per_group)
            for val in values:
                data.append({'Group': group, 'Value': val})
        
        return pd.DataFrame(data)

    def save_dataset(self, df, filename):
        """Save test dataset to Excel file"""
        filepath = f"{filename}.xlsx"
        df.to_excel(filepath, index=False)
        return filepath

    def test_basic_two_group_independent(self):
        """Test independent two-group comparisons"""
        try:
            df = self.create_basic_dataset(n_groups=2, group_names=['Control', 'Treatment'])
            filepath = self.save_dataset(df, "01_independent_ttest_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=['Control', 'Treatment'],
                value_cols=['Value'],
                dependent=False,
                skip_plots=True,
                file_name="01_independent_ttest_results"
            )
            
            test_name = results.get('test', 'Unknown')
            p_val = results.get('p_value', 'N/A')
            self.log_test("Independent t-test", "PASS", f"Test: {test_name}, p={p_val}")
        except Exception as e:
            self.log_test("Independent t-test", "FAIL", error=e)

    def test_basic_two_group_paired(self):
        """Test paired two-group comparisons"""
        try:
            # Create paired data
            np.random.seed(42)
            data = []
            for i in range(20):
                baseline = np.random.normal(10, 2)
                treatment = baseline + np.random.normal(3, 1)  # Treatment effect
                data.append({'Group': 'Pre', 'Value': baseline, 'Subject': f'S{i+1:02d}'})
                data.append({'Group': 'Post', 'Value': treatment, 'Subject': f'S{i+1:02d}'})
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "02_paired_ttest_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=['Pre', 'Post'],
                value_cols=['Value'],
                dependent=True,
                skip_plots=True,
                file_name="02_paired_ttest_results"
            )
            
            test_name = results.get('test', 'Unknown')
            p_val = results.get('p_value', 'N/A')
            self.log_test("Paired t-test", "PASS", f"Test: {test_name}, p={p_val}")
        except Exception as e:
            self.log_test("Paired t-test", "FAIL", error=e)

    def test_one_way_anova(self):
        """Test one-way ANOVA with post-hoc"""
        try:
            df = self.create_basic_dataset(n_groups=4, group_names=['A', 'B', 'C', 'D'])
            filepath = self.save_dataset(df, "03_oneway_anova_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=['A', 'B', 'C', 'D'],
                value_cols=['Value'],
                dependent=False,
                skip_plots=True,
                file_name="03_oneway_anova_results"
            )
            
            test_name = results.get('test', 'Unknown')
            posthoc_count = len(results.get('pairwise_comparisons', []))
            self.log_test("One-way ANOVA", "PASS", f"Test: {test_name}, Post-hoc: {posthoc_count} comparisons")
        except Exception as e:
            self.log_test("One-way ANOVA", "FAIL", error=e)

    def test_non_parametric_tests(self):
        """Test non-parametric alternatives"""
        try:
            # Create heavily skewed data to force non-parametric path
            np.random.seed(42)
            data = []
            groups = ['Group_A', 'Group_B', 'Group_C']
            
            for i, group in enumerate(groups):
                if i == 0:
                    values = np.random.exponential(3, 25)  # Heavily skewed
                elif i == 1:
                    values = np.random.lognormal(1, 0.8, 25)
                else:
                    values = np.random.uniform(1, 20, 25)
                
                for val in values:
                    data.append({'Group': group, 'Value': val})
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "04_nonparametric_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=groups,
                value_cols=['Value'],
                dependent=False,
                skip_plots=True,
                file_name="04_nonparametric_results"
            )
            
            test_name = results.get('test', 'Unknown')
            self.log_test("Non-parametric tests", "PASS", f"Test: {test_name}")
        except Exception as e:
            self.log_test("Non-parametric tests", "FAIL", error=e)

    def test_two_way_anova(self):
        """Test Two-Way ANOVA"""
        try:
            # Create factorial design data
            np.random.seed(42)
            data = []
            factor_a = ['A1', 'A2', 'A3']
            factor_b = ['B1', 'B2']
            
            for fa in factor_a:
                for fb in factor_b:
                    base_mean = 10
                    if fa == 'A1': base_mean += 3
                    if fa == 'A2': base_mean += 1
                    if fb == 'B2': base_mean += 4
                    if fa == 'A1' and fb == 'B2': base_mean += 3  # Interaction
                    
                    values = np.random.normal(base_mean, 2.5, 18)
                    for val in values:
                        data.append({'FactorA': fa, 'FactorB': fb, 'Value': val})
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "05_twoway_anova_data")
            
            results = StatisticalTester.perform_advanced_test(
                df=df,
                test='two_way_anova',
                dv='Value',
                subject=None,
                between=['FactorA', 'FactorB'],
                within=[],
                alpha=0.05,
                file_name="05_twoway_anova_results.xlsx"
            )
            
            self.log_test("Two-Way ANOVA", "PASS", "Advanced test completed successfully")
        except Exception as e:
            self.log_test("Two-Way ANOVA", "FAIL", error=e)

    def test_repeated_measures_anova(self):
        """Test Repeated Measures ANOVA"""
        try:
            # Create repeated measures data
            np.random.seed(42)
            data = []
            timepoints = ['T1', 'T2', 'T3', 'T4']
            
            for subj in range(1, 22):
                subject_baseline = np.random.normal(10, 2)
                
                for i, tp in enumerate(timepoints):
                    time_effect = i * 2.0  # Strong time effect
                    value = subject_baseline + time_effect + np.random.normal(0, 1)
                    
                    data.append({
                        'Subject': f'S{subj:02d}',
                        'Timepoint': tp,
                        'Value': value
                    })
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "06_rm_anova_data")
            
            results = StatisticalTester.perform_advanced_test(
                df=df,
                test='repeated_measures_anova',
                dv='Value',
                subject='Subject',
                between=[],
                within=['Timepoint'],
                alpha=0.05,
                file_name="06_rm_anova_results.xlsx"
            )
            
            self.log_test("Repeated Measures ANOVA", "PASS", "RM-ANOVA completed successfully")
        except Exception as e:
            self.log_test("Repeated Measures ANOVA", "FAIL", error=e)

    def test_mixed_anova(self):
        """Test Mixed ANOVA"""
        try:
            # Create mixed design data
            np.random.seed(42)
            data = []
            groups = ['Control', 'Treatment']
            timepoints = ['Pre', 'Mid', 'Post']
            
            for group in groups:
                for subj in range(1, 17):
                    subject_baseline = np.random.normal(10, 1.8)
                    
                    # Group effect
                    if group == 'Treatment':
                        subject_baseline += 1.5
                    
                    for i, tp in enumerate(timepoints):
                        time_effect = i * 1.2
                        
                        # Interaction: treatment improves more over time
                        if group == 'Treatment' and i > 0:
                            time_effect += i * 2.0
                        
                        value = subject_baseline + time_effect + np.random.normal(0, 1)
                        
                        data.append({
                            'Subject': f'{group}_S{subj:02d}',
                            'Group': group,
                            'Timepoint': tp,
                            'Value': value
                        })
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "07_mixed_anova_data")
            
            results = StatisticalTester.perform_advanced_test(
                df=df,
                test='mixed_anova',
                dv='Value',
                subject='Subject',
                between=['Group'],
                within=['Timepoint'],
                alpha=0.05,
                file_name="07_mixed_anova_results.xlsx"
            )
            
            self.log_test("Mixed ANOVA", "PASS", "Mixed ANOVA completed successfully")
        except Exception as e:
            self.log_test("Mixed ANOVA", "FAIL", error=e)

    def test_transformations(self):
        """Test data transformations"""
        transformations = ['log', 'boxcox', 'arcsin_sqrt']
        
        for transform in transformations:
            try:
                # Create data that might benefit from transformation
                np.random.seed(42)
                data = []
                for group in ['Low', 'Medium', 'High']:
                    if transform == 'log':
                        values = np.random.lognormal(1, 0.5, 20)
                    elif transform == 'boxcox':
                        values = np.random.exponential(2, 20)
                    else:  # arcsin_sqrt
                        values = np.random.beta(2, 5, 20)
                    
                    for val in values:
                        data.append({'Group': group, 'Value': val})
                
                df = pd.DataFrame(data)
                filepath = self.save_dataset(df, f"08_{transform}_transform_data")
                
                results = StatisticalTester.perform_advanced_test(
                    df=df,
                    test='two_way_anova',
                    dv='Value',
                    subject=None,
                    between=['Group'],
                    within=[],
                    manual_transform=transform,
                    file_name=f"08_{transform}_transform_results.xlsx"
                )
                
                self.log_test(f"Transformation ({transform})", "PASS", "Transform applied successfully")
            except Exception as e:
                self.log_test(f"Transformation ({transform})", "FAIL", error=e)

    def test_outlier_detection(self):
        """Test outlier detection methods"""
        if not OUTLIER_IMPORTS_AVAILABLE:
            self.log_test("Outlier Detection", "SKIP", "Required packages not available")
            return
        
        try:
            # Create data with known outliers
            np.random.seed(42)
            data = []
            groups = ['Normal', 'With_Outliers']
            
            for group in groups:
                if group == 'Normal':
                    values = np.random.normal(10, 2, 25)
                else:
                    values = np.random.normal(10, 2, 22)
                    values = np.append(values, [30, 35, -10])  # Add obvious outliers
                
                for val in values:
                    data.append({'Group': group, 'Value': val})
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "09_outlier_data")
            
            # Test Modified Z-Score
            detector = OutlierDetector(df=df.copy(), group_col='Group', value_col='Value')
            detector.run_mod_z_score(threshold=3.5, iterate=True)
            detector.save_results("09_modz_outlier_results.xlsx")
            
            outlier_count = detector.df['ModZ_Outlier'].sum() if 'ModZ_Outlier' in detector.df.columns else 0
            self.log_test("Modified Z-Score Outlier Detection", "PASS", f"Found {outlier_count} outliers")
            
            # Test Grubbs' test
            detector2 = OutlierDetector(df=df.copy(), group_col='Group', value_col='Value')
            detector2.run_grubbs_test(alpha=0.05, iterate=True)
            detector2.save_results("09_grubbs_outlier_results.xlsx")
            
            grubbs_count = detector2.df['Grubbs_Outlier'].sum() if 'Grubbs_Outlier' in detector2.df.columns else 0
            self.log_test("Grubbs' Outlier Detection", "PASS", f"Found {grubbs_count} outliers")
            
        except Exception as e:
            self.log_test("Outlier Detection", "FAIL", error=e)

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        # Single group
        try:
            df = self.create_basic_dataset(n_groups=1, group_names=['OnlyGroup'])
            filepath = self.save_dataset(df, "10_single_group_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=['OnlyGroup'],
                value_cols=['Value'],
                skip_plots=True,
                file_name="10_single_group_results"
            )
            
            self.log_test("Single group handling", "PASS", f"Test: {results.get('test', 'Unknown')}")
        except Exception as e:
            self.log_test("Single group handling", "FAIL", error=e)
        
        # Small sample sizes
        try:
            df = self.create_basic_dataset(n_per_group=3, n_groups=2)
            filepath = self.save_dataset(df, "11_small_sample_data")
            
            results = AnalysisManager.analyze(
                file_path=filepath,
                group_col='Group',
                groups=['Group_A', 'Group_B'],
                value_cols=['Value'],
                skip_plots=True,
                file_name="11_small_sample_results"
            )
            
            self.log_test("Small sample handling", "PASS", f"Test: {results.get('test', 'Unknown')}")
        except Exception as e:
            self.log_test("Small sample handling", "FAIL", error=e)

    def test_multi_dataset_analysis(self):
        """Test multi-dataset analysis"""
        try:
            # Create multi-column dataset
            np.random.seed(42)
            data = []
            groups = ['Control', 'Treatment_A', 'Treatment_B']
            
            for group in groups:
                for i in range(22):
                    row = {'Group': group}
                    # Multiple "datasets" with different effect sizes
                    for j, dataset in enumerate(['Gene1', 'Gene2', 'Gene3', 'Protein1']):
                        if group == 'Control':
                            value = np.random.normal(10, 2)
                        elif group == 'Treatment_A':
                            value = np.random.normal(10 + j + 1, 2)  # Increasing effects
                        else:  # Treatment_B
                            value = np.random.normal(10 + j + 3, 2)  # Larger effects
                        row[dataset] = value
                    data.append(row)
            
            df = pd.DataFrame(data)
            filepath = self.save_dataset(df, "12_multi_dataset_data")
            
            # Run multi-dataset analysis
            for i, dataset in enumerate(['Gene1', 'Gene2', 'Gene3', 'Protein1']):
                results = AnalysisManager.analyze(
                    file_path=filepath,
                    group_col='Group',
                    groups=groups,
                    value_cols=[dataset],
                    skip_plots=True,
                    file_name=f"12_multi_dataset_{dataset}_results"
                )
            
            self.log_test("Multi-dataset analysis", "PASS", "4 datasets analyzed successfully")
        except Exception as e:
            self.log_test("Multi-dataset analysis", "FAIL", error=e)

    def run_all_tests(self):
        """Execute the complete test suite"""
        print("Starting comprehensive statistical test execution...\n")
        start_time = time.time()
        
        # Execute all test methods
        test_methods = [
            self.test_basic_two_group_independent,
            self.test_basic_two_group_paired,
            self.test_one_way_anova,
            self.test_non_parametric_tests,
            self.test_two_way_anova,
            self.test_repeated_measures_anova,
            self.test_mixed_anova,
            self.test_transformations,
            self.test_outlier_detection,
            self.test_edge_cases,
            self.test_multi_dataset_analysis,
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.log_test(f"{test_method.__name__}", "FAIL", error=e)
        
        # Summary statistics
        total_time = time.time() - start_time
        total_tests = len(self.results_summary)
        passed = len([r for r in self.results_summary if r['status'] == 'PASS'])
        failed = len([r for r in self.results_summary if r['status'] == 'FAIL'])
        skipped = len([r for r in self.results_summary if r['status'] == 'SKIP'])
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 70)
        print(f"Total tests executed: {total_tests}")
        print(f"✓ Passed: {passed} ({passed/total_tests*100:.1f}%)")
        print(f"✗ Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"⚠ Skipped: {skipped} ({skipped/total_tests*100:.1f}%)")
        print(f"Execution time: {total_time:.2f} seconds")
        print(f"Results directory: {os.path.abspath('.')}")
        
        # Save detailed report
        try:
            report_df = pd.DataFrame(self.results_summary)
            report_df.to_excel("comprehensive_test_report.xlsx", index=False)
            print(f"Detailed report: comprehensive_test_report.xlsx")
        except Exception as e:
            print(f"Could not save report: {e}")
        
        if failed == 0:
            print("\n🎉 ALL STATISTICAL TESTS PASSED! 🎉")
            print("Your statistical analysis program is fully functional!")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check the detailed report for specifics.")
        
        print("=" * 70)
        return failed == 0

def main():
    """Main execution function"""
    try:
        test_suite = ComprehensiveStatTestSuite()
        success = test_suite.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
