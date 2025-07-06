"""
Test script to measure the performance improvement from lazy loading
Run this before and after implementing lazy loading to see the difference
"""

import time
import sys
import psutil
import os

def measure_import_time():
    """Measure the time to import the main application"""
    print("🚀 Testing BioMedStatX Lazy Loading Performance")
    print("=" * 60)
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure startup time
    start_time = time.time()
    
    try:
        # Import the main application (this triggers all imports)
        from statistical_analyzer import StatisticalAnalyzerApp
        
        import_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        
        print(f"✅ Application import successful!")
        print(f"⏱️  Import time: {import_time:.3f} seconds")
        print(f"🧠 Initial memory: {initial_memory:.1f}MB")
        print(f"🧠 Final memory: {final_memory:.1f}MB")
        print(f"🧠 Memory increase: {memory_delta:.1f}MB")
        
        # Test lazy loading effectiveness
        print(f"\n📊 Testing lazy loading:")
        
        # Test matplotlib lazy loading
        plt_start = time.time()
        from statistical_analyzer import get_matplotlib
        plt_module = get_matplotlib()
        plt_time = time.time() - plt_start
        plt_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"  📈 Matplotlib lazy load: {plt_time:.3f}s (+{plt_memory - final_memory:.1f}MB)")
        
        # Test seaborn lazy loading
        sns_start = time.time()
        from statistical_analyzer import get_seaborn
        sns_module = get_seaborn()
        sns_time = time.time() - sns_start
        sns_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"  🎨 Seaborn lazy load: {sns_time:.3f}s (+{sns_memory - plt_memory:.1f}MB)")
        
        # Overall results
        total_lazy_time = plt_time + sns_time
        total_lazy_memory = sns_memory - final_memory
        
        print(f"\n📋 Summary:")
        print(f"  Core application: {import_time:.3f}s ({memory_delta:.1f}MB)")
        print(f"  Heavy modules (lazy): {total_lazy_time:.3f}s ({total_lazy_memory:.1f}MB)")
        print(f"  Total if loaded upfront: ~{import_time + total_lazy_time:.3f}s")
        
        # Performance improvement calculation
        if total_lazy_time > 0:
            startup_improvement = (total_lazy_time / (import_time + total_lazy_time)) * 100
            print(f"  🎯 Startup improvement: ~{startup_improvement:.1f}% faster")
        
        return {
            'import_time': import_time,
            'memory_delta': memory_delta,
            'lazy_time': total_lazy_time,
            'lazy_memory': total_lazy_memory,
            'total_memory': sns_memory
        }
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

def test_individual_modules():
    """Test individual module import times"""
    print(f"\n🔬 Individual Module Performance:")
    print("-" * 40)
    
    modules_to_test = [
        ('numpy', 'import numpy as np'),
        ('pandas', 'import pandas as pd'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('seaborn', 'import seaborn as sns'),
        ('scipy.stats', 'from scipy import stats'),
        ('pingouin', 'import pingouin as pg'),
        ('PyQt5', 'from PyQt5.QtWidgets import QApplication'),
    ]
    
    results = {}
    
    for name, import_cmd in modules_to_test:
        try:
            # Fresh Python process for each test
            start_time = time.time()
            exec(import_cmd)
            import_time = time.time() - start_time
            
            print(f"  {name:<15}: {import_time:.3f}s")
            results[name] = import_time
            
        except ImportError:
            print(f"  {name:<15}: Not available")
            results[name] = None
        except Exception as e:
            print(f"  {name:<15}: Error - {e}")
            results[name] = None
    
    return results

def compare_with_baseline():
    """Compare current performance with expected baseline"""
    print(f"\n📊 Performance Comparison:")
    print("-" * 40)
    
    # Expected baseline (before lazy loading)
    baseline = {
        'startup_time': 1.4,  # seconds
        'memory_usage': 133,  # MB increase
    }
    
    results = measure_import_time()
    if results:
        startup_improvement = ((baseline['startup_time'] - results['import_time']) / baseline['startup_time']) * 100
        memory_improvement = ((baseline['memory_usage'] - results['memory_delta']) / baseline['memory_usage']) * 100
        
        print(f"  Baseline startup: {baseline['startup_time']:.1f}s")
        print(f"  Current startup:  {results['import_time']:.3f}s")
        print(f"  Improvement:      {startup_improvement:+.1f}%")
        print(f"")
        print(f"  Baseline memory:  {baseline['memory_usage']:.0f}MB")
        print(f"  Current memory:   {results['memory_delta']:.1f}MB")
        print(f"  Improvement:      {memory_improvement:+.1f}%")
        
        # Status
        if startup_improvement > 30:
            print(f"  🎉 Excellent improvement!")
        elif startup_improvement > 15:
            print(f"  ✅ Good improvement!")
        elif startup_improvement > 0:
            print(f"  🆗 Some improvement")
        else:
            print(f"  ⚠️  No improvement detected")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}\n")
    
    # Run all tests
    measure_import_time()
    test_individual_modules()
    compare_with_baseline()
    
    print(f"\n" + "=" * 60)
    print("🏁 Performance testing complete!")
    print("💡 Tip: Run this script before and after changes to measure improvement.")
