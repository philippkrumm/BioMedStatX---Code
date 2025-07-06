"""
Performance Benchmark Script for BioMedStatX
Tests startup time, memory usage, and operation performance
"""

import time
import sys
import os
import gc
import psutil
from contextlib import contextmanager
import pandas as pd
import numpy as np

@contextmanager
def timer(operation_name):
    """Context manager to time operations"""
    start_time = time.time()
    start_memory = get_memory_usage()
    print(f"\n[BENCHMARK] Starting {operation_name}...")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = get_memory_usage()
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print(f"[BENCHMARK] {operation_name} completed:")
        print(f"  ⏱️  Time: {duration:.3f} seconds")
        print(f"  🧠 Memory: {end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def benchmark_imports():
    """Benchmark import times"""
    import_tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("PyQt5", "from PyQt5.QtWidgets import QApplication"),
        ("scipy.stats", "from scipy import stats"),
        ("seaborn", "import seaborn as sns"),
        ("pingouin", "import pingouin as pg"),
        ("lazy_imports", "from lazy_imports import get_pingouin, get_scipy_stats"),
    ]
    
    print("=" * 60)
    print("IMPORT PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    results = {}
    for name, import_code in import_tests:
        try:
            with timer(f"Import {name}"):
                exec(import_code)
                results[name] = "✅ Success"
        except ImportError as e:
            results[name] = f"❌ Failed: {e}"
        except Exception as e:
            results[name] = f"⚠️  Error: {e}"
        
        # Force garbage collection between imports
        gc.collect()
        time.sleep(0.1)
    
    print(f"\n{'Module':<15} {'Status':<30}")
    print("-" * 45)
    for module, status in results.items():
        print(f"{module:<15} {status}")

def benchmark_data_operations():
    """Benchmark common data operations"""
    print("\n" + "=" * 60)
    print("DATA OPERATIONS BENCHMARK")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\n--- Testing with {size} rows ---")
        
        with timer(f"Create DataFrame ({size} rows)"):
            df = pd.DataFrame({
                'group': np.random.choice(['A', 'B', 'C', 'D'], size),
                'value': np.random.normal(100, 15, size),
                'condition': np.random.choice(['Control', 'Treatment'], size)
            })
        
        with timer(f"Basic statistics ({size} rows)"):
            stats = df.groupby('group')['value'].agg(['mean', 'std', 'count'])
        
        with timer(f"Copy DataFrame ({size} rows)"):
            df_copy = df.copy()
            del df_copy
        
        with timer(f"Memory cleanup ({size} rows)"):
            del df
            gc.collect()

def benchmark_plotting():
    """Benchmark plotting operations"""
    print("\n" + "=" * 60)
    print("PLOTTING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create test data
        np.random.seed(42)
        data = {
            'A': np.random.normal(100, 15, 50),
            'B': np.random.normal(105, 12, 50),
            'C': np.random.normal(95, 18, 50)
        }
        
        plot_types = ['bar', 'box', 'violin']
        
        for plot_type in plot_types:
            with timer(f"Create {plot_type} plot"):
                fig, ax = plt.subplots(figsize=(8, 6))
                
                if plot_type == 'bar':
                    means = [np.mean(values) for values in data.values()]
                    ax.bar(data.keys(), means)
                elif plot_type == 'box':
                    ax.boxplot(data.values(), labels=data.keys())
                elif plot_type == 'violin':
                    try:
                        ax.violinplot(data.values())
                        ax.set_xticks(range(1, len(data) + 1))
                        ax.set_xticklabels(data.keys())
                    except Exception as e:
                        print(f"    ⚠️  Violin plot failed: {e}")
                
                plt.close(fig)
                gc.collect()
                
    except ImportError:
        print("❌ Matplotlib not available for plotting benchmark")

def benchmark_application_startup():
    """Benchmark application startup simulation"""
    print("\n" + "=" * 60)
    print("APPLICATION STARTUP SIMULATION")
    print("=" * 60)
    
    startup_steps = [
        ("Load core modules", lambda: exec("import os, sys, time")),
        ("Load PyQt5", lambda: exec("from PyQt5.QtWidgets import QApplication")),
        ("Load scientific stack", lambda: exec("import numpy as np; import pandas as pd")),
        ("Load plotting", lambda: exec("import matplotlib.pyplot as plt")),
        ("Load statistics", lambda: exec("from scipy import stats")),
        ("Load optional modules", lambda: exec("import seaborn as sns; import pingouin as pg")),
    ]
    
    total_time = 0
    for step_name, step_func in startup_steps:
        try:
            with timer(step_name):
                step_func()
                step_time = time.time()
            gc.collect()
        except Exception as e:
            print(f"    ⚠️  {step_name} failed: {e}")

def run_comprehensive_benchmark():
    """Run all benchmarks"""
    print("🚀 BioMedStatX Performance Benchmark Suite")
    print(f"Python {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Process ID: {os.getpid()}")
    
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        benchmark_imports()
        benchmark_data_operations() 
        benchmark_plotting()
        benchmark_application_startup()
        
    finally:
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        memory_delta = final_memory - initial_memory
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total benchmark time: {total_time:.3f} seconds")
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory delta: {memory_delta:+.1f}MB")
        print(f"Peak memory: {get_memory_usage():.1f}MB")

if __name__ == "__main__":
    run_comprehensive_benchmark()
