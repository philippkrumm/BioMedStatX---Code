"""
Memory Management Utilities for BioMedStatX
Optimizes memory usage and prevents memory leaks in plotting operations
"""

import gc
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Optional, Any
import weakref

class MemoryManager:
    """Manages memory usage for plots and heavy operations"""
    
    def __init__(self):
        self._plot_figures = weakref.WeakSet()
        self._temp_objects = []
    
    def register_figure(self, fig):
        """Register a matplotlib figure for cleanup"""
        if fig is not None:
            self._plot_figures.add(fig)
    
    def cleanup_figures(self):
        """Clean up all registered matplotlib figures"""
        for fig in list(self._plot_figures):
            try:
                plt.close(fig)
            except:
                pass
        plt.close('all')  # Fallback cleanup
        gc.collect()
    
    def add_temp_object(self, obj):
        """Add temporary object for cleanup"""
        self._temp_objects.append(obj)
    
    def cleanup_temp_objects(self):
        """Clean up temporary objects"""
        self._temp_objects.clear()
        gc.collect()
    
    def cleanup_all(self):
        """Clean up everything"""
        self.cleanup_figures()
        self.cleanup_temp_objects()

# Global memory manager instance
memory_manager = MemoryManager()

@contextmanager
def managed_plot():
    """Context manager for memory-safe plotting"""
    initial_figures = set(plt.get_fignums())
    try:
        yield
    finally:
        # Close any new figures created during the context
        final_figures = set(plt.get_fignums())
        new_figures = final_figures - initial_figures
        for fig_num in new_figures:
            plt.close(fig_num)
        gc.collect()

@contextmanager
def optimized_dataframe_operation(df):
    """Context manager for memory-efficient DataFrame operations"""
    try:
        # Create a view instead of copy when possible
        yield df
    finally:
        # Force garbage collection after DataFrame operations
        gc.collect()

def efficient_copy(df, columns=None):
    """Create an efficient copy of DataFrame with only needed columns"""
    if columns is None:
        return df.copy()
    else:
        # Only copy needed columns to save memory
        return df[columns].copy()

def clear_matplotlib_cache():
    """Clear matplotlib's internal caches"""
    try:
        from matplotlib import font_manager
        font_manager._rebuild()
    except:
        pass
    
    try:
        plt.rcdefaults()
    except:
        pass

class PlotMemoryOptimizer:
    """Optimizes memory usage for plot generation"""
    
    @staticmethod
    def optimize_plot_params():
        """Set matplotlib parameters for better memory usage"""
        plt.rcParams.update({
            'figure.max_open_warning': 5,  # Warn about too many open figures
            'axes.unicode_minus': False,   # Disable unicode minus (saves memory)
            'font.size': 10,               # Reasonable default font size
            'figure.autolayout': False,    # Disable automatic layout (can be memory intensive)
        })
    
    @staticmethod
    def create_figure_efficiently(figsize=(10, 6), dpi=100):
        """Create a figure with memory-efficient settings"""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        memory_manager.register_figure(fig)
        return fig, ax
    
    @staticmethod 
    def save_and_close(fig, filepath=None, **save_kwargs):
        """Save figure and immediately close it to free memory"""
        try:
            if filepath:
                fig.savefig(filepath, **save_kwargs)
            return fig
        finally:
            plt.close(fig)
            gc.collect()

def get_memory_usage():
    """Get current memory usage information"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),       # Percentage of total system memory
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }

def log_memory_usage(operation_name="Operation"):
    """Log current memory usage"""
    try:
        usage = get_memory_usage()
        print(f"[MEMORY] {operation_name}: {usage['rss_mb']:.1f}MB RSS, "
              f"{usage['percent']:.1f}% of system memory")
    except Exception:
        pass  # Fail silently if psutil not available

# Decorator for memory monitoring
def monitor_memory(func):
    """Decorator to monitor memory usage of functions"""
    def wrapper(*args, **kwargs):
        log_memory_usage(f"Before {func.__name__}")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            log_memory_usage(f"After {func.__name__}")
            gc.collect()
    return wrapper
