import psutil
import time
import threading
import sys
from typing import Dict, List, Optional

try:
    import GPUtil
    import nvidia_ml_py3 as nvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU monitoring libraries not available. Install nvidia-ml-py3 and GPUtil for GPU monitoring.")

class ResourceMonitor:
    def __init__(self, interval: float = 1.0):
        """
        Resource monitor for tracking CPU, memory, and GPU usage.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.cpu_count = psutil.cpu_count() or 1  # Handle None case
        self.data = {
            'cpu_cores': [],      # Changed from cpu_percent to cpu_cores
            'memory_percent': [],
            'memory_gb': [],      # Changed from memory_mb to memory_gb
            'timestamps': []
        }
        
        if GPU_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_count = nvml.nvmlDeviceGetCount()
                for i in range(self.gpu_count):
                    self.data[f'gpu_{i}_utilization'] = []
                    self.data[f'gpu_{i}_memory_used_gb'] = []      # Changed to GB
                    self.data[f'gpu_{i}_memory_total_gb'] = []     # Changed to GB
            except Exception as e:
                print(f"GPU monitoring initialization failed: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_cores_used = (cpu_percent / 100) * self.cpu_count  # Convert to cores
                memory = psutil.virtual_memory()
                
                self.data['cpu_cores'].append(cpu_cores_used)
                self.data['memory_percent'].append(memory.percent)
                self.data['memory_gb'].append(memory.used / 1024 / 1024 / 1024)  # Convert to GB
                self.data['timestamps'].append(time.time())
                
                # GPU monitoring
                if GPU_AVAILABLE and self.gpu_count > 0:
                    for i in range(self.gpu_count):
                        try:
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            
                            # GPU utilization
                            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                            self.data[f'gpu_{i}_utilization'].append(utilization.gpu)
                            
                            # GPU memory
                            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                            self.data[f'gpu_{i}_memory_used_gb'].append(memory_info.used / 1024 / 1024 / 1024)  # Convert to GB
                            self.data[f'gpu_{i}_memory_total_gb'].append(memory_info.total / 1024 / 1024 / 1024)  # Convert to GB
                            
                        except Exception as e:
                            print(f"Error monitoring GPU {i}: {e}")
                            self.data[f'gpu_{i}_utilization'].append(0)
                            self.data[f'gpu_{i}_memory_used_gb'].append(0)
                            self.data[f'gpu_{i}_memory_total_gb'].append(0)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print(f"Resource monitoring started (interval: {self.interval}s)")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        if self.monitoring:
            self.monitoring = False
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=2)
            print("Resource monitoring stopped")
    
    def get_current_usage(self) -> Dict:
        """Get current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_cores_used = (cpu_percent / 100) * self.cpu_count  # Convert to cores
            memory = psutil.virtual_memory()
            
            result = {
                'cpu_percent': cpu_percent,
                'cpu_cores': cpu_cores_used,    # Added cores metric
                'memory_percent': memory.percent,
                'memory_gb': memory.used / 1024 / 1024 / 1024,  # Convert to GB
                'timestamp': time.time()
            }
            
            if GPU_AVAILABLE and self.gpu_count > 0:
                for i in range(self.gpu_count):
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        result[f'gpu_{i}_utilization'] = utilization.gpu
                        result[f'gpu_{i}_memory_used_gb'] = memory_info.used / 1024 / 1024 / 1024  # Convert to GB
                        result[f'gpu_{i}_memory_total_gb'] = memory_info.total / 1024 / 1024 / 1024  # Convert to GB
                        result[f'gpu_{i}_memory_percent'] = (memory_info.used / memory_info.total) * 100
                        
                    except Exception as e:
                        print(f"Error getting GPU {i} usage: {e}")
                        result[f'gpu_{i}_utilization'] = 0
                        result[f'gpu_{i}_memory_used_gb'] = 0
                        result[f'gpu_{i}_memory_total_gb'] = 0
                        result[f'gpu_{i}_memory_percent'] = 0
            
            return result
        except Exception as e:
            print(f"Error getting current usage: {e}")
            return {}
    
    def get_summary(self) -> Dict:
        """Get monitoring summary statistics"""
        if not self.data['cpu_cores']:
            return {}
        
        summary = {
            'duration': len(self.data['cpu_cores']) * self.interval,
            'cpu_cores_avg': sum(self.data['cpu_cores']) / len(self.data['cpu_cores']),
            'cpu_cores_max': max(self.data['cpu_cores']),
            'memory_avg': sum(self.data['memory_percent']) / len(self.data['memory_percent']),
            'memory_max': max(self.data['memory_percent']),
            'memory_gb_avg': sum(self.data['memory_gb']) / len(self.data['memory_gb']),
            'memory_gb_max': max(self.data['memory_gb'])
        }
        
        if GPU_AVAILABLE and self.gpu_count > 0:
            for i in range(self.gpu_count):
                if f'gpu_{i}_utilization' in self.data and self.data[f'gpu_{i}_utilization']:
                    summary[f'gpu_{i}_util_avg'] = sum(self.data[f'gpu_{i}_utilization']) / len(self.data[f'gpu_{i}_utilization'])
                    summary[f'gpu_{i}_util_max'] = max(self.data[f'gpu_{i}_utilization'])
                    summary[f'gpu_{i}_mem_gb_avg'] = sum(self.data[f'gpu_{i}_memory_used_gb']) / len(self.data[f'gpu_{i}_memory_used_gb'])
                    summary[f'gpu_{i}_mem_gb_max'] = max(self.data[f'gpu_{i}_memory_used_gb'])
        
        return summary
    
    def print_summary(self):
        """Print monitoring summary"""
        summary = self.get_summary()
        if not summary:
            print("No monitoring data available")
            return
        
        print("\n" + "="*50)
        print("RESOURCE MONITORING SUMMARY")
        print("="*50)
        print(f"Duration: {summary['duration']:.1f} seconds")
        print(f"CPU Usage - Avg: {summary['cpu_cores_avg']:.2f} cores, Max: {summary['cpu_cores_max']:.2f} cores")
        print(f"Memory Usage - Avg: {summary['memory_avg']:.1f}%, Max: {summary['memory_max']:.1f}%")
        print(f"Memory Usage - Avg: {summary['memory_gb_avg']:.2f}GB, Max: {summary['memory_gb_max']:.2f}GB")
        
        if GPU_AVAILABLE and self.gpu_count > 0:
            print("\nGPU Usage:")
            for i in range(self.gpu_count):
                if f'gpu_{i}_util_avg' in summary:
                    print(f"  GPU {i} - Util Avg: {summary[f'gpu_{i}_util_avg']:.1f}%, Max: {summary[f'gpu_{i}_util_max']:.1f}%")
                    print(f"  GPU {i} - Memory Avg: {summary[f'gpu_{i}_mem_gb_avg']:.2f}GB, Max: {summary[f'gpu_{i}_mem_gb_max']:.2f}GB")
        
        print("="*50)
    
    def clear_data(self):
        """Clear monitoring data"""
        for key in self.data:
            self.data[key].clear()


def print_system_info():
    """Print system information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    # CPU info
    print(f"CPU Count: {psutil.cpu_count()} cores")
    print(f"CPU Count (Physical): {psutil.cpu_count(logical=False)} cores")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"Available Memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
    
    # GPU info
    if GPU_AVAILABLE:
        try:
            nvml.nvmlInit()
            gpu_count = nvml.nvmlDeviceGetCount()
            print(f"GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                print(f"GPU {i}: {name}")
                print(f"  Memory: {memory_info.total / 1024 / 1024 / 1024:.1f} GB")
                
        except Exception as e:
            print(f"GPU info error: {e}")
    else:
        print("GPU monitoring not available")
    
    print("="*50)


if __name__ == "__main__":
    print_system_info()
    
    # Test monitoring
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print("\nTesting monitoring for 5 seconds...")
    time.sleep(5)
    
    monitor.stop_monitoring()
    monitor.print_summary() 