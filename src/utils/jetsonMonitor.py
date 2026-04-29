import threading
import time
from jtop import jtop

class JetsonMonitor(threading.Thread):
    def __init__(self, delay=0.5):
        super().__init__()
        self.stopped = False
        self.delay = delay
        self.cpu_keys = ['CPU1', 'CPU2', 'CPU3', 'CPU4', 'CPU5', 'CPU6']
        # total Watts consumption
        self.power_samples = []
        # percentage of GPU utilization
        self.gpu_util_samples = []
        # percentage of CPU utilization (sum of all cores, max 600% for 6 cores)
        self.cpu_util_samples = []
        self.gpu_temp_samples = []
        self.cpu_temp_samples = []
        self.ram_util_samples = []
        self.nvp_model = None

    def run(self):
        try:
            with jtop() as jetson:
                self.nvp_model = jetson.stats["nvp model"]
                while not self.stopped:
                    if jetson.ok():
                        self.power_samples.append(jetson.stats["Power TOT"])
                        self.gpu_util_samples.append(jetson.stats["GPU"])
                        self.gpu_temp_samples.append(jetson.stats["Temp gpu"])
                        self.cpu_temp_samples.append(jetson.stats["Temp cpu"])
                        self.ram_util_samples.append(jetson.stats["RAM"])
                        core_values = [jetson.stats.get(k, 0) for k in self.cpu_keys]
                        self.cpu_util_samples.append(sum(core_values))
                    time.sleep(self.delay)
        except Exception as e:
            print(f"Error in JetsonMonitor: {e}")

    def get_stats(self):
        if not self.power_samples:
            return {
            "avg_power": 0, 
            "max_power": 0, 
            "avg_gpu": 0,
            "avg_cpu": 0,
            "max_gpu_temp": 0,
            "avg_gpu_temp": 0,
            "max_cpu_temp": 0,
            "avg_cpu_temp": 0, 
            "avg_ram": 0
        }
        return {
            "avg_power": sum(self.power_samples) / len(self.power_samples),
            "max_power": max(self.power_samples),
            "avg_gpu": sum(self.gpu_util_samples) / len(self.gpu_util_samples),
            "avg_cpu": sum(self.cpu_util_samples) / len(self.cpu_util_samples),
            "max_gpu_temp": max(self.gpu_temp_samples),
            "avg_gpu_temp": sum(self.gpu_temp_samples) / len(self.gpu_temp_samples),
            "max_cpu_temp": max(self.cpu_temp_samples),
            "avg_cpu_temp": sum(self.cpu_temp_samples) / len(self.cpu_temp_samples),
            "avg_ram": sum(self.ram_util_samples) / len(self.ram_util_samples)
        }