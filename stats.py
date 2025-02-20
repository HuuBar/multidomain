import time
from collections import defaultdict

class SearchStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.timings = defaultdict(float)
        self.counters = defaultdict(int)
        self.start_time = None
    
    def start_timer(self, phase):
        self.start_time = time.time()
        self.current_phase = phase
    
    def end_timer(self):
        if self.start_time:
            self.timings[self.current_phase] += time.time() - self.start_time
            self.start_time = None
    
    def log_counter(self, name, value=1):
        self.counters[name] += value

    def get_report(self):
        return {
            "time": dict(self.timings),
            "counts": dict(self.counters)
        }