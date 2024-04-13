"""
https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/metrics.py
"""

class DataLogger(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt
