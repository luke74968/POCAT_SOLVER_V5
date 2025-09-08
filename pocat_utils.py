# pocat_utils.py
from utils.common import TimeEstimator, batchify, unbatchify, clip_grad_norms


class AverageMeter:
    """ 여러 값의 평균을 계속 추적하는 클래스 """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0
