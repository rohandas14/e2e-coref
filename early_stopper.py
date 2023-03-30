import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, max_val_f1=-np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_val_f1 = max_val_f1

    def early_stop(self, val_f1):
        if val_f1 > self.max_val_f1:
            self.max_val_f1 = val_f1
            self.counter = 0
        elif val_f1 < (self.max_val_f1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False