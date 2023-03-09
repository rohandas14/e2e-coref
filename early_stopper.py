import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_training_loss=np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = min_training_loss

    def early_stop(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
