class FoldResult:
    def __init__(self):
        self.loss = 0
        self.acc_train = 0
        self.acc_test = 0
        self.nrules = 0
        self.nconditions = 0
        self.time = 0


class TestRunResult:
    def __init__(self):
        self.model = ""
        self.dataset = ""
        self.n_samples = 0
        self.n_features = 0
        self.loss_mean = 0
        self.acc_train_mean = 0
        self.acc_test_mean = 0
        self.nrules_mean = 0
        self.nconditions_mean = 0
        self.time_mean = 0