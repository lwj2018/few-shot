class Arguments:
    def __init__(self,shot,dataset):
        self.num_class = 100

        # Settings for 5-shot
        if shot == 5:
            self.shot = 5
            self.query = 5
            self.query_val = 15
        # Settings for 1-shot
        elif shot == 1:
            self.shot = 1
            self.query = 1
            self.query_val = 5
        
        if dataset == 'miniImage':
            self.n_base = 80
        elif dataset == 'omniglot':
            self.n_base = 964
        self.train_way = 20
        self.test_way = 5
        self.feature_dim = 1600
        # Options
        self.num_workers = 8