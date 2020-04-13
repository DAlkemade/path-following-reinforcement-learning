class Config:
    def __init__(self):
        self.epsilon = .1
        self.gamma = .99
        self.batch_size = 1000
        self.copy_step = 25
        self.max_steps_in_run = 1000
        self.memory_size = 1000000
        self.num_layers = 2
        self.learning_rate = .01
