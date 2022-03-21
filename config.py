train_percentage = 0.8
known_anomaly_num = 30
device = 'cuda'
sample_num = 200
max_trajectory = 1000
check_num = 4
reward1 = -2
reward2 = 0.5
dataset_name = 'ann'
strategy_distribution = [0.3, 0.3, 0.4]
target_anomaly_classes = [1, 4, 2]
refresh_interval = 500
seed = 1
normalization = True


class Config():
    def __init__(self):
        # env
        self.train_percentage = train_percentage
        self.known_anomaly_num = known_anomaly_num
        self.device = device
        self.sample_num = sample_num
        self.max_trajectory = max_trajectory
        self.check_num = check_num
        self.reward1 = reward1
        self.reward2 = reward2
        self.strategy_distribution = strategy_distribution
        self.target_anomaly_classes = target_anomaly_classes
        self.refresh_interval = refresh_interval
        self.dataset_name = dataset_name
        self.normalization = normalization

        # main
        self.seed = seed
        self.num_episodes_to_run = 20
        self.standard_deviation_results = 1.0
        self.runs_per_agent = 10
        self.use_GPU = True
        self.randomise_random_seed = False

        self.hyperparameters = {
            "Actor_Critic_Agents": {
                "Actor": {
                    "learning_rate": 0.002,
                    "linear_hidden_units": [64, 32],
                    "final_layer_activation": "Softmax",
                    "batch_norm": False,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },
                "Critic": {
                    "learning_rate": 0.002,
                    "linear_hidden_units": [64, 32],
                    "final_layer_activation": None,
                    "batch_norm": False,
                    "buffer_size": 2000,
                    "tau": 0.2,
                    "gradient_clipping_norm": 5,
                    "initialiser": "Xavier"
                },

                "min_steps_before_learning": 1000,
                "batch_size": 64,
                "update_every_n_steps": 64,
                "learning_updates_per_learning_session": 64,
                "automatically_tune_entropy_hyperparameter": True,
                "entropy_term_weight": None,
                "add_extra_noise": False,
                "discount_rate": 0.99,
                "clip_rewards": False,
            }
        }
