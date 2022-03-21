import copy
import random
import numpy as np
from Utility_Functions import Logger


class Trainer(object):
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        env_name = "Anomaly_dectection"
        self.logger = Logger("logs", prefix=env_name + "-" + "", print_to_terminal=True)
        self.logger.log_str("logging to {}".format(self.logger.log_path))
        self.logger.log_str(config.__dict__)

    def run_game_for_agent(self):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_group = "Actor_Critic_Agents"
        agent_round = 1
        auc_roc_list = []
        auc_pr_list = []
        time_list = []
        for run in range(self.config.runs_per_agent):
            print("Run ", run + 1)
            agent_config = copy.deepcopy(self.config)

            if self.config.randomise_random_seed: agent_config.seed = random.randint(0, 2**32 - 2)
            agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
            agent = self.agent(agent_config)
            time_taken, auc_roc, auc_pr = agent.run_n_episodes(self.logger, run)
            auc_roc_list.append(auc_roc)
            auc_pr_list.append(auc_pr)
            time_list.append(time_taken)
            time_str = "Time taken: {}".format(time_taken)
            self.logger.log_str(time_str)
            print(time_str, flush=True)
            self.print_two_empty_lines()
            agent_round += 1
        auc_roc_str = "auc_roc: mean {}, variance {}".format(np.mean(auc_roc_list), np.var(auc_roc_list))
        self.logger.log_str(auc_roc_str)
        print(auc_roc_str)
        auc_pr_str = "auc_pr: mean {}, variance {}".format(np.mean(auc_pr_list), np.var(auc_pr_list))
        self.logger.log_str(auc_pr_str)
        print(auc_pr_str)
        time_str = "time: mean {}, variance {}".format(np.mean(time_list), np.var(time_list))
        self.logger.log_str(time_str)
        print(time_str)

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")
