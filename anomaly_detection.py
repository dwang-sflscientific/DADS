import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
from sklearn.utils import shuffle
from gym import spaces
import os
import config
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.manifold import TSNE

all_anomaly_classes = [[1, 2], [4, 6], [2, 3]]
all_normal_classes = [[3], [2], [1]]
all_classes = [[3, 1, 2], [2, 4, 6], [1, 2, 3]]

train_percentage = config.train_percentage
known_anomaly_num = config.known_anomaly_num
device = config.device
sample_num = config.sample_num
max_trajectory = config.max_trajectory
check_num = config.check_num
reward1 = config.reward1
reward2 = config.reward2
strategy_distribution = config.strategy_distribution
target_anomaly_classes = config.target_anomaly_classes
refresh_interval = config.refresh_interval
dataset_name = config.dataset_name
normalization = config.normalization


def load_data():
    if dataset_name == 'ann':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/annthyroid.csv"))
        index = 0
    elif dataset_name == 'cov':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/covertype.csv"))
        if normalization:
            source.iloc[:, :10] = (source.iloc[:, :10] - source.iloc[:, :10].min()) / (
                    source.iloc[:, :10].max() - source.iloc[:, :10].min())
        index = 1
    elif dataset_name == 'car':
        source = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/cardio.csv"))
        if normalization:
            source.iloc[:, :23] = (source.iloc[:, :23] - source.iloc[:, :23].min()) / (
                    source.iloc[:, :23].max() - source.iloc[:, :23].min())
        index = 2
    else:
        assert 0, "Dataset not existed."
    target_anomaly_class = target_anomaly_classes[index]
    all_anomaly_class = all_anomaly_classes[index]
    source = shuffle(source)
    width = source.shape[1]
    length = len(source)
    dataset_train = source.iloc[:int(length*train_percentage), :]
    dataset_test = source.iloc[int(length*train_percentage):, :]
    dataset_a = pd.DataFrame(columns=source.columns)
    dataset_u = pd.DataFrame(columns=source.columns)
    for i in range(len(dataset_train)):
        label = dataset_train.iloc[i, width-1]
        if label == target_anomaly_class and len(dataset_a) < known_anomaly_num:
            dataset_a = dataset_a.append(dataset_train.iloc[i, :])
        else:
            if len(dataset_a) < known_anomaly_num:
                dataset_u = dataset_u.append(dataset_train.iloc[i, :])
            else:
                dataset_u = dataset_u.append(dataset_train.iloc[i:, :])
                break
    dataset_a = dataset_a.reset_index(drop=True)
    dataset_u = dataset_u.reset_index(drop=True)

    source = np.concatenate([np.array(dataset_a.values.astype(float)), np.array(dataset_u.values.astype(float))])
    dataset_a = torch.tensor(dataset_a.values.astype(float))[:, :-1].float().to(device)
    dataset_u = torch.tensor(dataset_u.values.astype(float))[:, :-1].float().to(device)
    test_label = torch.tensor(dataset_test.values.astype(float))[:, -1].float().to(device)
    test_label = [1 if i in all_anomaly_class else 0 for i in test_label]
    dataset_test = torch.tensor(dataset_test.values.astype(float))[:, :-1].float().to(device)

    return dataset_a, dataset_u, dataset_test, test_label, source, index


class ad(gym.Env):
    def __init__(self):
        print("Start loading")
        dataset_a, dataset_u, dataset_test, test_label, source, index = load_data()
        print("Finish loading")
        print("Anomaly num: ", len(dataset_a), "   Normal num: ", len(dataset_u))
        self.dataset_anomaly = dataset_a
        self.dataset_unlabeled = dataset_u
        self.dataset_anomaly_backup = dataset_a
        self.dataset_unlabeled_backup = dataset_u
        self.dataset_temp = torch.tensor([]).to(device)
        self.dataset_test = dataset_test
        self.test_label = test_label
        self.tempdata_confidence = []

        self.source = source
        self.classes = all_classes[index]

        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = "unlabeled"

        self.net = None

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Discrete(2)
        self.tot_steps = 0

        self.clf = IsolationForest()

    def reset(self):
        self.dataset_anomaly = self.dataset_anomaly_backup
        self.dataset_unlabeled = self.dataset_unlabeled_backup
        self.dataset_temp = torch.tensor([]).to(device)

        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = 'unlabeled'

        return self.current_data

    def refresh_net(self, net):
        self.net = net

    def calculate_reward(self, action):
        if self.current_class == 'anomaly':
            if action == 1:
                score = 1
            else:
                score = reward1
        elif self.current_class == 'unlabeled':
            score = 0
        elif self.current_class == 'temp':
            if action == 1 and self.tempdata_confidence[self.current_index] >= check_num:
                score = reward2
            else:
                score = 0
        else:
            assert 0

        return score

    def sample_method_one(self):
        self.current_class = 'anomaly'
        self.current_index = random.randint(0, len(self.dataset_anomaly) - 1)
        self.current_data = self.dataset_anomaly[self.current_index]

    def sample_method_two(self):
        self.current_class = 'temp'
        self.current_index = random.randint(0, len(self.dataset_temp) - 1)
        self.current_data = self.dataset_temp[self.current_index]

    def sample_method_three(self, action):
        self.current_class = 'unlabeled'
        candidate_index = np.random.choice([i for i in range(len(self.dataset_unlabeled))], size=sample_num,
                                           replace=False)
        candidate = self.dataset_unlabeled[candidate_index]
        feature = self.net.map(candidate)
        score = -self.clf.score_samples(feature.detach().cpu().numpy())
        self.current_index = np.argmax(score)
        self.current_data = self.dataset_unlabeled[self.current_index]

    def refresh_dataset(self, action):
        if action == 1 and self.current_class == 'unlabeled':
            self.dataset_unlabeled = torch.cat(
                [self.dataset_unlabeled[:self.current_index], self.dataset_unlabeled[self.current_index+1:]])
            self.tempdata_confidence.append(1)
            self.dataset_temp = torch.cat([self.dataset_temp, self.current_data.unsqueeze(0)])
        elif action == 1 and self.current_class == 'temp':
            if self.tempdata_confidence[self.current_index] >= check_num:
                self.dataset_temp = torch.cat(
                    [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
                self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                           self.tempdata_confidence[self.current_index + 1:]
                self.dataset_anomaly = torch.cat([self.dataset_anomaly, self.current_data.unsqueeze(0)])
            else:
                self.tempdata_confidence[self.current_index] = self.tempdata_confidence[self.current_index] + 1
        elif action == 0 and self.current_class == 'temp':
            self.dataset_temp = torch.cat(
                [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
            self.dataset_unlabeled = torch.cat([self.dataset_unlabeled, self.current_data.unsqueeze(0)])
            self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                       self.tempdata_confidence[self.current_index + 1:]

    def step(self, action):
        reward = self.calculate_reward(action)
        if self.tot_steps % refresh_interval == 0:
            with torch.no_grad():
                mapped = self.net.map(self.dataset_unlabeled).cpu()
            self.clf.fit(mapped)
        self.tot_steps = self.tot_steps + 1
        self.refresh_dataset(action)

        done = False
        if self.tot_steps % max_trajectory == 0:
            done = True

        while True:
            choice = np.random.choice([0, 1, 2], size=1, p=strategy_distribution)
            if choice == 0 and len(self.dataset_anomaly) != 0:
                self.sample_method_one()
                break
            elif choice == 1 and len(self.dataset_temp) != 0:
                self.sample_method_two()
                break
            elif choice == 2 and len(self.dataset_unlabeled) != 0:
                self.sample_method_three(action)
                break
        return self.current_data, reward, done, " "

    def evaluate(self):
        q_values = self.net(self.dataset_test)
        anomaly_score = q_values[:, 1]
        auc_roc = roc_auc_score(self.test_label, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(self.test_label, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr

    def plot(self):
        anomaly = self.dataset_anomaly.cpu().numpy()
        temp = self.dataset_temp.cpu().numpy()
        unlabeled = self.dataset_unlabeled.cpu().numpy()
        if len(temp) != 0:
            dataset_for_plot = np.vstack([unlabeled, temp, anomaly])
        else:
            dataset_for_plot = np.vstack([unlabeled, anomaly])
        decomposer = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0)
        decomposed_data = iter(decomposer.fit_transform(dataset_for_plot))

        color_list = ['green', 'orange', 'red']
        shape_list = [".", "^", "+", "x"]
        for i in range(3):
            color = color_list[i]
            for item in [unlabeled, temp, anomaly][i]:
                line = np.where(np.sum((self.source[:, :-1] - item)**2, axis=1) < 1e-9)[0][0]
                type = self.source[line, -1]
                shape = shape_list[self.classes.index(type)]
                dot = next(decomposed_data)
                plt.scatter(dot[0], dot[1], c=color, marker=shape)
        plt.show()
