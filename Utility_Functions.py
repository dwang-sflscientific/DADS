import numpy as np
from abc import ABCMeta
import os
from datetime import datetime
import json
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.distributions import Categorical, normal
from nn_builder.pytorch.Base_Network import Base_Network
from collections import namedtuple, deque
import random
import copy


def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


class Logger():
    def __init__(self, log_path, prefix="", warning_level=3, print_to_terminal=True):
        unique_path = self.make_simple_log_path(prefix)
        log_path = os.path.join(log_path, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "output.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level

    def make_simple_log_path(self, prefix):
        now = datetime.now()
        suffix = now.strftime("%d(%H.%M)")
        pid_str = os.getpid()
        return "{}-{}-{}".format(prefix, suffix, pid_str)

    @property
    def log_dir(self):
        return self.log_path

    def log_str(self, content, print_to_terminal=False, level=4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path, 'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, ite):
        self.tb_writer.add_scalar(name, val, ite)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict is not None:
            log_str = json.dumps(log_dict, indent=4)
        elif log_str is not None:
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path, 'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))


class NN(nn.Module, Base_Network):
    """Creates a PyTorch neural network
    Args:
        - input_dim: Integer to indicate the dimension of the input into the network
        - layers_info: List of integers to indicate the width and number of linear layers you want in your network,
                      e.g. [5, 8, 1] would produce a network with 3 linear layers of width 5, 8 and then 1
        - hidden_activations: String or list of string to indicate the activations you want used on the output of hidden layers
                              (not including the output layer). Default is ReLU.
        - output_activation: String to indicate the activation function you want the output to go through. Provide a list of
                             strings if you want multiple output heads
        - dropout: Float to indicate what dropout probability you want applied after each hidden layer
        - initialiser: String to indicate which initialiser you want used to initialise all the parameters. All PyTorch
                       initialisers are supported. PyTorch's default initialisation is the default.
        - batch_norm: Boolean to indicate whether you want batch norm applied to the output of every hidden layer. Default is False
        - columns_of_data_to_be_embedded: List to indicate the columns numbers of the data that you want to be put through an embedding layer
                                          before being fed through the other layers of the network. Default option is no embeddings
        - embedding_dimensions: If you have categorical variables you want embedded before flowing through the network then
                                you specify the embedding dimensions here with a list like so: [ [embedding_input_dim_1, embedding_output_dim_1],
                                [embedding_input_dim_2, embedding_output_dim_2] ...]. Default is no embeddings
        - y_range: Tuple of float or integers of the form (y_lower, y_upper) indicating the range you want to restrict the
                   output values to in regression tasks. Default is no range restriction
        - random_seed: Integer to indicate the random seed you want to use
        - print_model_summary: Boolean to indicate whether you want a model summary printed after model is created. Default is False.
    """
    def __init__(self, input_dim, layers_info, output_activation=None,
                 hidden_activations="relu", dropout=0.0, initialiser="default", batch_norm=False,
                 columns_of_data_to_be_embedded=[], embedding_dimensions=[], y_range= (),
                 random_seed=0, print_model_summary=False):
        nn.Module.__init__(self)
        self.embedding_to_occur = len(columns_of_data_to_be_embedded) > 0
        self.columns_of_data_to_be_embedded = columns_of_data_to_be_embedded
        self.embedding_dimensions = embedding_dimensions
        self.embedding_layers = self.create_embedding_layers()
        Base_Network.__init__(self, input_dim, layers_info, output_activation,
                              hidden_activations, dropout, initialiser, batch_norm, y_range, random_seed, print_model_summary)

    def check_all_user_inputs_valid(self):
        """Checks that all the user inputs were valid"""
        self.check_NN_input_dim_valid()
        self.check_NN_layers_valid()
        self.check_activations_valid()
        self.check_embedding_dimensions_valid()
        self.check_initialiser_valid()
        self.check_y_range_values_valid()

    def create_hidden_layers(self):
        """Creates the linear layers in the network"""
        linear_layers = nn.ModuleList([])
        input_dim = int(self.input_dim - len(self.embedding_dimensions) + np.sum([output_dims[1] for output_dims in self.embedding_dimensions]))
        for hidden_unit in self.layers_info[:-1]:
            linear_layers.extend([nn.Linear(input_dim, hidden_unit)])
            input_dim = hidden_unit
        return linear_layers

    def create_output_layers(self):
        """Creates the output layers in the network"""
        output_layers = nn.ModuleList([])
        if len(self.layers_info) >= 2: input_dim = self.layers_info[-2]
        else: input_dim = self.input_dim
        if not isinstance(self.layers_info[-1], list): output_layer = [self.layers_info[-1]]
        else: output_layer = self.layers_info[-1]
        for output_dim in output_layer:
            output_layers.extend([nn.Linear(input_dim, output_dim)])
        return output_layers

    def create_batch_norm_layers(self):
        """Creates the batch norm layers in the network"""
        batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_unit) for hidden_unit in self.layers_info[:-1]])
        return batch_norm_layers

    def initialise_all_parameters(self):
        """Initialises the parameters in the linear and embedding layers"""
        self.initialise_parameters(self.hidden_layers)
        self.initialise_parameters(self.output_layers)
        self.initialise_parameters(self.embedding_layers)

    def forward(self, x):
        """Forward pass for the network"""
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x.cpu())
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        x = self.process_hidden_layers(x)
        out = self.process_output_layers(x)
        if self.y_range: out = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(out)
        return out

    def map(self, x):
        if not self.checked_forward_input_data_once: self.check_input_data_into_forward_once(x.cpu())
        if self.embedding_to_occur: x = self.incorporate_embeddings(x)
        x = self.process_hidden_layers(x)
        return x

    def check_input_data_into_forward_once(self, x):
        """Checks the input data into forward is of the right format. Then sets a flag indicating that this has happened once
        so that we don't keep checking as this would slow down the model too much"""
        for embedding_dim in self.columns_of_data_to_be_embedded:
            data = x[:, embedding_dim]
            data_long = data.long()
            assert all(data_long >= 0), "All data to be embedded must be integers 0 and above -- {}".format(data_long)
            assert torch.sum(abs(data.float() - data_long.float())) < 0.0001, """Data columns to be embedded should be integer 
                                                                                values 0 and above to represent the different 
                                                                                classes"""
        if self.input_dim > len(self.columns_of_data_to_be_embedded): assert isinstance(x, torch.FloatTensor), f'Incorrect Tensor type x is {type(x)} is {x}'
        assert len(x.shape) == 2, "X should be a 2-dimensional tensor: {}".format(x.shape)
        self.checked_forward_input_data_once = True #So that it doesn't check again

    def incorporate_embeddings(self, x):
        """Puts relevant data through embedding layers and then concatenates the result with the rest of the data ready
        to then be put through the linear layers"""
        all_embedded_data = []
        for embedding_layer_ix, embedding_var in enumerate(self.columns_of_data_to_be_embedded):
            data = x[:, embedding_var].long()
            embedded_data = self.embedding_layers[embedding_layer_ix](data)
            all_embedded_data.append(embedded_data)
        all_embedded_data = torch.cat(tuple(all_embedded_data), dim=1)
        x = torch.cat((x[:, [col for col in range(x.shape[1]) if col not in self.columns_of_data_to_be_embedded]].float(), all_embedded_data), dim=1)
        return x

    def process_hidden_layers(self, x):
        """Puts the data x through all the hidden layers"""
        for layer_ix, linear_layer in enumerate(self.hidden_layers):
            x = self.get_activation(self.hidden_activations, layer_ix)(linear_layer(x))
            if self.batch_norm: x = self.batch_norm_layers[layer_ix](x)
            if self.dropout != 0.0: x = self.dropout_layer(x)
        return x

    def process_output_layers(self, x):
        """Puts the data x through all the output layers"""
        out = None
        for output_layer_ix, output_layer in enumerate(self.output_layers):
            activation = self.get_activation(self.output_activation, output_layer_ix)
            temp_output = output_layer(x)
            if activation is not None: temp_output = activation(temp_output)
            if out is None: out = temp_output
            else: out = torch.cat((out, temp_output), dim=1)
        return out


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, device=None):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        states, next_states = states.cpu(), next_states.cpu()
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state
