import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 critic_in_dim,
                 continuous_actions=False):
        super(MAModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        hid1_size = 64
        hid2_size = 64
        self.fc1 = nn.Linear(
            obs_dim,
            hid1_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(
            hid2_size,
            act_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        if self.continuous_actions:
            std_hid_size = 64
            self.std_fc = nn.Linear(
                std_hid_size,
                act_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs):
        #print(obs)
        #print(type(obs))
        #print(obs.shape)
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            act_std = self.std_fc(hid2)
            return (means, act_std)
        return means


class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        out_dim = 1
        self.fc1 = nn.Linear(
            critic_in_dim,
            hid1_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))
        self.fc3 = nn.Linear(
            hid2_size,
            out_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()))

    def forward(self, obs_n, act_n):
        # print(type(obs_n))
        # print(type(act_n))
        # print(len(obs_n))
        # print(len(act_n))
        inputs = paddle.concat(obs_n + act_n, axis=1)
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        Q = paddle.squeeze(Q, axis=1)
        return Q