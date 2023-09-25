import parl
import paddle
import numpy as np
from parl.utils import ReplayMemory
from parl.utils import logger, summary

sample_num = 50
num = 30
tem = 1
boltzmann_exploration = 0.2

class MAAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 ):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.n = len(act_dim_n)
        self.memory_size = int(1e5)
        self.min_memory_size = batch_size * 25  # batch_size * args.max_episode_len
        self.rpm = ReplayMemory(
            max_size=self.memory_size,
            obs_dim=self.obs_dim_n[agent_index],
            act_dim=self.act_dim_n[agent_index])
        self.global_train_step = 0

        super(MAAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        """ predict action by model
        """
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        act = self.alg.predict(obs)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy


    def sample(self, obs, agents, use_target_model = False):
        # print(obs)
        # print(type(obs))
        obs1 = [None]*(len(obs))
        action_numpy = [None] * (len(obs))
        action_numpy[self.agent_index] = (paddle.to_tensor((self.predict(obs[self.agent_index])))).clip(-1.0, 1.0)
        for i in range(len(obs)):
            if i != self.agent_index:
                action_numpy[i] = (paddle.to_tensor((agents[i].predict(obs[i])))).clip(-1.0, 1.0)
        for i in range(len(obs)):
            obs1[i] = paddle.to_tensor(obs[i])
            obs1[i] = paddle.tile(obs1[i], [sample_num+1, 1])
        action_numpy[self.agent_index] = self.expand_action(action_numpy[self.agent_index])
        for i in range(len(obs)):
            if i != self.agent_index:
                action_numpy[i] = self.expand_action1(action_numpy[i])
        if not use_target_model:
            q_value = self.alg.Q(obs1, action_numpy)
        else:
            q_value = self.alg.Q(obs1, action_numpy, use_target_model=True)
        action_dist = self.softmax(q_value)
        action_dist_cusum = paddle.cumsum(action_dist)
        rand_num = paddle.rand(shape=[1])
        flag = paddle.greater_than(action_dist_cusum, rand_num)
        flag = paddle.cast(flag, dtype=paddle.int32)
        flag = paddle.sum(flag)
        index = sample_num + 1 - int(flag)
        if index == (sample_num + 1):
            return paddle.tanh(action_numpy[self.agent_index][0]).numpy()
        else:
            return paddle.tanh(action_numpy[self.agent_index][index]).numpy()

    def learn(self, agents):
        """ sample batch, compute q_target and train
        """
        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        # sample batch
        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _ \
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)
        _, _, batch_rew, _, batch_isOver = self.rpm.sample_batch_by_index(
            rpm_sample_index)
        batch_obs_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_n
        ]
        batch_act_n = [
            paddle.to_tensor(act, dtype='float32') for act in batch_act_n
        ]
        batch_rew = paddle.to_tensor(batch_rew, dtype='float32')
        batch_isOver = paddle.to_tensor(batch_isOver, dtype='float32')

        # compute target q
        target_act_next_n = []
        batch_obs_next_n = [
            paddle.to_tensor(obs, dtype='float32') for obs in batch_obs_next_n
            ]

        with paddle.no_grad():
            for i in range(self.n):
                target_act_next = agents[i].alg.sample(
                    batch_obs_next_n[i], use_target_model=True)
                target_act_next = target_act_next.detach()
                target_act_next_n.append(target_act_next)

            target_act_next_n1 = []
            batch_obs_next_n1 = []
            for i in range(self.n):
                batch_obs_next_n1.append(paddle.to_tensor(paddle.tile(batch_obs_next_n[i], [num + 1, 1])))
                target_act_next_n1.append(self.alg._expand(target_act_next_n[i]))
                target_act_next_n1[i] = target_act_next_n1[i].clip(-1.0, 1.0)
            target_q_next = self.alg.Q(batch_obs_next_n1, target_act_next_n1, use_target_model=True)
            target_q_next = paddle.reshape(target_q_next, shape=(num + 1, -1))
            p_next_s = self.alg.mellow_max(target_q_next)
            p_next_s = paddle.reshape(p_next_s, shape=(-1, 1))

            target_obs = []
            current_action1 = []
            for i in range(self.n):
                target_obs.append(paddle.to_tensor(paddle.tile(batch_obs_n[i], [num + 1, 1])))
                current_action = batch_act_n
                current_action1.append(self.alg._expand(current_action[i]))
            target_current_P = self.alg.Q(target_obs, current_action1, use_target_model=True)
            target_current_P = paddle.reshape(target_current_P, shape=(num + 1, -1))
            p_current_s = self.alg.mellow_max(target_current_P)
            p_current_s = paddle.reshape(p_current_s, shape=(-1, 1))

            # finally compute current Q(s_a)
            target_current_Q = self.alg.Q(batch_obs_n, batch_act_n, use_target_model=True)

            target_q = (batch_rew + ((1. - batch_isOver) * self.alg.gamma * p_next_s) +
                        self.alg.alpha * (target_current_Q - p_current_s))

        # learn
        actor_cost, critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        actor_cost = actor_cost.cpu().detach().numpy()[0]
        critic_cost = critic_cost.cpu().detach().numpy()[0]
        summary.add_scalar('actor_loss/step', actor_cost, self.global_train_step)
        summary.add_scalar('critic_loss/step', critic_cost, self.global_train_step)

        return actor_cost, critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)

    def expand_action(self, action):
        action = paddle.reshape(action, shape=(1, -1))
        tile_action = paddle.tile(action, [sample_num, 1])
        s_action = paddle.normal(mean=0, std=boltzmann_exploration, shape=[1, sample_num])
        s_action = paddle.reshape(s_action, shape=(-1, 1))
        e_action = s_action + tile_action
        e_action = (paddle.concat(x=[action, e_action], axis=0))
        return e_action

    def expand_action1(self, action):
        action = paddle.reshape(action, shape=(1, -1))
        tile_action = paddle.tile(action, [sample_num, 1])
        e_action = (paddle.concat(x=[action, tile_action], axis=0))
        return e_action

    def softmax(self, x):
        e_x = paddle.exp(tem * x)
        sum_e_x = paddle.sum(e_x)
        dist_x = e_x / sum_e_x
        return dist_x