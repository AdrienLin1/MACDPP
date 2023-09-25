#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle
import paddle.nn.functional as F
from copy import deepcopy
from parl.utils.utils import check_model_method
import numpy as np

__all__ = ['DDPG']
beta = 0.05
alpha = 1.0
num = 30

class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """ DDPG algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal, agents):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal, agents)
        actor_loss = self._actor_learn(obs, agents)

        self.sync_target()
        return critic_loss, actor_loss


    def _critic_learn(self, obs, action, reward, next_obs, terminal, agents):
        with paddle.no_grad():
            # Compute the target Q value
            # first compute next s'_A
            for i, agent in enumerate(agents):
                if i == 0:
                    action1 = agent.alg.target_model.policy(next_obs)
                elif i == 1:
                    action2 = agent.alg.target_model.policy(next_obs)
                elif i == 2:
                    action3 = agent.alg.target_model.policy(next_obs)
                elif i == 3:
                    action4 = agent.alg.target_model.policy(next_obs)
                else:
                    action5 = agent.alg.target_model.policy(next_obs)
                    action6 = paddle.concat([action1, action2, action3, action4, action5], axis=-1)
            # action3 = np.array(action3)
            next_action = action6
            next_obs = paddle.tile(next_obs, [num + 1, 1])
            next_action = self._expand(next_action)
            next_action = next_action.clip(-1.0, 1.0)
            target_next_P = self.target_model.value(next_obs, next_action)
            target_next_P = paddle.reshape(target_next_P, shape=(num + 1, -1))
            # p_next_s = self.boltzmann_max(target_next_P)
            p_next_s = self.mellow_max(target_next_P)
            # p_next_s = self.softmax_operator(target_next_P)
            p_next_s = paddle.reshape(p_next_s, shape=(-1, 1))

            # then compute current s_A
            target_obs = paddle.tile(obs, [num + 1, 1])
            current_action = action
            current_action = self._expand(current_action)
            current_action.clip(-1.0, 1.0)
            target_current_P = self.target_model.value(target_obs, current_action)
            target_current_P = paddle.reshape(target_current_P, shape=(num + 1, -1))
            # p_current_s = self.boltzmann_max(target_current_P)
            p_current_s = self.mellow_max(target_current_P)
            # p_current_s = self.softmax_operator(target_current_P)
            p_current_s = paddle.reshape(p_current_s, shape=(-1, 1))

            # finally compute current Q(s_a)
            target_current_Q = self.target_model.value(obs, action)

            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = (reward + ((1. - terminal) * self.gamma * p_next_s) +
                        alpha * (target_current_Q - p_current_s))


        # Get current Q estimate
        current_Q = self.model.value(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs, agents):
        # Compute actor loss and Update the frozen target models
        for i, agent in enumerate(agents):
            if i == 0:
                action8 = agent.alg.model.policy(obs)
            elif i == 1:
                action9 = agent.alg.model.policy(obs)
            elif i == 2:
                action10 = agent.alg.model.policy(obs)
            elif i == 3:
                action11 = agent.alg.model.policy(obs)
            else:
                action12 = agent.alg.model.policy(obs)
                action13 = paddle.concat([action8, action9, action10, action11, action12], axis=-1)

        actor_loss = -self.model.value(obs, action13).mean()
        # actor_loss = -self.model.value(obs, self.model.policy(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        """ update the target network with the training network

        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)

    def _expand(self, action):

        action_batch = action.shape[0]
        tile_action = paddle.tile(action, [num, 1])
        sample = paddle.normal(mean=0, std=0.1, shape=[1, num])
        #sample = sample.clip(-0.5, 0.5)
        sample = paddle.to_tensor(sample)
        sample = paddle.reshape(sample, shape=(-1, 1))
        sample = paddle.tile(sample, [1, action_batch])
        sample = paddle.flatten(sample)
        sample = paddle.to_tensor(sample)
        sample = paddle.reshape(sample, shape=(-1, 1))
        sample = paddle.cast(sample, dtype=paddle.float32)
        expand_action = tile_action + sample
        action = paddle.concat(x=[action, expand_action], axis=0)
        return action

    # 解决了溢出问题
    def mellow_max(self, q_vals):
        max_q_vals = paddle.max(q_vals, axis=0, keepdim=True)
        q_vals = q_vals - max_q_vals
        e_beta_Q = paddle.exp(beta * q_vals)

        sum_e_beta_Q = paddle.sum(e_beta_Q, 0) / (num + 1)
        max_q_vals = paddle.squeeze(max_q_vals)
        log_sum_Q = paddle.log(sum_e_beta_Q) + max_q_vals * beta

        softmax_q_vals = log_sum_Q/beta

        return softmax_q_vals

    def get_q(self, obs, action):
        return self.model.value(obs, action)