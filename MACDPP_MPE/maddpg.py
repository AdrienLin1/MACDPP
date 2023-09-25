import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
from parl.utils import logger, summary
import time # 使用时间模块

__all__ = ['MADDPG']


class MADDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """  MADDPG algorithm
        Args:
            model (parl.Model): forward network of actor and critic.
                                The function get_actor_params() of model should be implemented.
            agent_index (int): index of agent, in multiagent env
            act_space (list): action_space, gym space
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            critic_lr (float): learning rate of the critic model
            actor_lr (float): learning rate of the actor model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.continuous_actions = False
        if not len(act_space) == 0 and hasattr(act_space[0], 'high'):
            self.continuous_actions = True

        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)
        self.sync_target(0)

        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=self.actor_lr,
            parameters=self.model.get_actor_params(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5))
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=self.critic_lr,
            parameters=self.model.get_critic_params(),
            grad_clip=nn.ClipGradByNorm(clip_norm=0.5))

    def predict(self, obs):
        """ use the policy model to predict actions

        Args:
            obs (paddle tensor): observation, shape([B] + shape of obs_n[agent_index])

        Returns:
            act (paddle tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        policy = self.model.policy(obs)
        if self.continuous_actions:
            mean = policy[0]
            action = paddle.tanh(mean)
            # mean = policy[0].clip(0.0, 1.0)
        else:
            action = F.softmax(policy, axis=-1)
        return action


    def sample(self, obs, use_target_model=False):
        """ use the policy model to sample actions

        Args:
            obs (paddle tensor): observation, shape([B] + shape of obs_n[agent_index])
            use_target_model (bool): use target_model or not

        Returns:
            act (paddle tensor): action, shape([B] + shape of act_n[agent_index]),
                noted that in the discrete case we take the argmax along the last axis as action
        """

        if use_target_model:
            policy = self.target_model.policy(obs)
            # print(policy)
        else:
            policy = self.model.policy(obs)

        # add noise for action exploration
        if self.continuous_actions:
            mean, std = policy[0], paddle.exp(policy[1])
            mean_shape = paddle.to_tensor(mean.shape, dtype='int64')
            random_normal = paddle.normal(shape=mean_shape)   # 返回符合正态分布的随机tensor
            # random_normal1 = paddle.normal(0,0.1,shape=mean_shape)  # 探索噪声为0.1
            action = mean + std * random_normal # 把std给去了，换成一个常量
            # action = mean + random_normal
            action = paddle.tanh(action)
            # print(action)
        else:
            eps = 1e-4
            logits_shape = paddle.to_tensor(policy.shape, dtype='int64')
            uniform = paddle.uniform(logits_shape, min=eps, max=1.0 - eps)
            soft_uniform = paddle.log(-1.0 * paddle.log(uniform))
            action = F.softmax(policy - soft_uniform, axis=-1)
        return action  # 返回一个agent的动作值

    def Q(self, obs_n, act_n, use_target_model=False):
        """ use the value model to predict Q values

        Args:
            obs_n (list of paddle tensor): all agents' observation, len(agent's num) + shape([B] + shape of obs_n)
            act_n (list of paddle tensor): all agents' action, len(agent's num) + shape([B] + shape of act_n)
            use_target_model (bool): use target_model or not
        Returns:
            Q (paddle tensor): Q value of this agent, shape([B])
        """
        if use_target_model:
            return self.target_model.value(obs_n, act_n)
        else:
            return self.model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, target_q):
        """ update actor and critic model with MADDPG algorithm
        """
        actor_cost = self._actor_learn(obs_n, act_n)
        critic_cost = self._critic_learn(obs_n, act_n, target_q)
        self.sync_target()
        return actor_cost, critic_cost

    """
    def _actor_learn(self, obs_n, act_n, agents):
        i = self.agent_index    # 获取当前agent的编号
        # print(obs_n)
        # print(type(obs_n))
        sample_this_action = paddle.tanh(self.model.policy((obs_n[i]))[0])
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action  # 把动作替换成了采样得到的动作
        eval_q = self.Q(obs_n, action_input_n)  # 计算Q值
        act_cost = paddle.mean(-1.0 * eval_q)     # 乘以-1再求平均就是actor网络对应的损失函数

        # this_policy = self.model.policy(obs_n[i])
        # # when continuous, 'this_policy' will be a tuple with two element: (mean, std)
        # if self.continuous_actions:
        #     this_policy = paddle.concat(this_policy, axis=-1)   # 把mean和std相加
        # act_reg = paddle.mean(paddle.square(this_policy))  # 加这个的目的，就是为了让输出的动作值不要太大
        #
        # cost = act_cost + act_reg * 1e-3
        cost = act_cost
        # summary.add_scalar('actor_loss', cost,int(time.time()))
        self.actor_optimizer.clear_grad()
        cost.backward()
        self.actor_optimizer.step()   # 来更新参数
        return cost
    """
    def _actor_learn(self, obs_n, act_n):
        i = self.agent_index
        #print(obs_n)
        sample_this_action = self.sample(obs_n[i])  # 这里要动
        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q(obs_n, action_input_n)
        #print(eval_q)
        act_cost = paddle.mean(-1.0 * eval_q)

        this_policy = self.model.policy(obs_n[i])
        # when continuous, 'this_policy' will be a tuple with two element: (mean, std)
        if self.continuous_actions:
            this_policy = paddle.concat(this_policy, axis=-1)
        act_reg = paddle.mean(paddle.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        self.actor_optimizer.clear_grad()
        cost.backward()
        self.actor_optimizer.step()
        return cost

    def _critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q(obs_n, act_n) # pred_q是shape为1024的张量，因为batch_size是1024，一次取一个batch_size的数据
        #print(pred_q)
        #print(type(pred_q))
        cost = paddle.mean(F.square_error_cost(pred_q, target_q))  # mean函数的参数axis=None,所以对所有元素计算平均值
        self.critic_optimizer.clear_grad()   # 不清空的话会一直累加
        cost.backward()
        self.critic_optimizer.step()   # 来更新参数
        return cost

    def sync_target(self, decay=None):
        """ update the target network with the training network
        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
