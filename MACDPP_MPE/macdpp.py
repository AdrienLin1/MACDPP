import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
from maddpg import MADDPG
from parl.utils import ReplayMemory
from scipy.stats import truncnorm

__all__ = ['MACDPP']
beta = 20.0
alpha = 1.0
num = 30




class MACDPP(MADDPG):
    alpha = 1.0


    def _expand(self, action):

        action_batch = action.shape[0]

        tile_action = paddle.tile(action, [num, 1])
        sample = paddle.normal(mean=0, std=0.1, shape=[1, num])
        # sample = sample.clip(-0.5, 0.5)
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


    def mellow_max(self, q_vals):
        max_q_vals = paddle.max(q_vals, axis=0, keepdim=True)
        q_vals = q_vals - max_q_vals
        e_beta_Q = paddle.exp(beta * q_vals)

        sum_e_beta_Q = paddle.sum(e_beta_Q, 0) / (num + 1)
        max_q_vals = paddle.squeeze(max_q_vals)
        log_sum_Q = paddle.log(sum_e_beta_Q) + max_q_vals * beta

        softmax_q_vals = log_sum_Q/beta

        return softmax_q_vals
