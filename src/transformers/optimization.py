# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch optimization for BERT model."""
import numpy as np
import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version

from .eva_backend import _TorchBackend, broadcast_optimizer_state
from .eva_utils import get_vector_a, get_vector_g

logger = logging.get_logger(__name__)


def _get_constant_lambda(_=None):
    return 1


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


def get_reduce_on_plateau_schedule(optimizer: Optimizer):
    """
    Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.

    Return:
        `torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.
    """

    return ReduceLROnPlateau(optimizer)


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT or name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

class DistributedOptimizer(torch.optim.AdamW):
    def __init__(self, params,lr=0.001,betas: Tuple[float, float] = (0.9, 0.999),eps: float = 1e-6,weight_decay: float = 0.0,named_parameters=None, rank=4, threshold=25):
        r"""Distributed ACP-SGD optimizer with WFBP and tensor fusion. 
        Args:
            params: optimizer parameters. 
            named_parameters: A mapping between parameter names and values. 
            rank: rank size for power iteration. 
            threshold: buffer size threshold (in MB) for tensor fusion.
        """
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(self.__class__, self).__init__(params,lr)
        self._rank = rank
        self._threshold = threshold
        self._num_steps = 0
        self._compute_p = True  # compute P or Q
        self._grad_accs = []
        self._target_modules=["query","key","value","dense"]
        # generate backend (Torch)
        try:
            print(f"world size:{torch.distributed.get_world_size()}")
        except:
            return RuntimeError('Torch.distributed much be init before create TorchBackend.')
        
        self.backend=_TorchBackend()

        # parameter names
        if named_parameters is not None:
            self._param_names = {param : name for name,param in named_parameters.items()}
        else:
            self._param_names = {v: 'param.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self._register_hooks()
        self._generate_groups_with_threshold()
        self._streams = {}
        
        print(self.backend)

        if torch.cuda.is_available() and self.backend.size() > 1:
            # Stream for grad reduction in the backward pass.  
            self._streams["reduce"] = torch.cuda.Stream()

    def _if_target(self,name):
        for module in self._target_modules:
            if(module in name):
                return True
        return False
    
    def _generate_groups_with_threshold(self):
        """
        Generate groups with buffer size threshold (in MB) for tensor fusion. 
        """
        p_size = []
        q_size = []
        model_size = []
        self._param_errors = {} # error-feedback for compressed gradients
        for p in self._register_parameters[::-1]:
            p_name = self._param_names[p]
            if p.ndimension() > 1: # compressed tensors
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape)) # ensure r<=min(n, m)
                self._param_errors[p_name] = torch.zeros_like(M)
                p_size.append(M.shape[0] * r * 4/1024/1024) #MB
                q_size.append(M.shape[1] * r * 4/1024/1024)
                model_size.append(p.data.numel() * 4/1024/1024)
            else: # uncompressed tensors
                p_size.append(p.data.numel() * 4/1024/1024)
                q_size.append(p.data.numel() * 4/1024/1024)
                model_size.append(p.data.numel() * 4/1024/1024)
        if self.backend.rank() == 0: 
            print('Memory (MB) Grad: %.2f, P: %.2f, Q: %.2f'%(sum(model_size), sum(p_size), sum(q_size))) 
        
        # compressed buffer size
        threshold_p = self._threshold * sum(p_size) / sum(model_size)
        threshold_q = self._threshold * sum(q_size) / sum(model_size)

        # Generate P Groups e.g. [[p3, p2], [p1]]
        p_groups = []
        current_group = []
        tot_size = 0
        for i, p in enumerate(self._register_parameters[::-1]):
            ps = p_size[i]
            if tot_size == 0 or tot_size + ps <= threshold_p:
                current_group.append(p)
                tot_size += ps
            else:
                p_groups.append(current_group)
                current_group = [p]
                tot_size = ps
        if len(current_group) > 0:
            p_groups.append(current_group)
        self._prepare_tensor_fusion_p(p_groups)

        # Generate Q Groups e.g. [[p3, p2], [p1]]
        q_groups = []
        current_group = []
        tot_size = 0
        for i, p in enumerate(self._register_parameters[::-1]):
            qs = q_size[i]
            if tot_size == 0 or tot_size + qs <= threshold_q:
                current_group.append(p)
                tot_size += qs
            else:
                q_groups.append(current_group)
                current_group = [p]
                tot_size = qs
        if len(current_group) > 0:
            q_groups.append(current_group)
        self._prepare_tensor_fusion_q(q_groups)      


    def _prepare_tensor_fusion_p(self, p_groups):
        """
        Prepare tensor fusion based on groups. 
        """
        self._buffers_p = []            # P buffers
        self._param_ps = {}             # get P by name
        self._param_group_idx_p = {}    # get (group id, sub id) by name
        
        for group_idx, p_group in enumerate(p_groups):
            for sub_idx, p in enumerate(p_group):
                p_name = self._param_names[p]
                self._param_group_idx_p[p_name] = (group_idx, sub_idx)
            buffer_p = self._get_buffer_p(p_group)
            self._buffers_p.append(buffer_p)

        self._param_group_flags_p = [[False]*len(g) for g in p_groups] # check whether param group is ready

        if self.backend.rank() == 0: 
            print('P Buffer sizes (MB):', 
                    ', '.join('{:.2f}'.format(buf.numel()*4/1024/1024) for buf in self._buffers_p))

    def _get_buffer_p(self, p_group):
        # buffer initialization
        start_p = 0
        for p in p_group:
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                start_p += M.shape[0] * r
            else:
                start_p += p.data.numel()
        buffer_p = torch.randn(start_p, device=p.device) # check: set seed

        # param_ps mapping
        start_p = 0
        for p in p_group:
            p_name = self._param_names[p]
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                ps = M.shape[0] * r
                P = buffer_p[start_p:start_p+ps].view(M.shape[0], r)
                self._param_ps[p_name] = P
                start_p += ps
            else:
                ps = p.data.numel()
                P = buffer_p[start_p:start_p+ps].view(p.data.shape)
                self._param_ps[p_name] = P
                start_p += ps
        return buffer_p

    def _prepare_tensor_fusion_q(self, q_groups):
        """
        Prepare tensor fusion based on groups. 
        """
        self._buffers_q = []            # Q buffers
        self._param_qs = {}             # get Q by name
        self._param_group_idx_q = {}    # get (group id, sub id) by name
        
        for group_idx, q_group in enumerate(q_groups):
            for sub_idx, p in enumerate(q_group):
                p_name = self._param_names[p]
                self._param_group_idx_q[p_name] = (group_idx, sub_idx)
            buffer_q = self._get_buffer_q(q_group)
            self._buffers_q.append(buffer_q)

        self._param_group_flags_q = [[False]*len(g) for g in q_groups] # check whether param group is ready

        if self.backend.rank() == 0: 
            print('Q Buffer sizes (MB):', 
                    ', '.join('{:.2f}'.format(buf.numel()*4/1024/1024) for buf in self._buffers_q))

    def _get_buffer_q(self, q_group):
        # buffer initialization
        start_p = 0
        for p in q_group:
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                start_p += M.shape[1] * r
            else:
                start_p += p.data.numel()
        buffer_q = torch.randn(start_p, device=p.device) # check: set seed

        # param_ps mapping
        start_p = 0
        for p in q_group:
            p_name = self._param_names[p]
            if p.ndimension() > 1:
                M = p.view(p.shape[0], -1)
                r = min(self._rank, min(M.shape))
                qs = M.shape[1] * r
                Q = buffer_q[start_p:start_p+qs].view(M.shape[1], r)
                self._param_qs[p_name] = Q
                start_p += qs
            else:
                qs = p.data.numel()
                Q = buffer_q[start_p:start_p+qs].view(p.data.shape)
                self._param_qs[p_name] = Q
                start_p += qs
        return buffer_q

    def _register_hooks(self):
        """
        Register hooks. 
        """
        self._register_parameters = []
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    if(len(p.shape)==2 and self._if_target(self._param_names[p])):
                        p.grad = p.data.new(p.size()).zero_()
                        p_tmp = p.expand_as(p)
                        grad_acc = p_tmp.grad_fn.next_functions[0][0]
                        grad_acc.register_hook(self._make_hook(p))
                        self._grad_accs.append(grad_acc)
                        self._register_parameters.append(p)

    def _make_hook(self, p):
        """
        Add hooks for backward propagation. 
        """
        def hook(*ignore):
            assert not p.grad.requires_grad
            name = self._param_names.get(p)
            # get grad, and previous P, Q, Error-feedback
            tensor = p.grad.data
            P = self._param_ps.get(name)
            Q = self._param_qs.get(name)
            E = self._param_errors.get(name)

            if E is not None: # update compressed tensor in the buffer
                self._alternative_compress(name, tensor, P, Q, E)
            else: # update uncompressed tensor in the buffer
                if self._compute_p:
                    P.copy_(tensor)
                else:
                    Q.copy_(tensor)
            
            # check whether buffer is ready to call an all-reduce
            new_name, buffer = self._buffer_is_ready(name)

            if buffer is not None and self.backend.size() > 1: 
                self._streams["reduce"].wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._streams["reduce"]):
                    self.backend.allreduce(buffer)
        return hook

    def _alternative_compress(self, name, tensor, P, Q, E):
        with torch.no_grad():
            if(len(tensor.shape)==1):
                return
            # M = tensor.view(tensor.shape[0], -1)
            M = tensor
            if self._compute_p: 
                if(Q.shape[0]<Q.shape[1]):
                    Q=Q.T()
                Q.copy_(torch.linalg.qr(Q).Q)
                P.copy_((M + E) @ Q)
                E.add_(M - P @ Q.T)
            else:
                if(P.shape[0]<P.shape[1]):
                    P=P.T()
                P.copy_(torch.linalg.qr(P).Q)
                Q.copy_((M + E).T @ P)
                E.add_(M - P @ Q.T)

    def _buffer_is_ready(self, name):
        """
        Check whether buffer is ready to call an all-reduce.
        """
        with torch.no_grad():
            if self._compute_p:
                group_idx, sub_idx = self._param_group_idx_p[name]
                self._param_group_flags_p[group_idx][sub_idx] = True
                for flag in self._param_group_flags_p[group_idx]:
                    if not flag: # not ready
                        return name, None
                buffer = self._buffers_p[group_idx]
                comm_name = 'reduce-p-group-%d' % group_idx
                return comm_name, buffer
            else:
                group_idx, sub_idx = self._param_group_idx_q[name]
                self._param_group_flags_q[group_idx][sub_idx] = True
                for flag in self._param_group_flags_q[group_idx]:
                    if not flag: # not ready
                        return name, None
                buffer = self._buffers_q[group_idx]
                comm_name = 'reduce-q-group-%d' % group_idx
                return comm_name, buffer

    def _bp_barrier(self):
        """
        Synchronize the all-reduce operations.
        """
        if self.backend.size() > 1:
            torch.cuda.current_stream().wait_stream(self._streams["reduce"])

        # approximate gradient
        for p in self._register_parameters:
            if p.ndimension() > 1: 
                name = self._param_names.get(p)
                # get P and Q
                P = self._param_ps.get(name)
                Q = self._param_qs.get(name)
                p.grad.data.view(p.grad.shape[0], -1).copy_(P @ Q.T)
            else:
                name = self._param_names.get(p)
                bias = self._param_ps.get(name) if self._compute_p else self._param_qs.get(name)
                p.grad.data.copy_(bias)
            p.grad.data.div_(self.backend.size())
        
        # clear flags
        self._param_group_flags_p = [[False]*len(g) for g in self._param_group_flags_p]
        self._param_group_flags_q = [[False]*len(g) for g in self._param_group_flags_q]
        self._compute_p = not self._compute_p


    def step(self, closure=None): 
        """
        Performs a single optimization step.
        """
        self._bp_barrier()
        self._num_steps += 1
        return super(self.__class__, self).step(closure)


class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://arxiv.org/abs/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`Tuple[float, float]`, *optional*, defaults to (1e-30, 1e-3)):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://arxiv.org/abs/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```"""

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss

class KFAC(torch.optim.SGD):
    """Accelerate Distributed K-FAC with Sublinear Memory Cost
    Args:
      training_model (nn): Torch model
      lr (float): learning rate (default: 0.1)
      damping (float): Tikhonov damping parameter (default: 0.03)
      kl_clip (float): clipping parameter for gradient scaling (kl_clip > 0: kl-clip, kl_clip = 0: re-scale, kl-clip < 0: None)
      factor_decay (float): running average coefficient for KVs
      exclude_vocabulary_size: exclude the pre-softmax linear layer in the Transformer
      hook_enabled (bool): enable the hook events to save the immediate states (a and g)
      exclude_parts='': exclude CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor for time breakdowns
    """
    def __init__(self,
                 model,
                 lr=1e-5,
                 damping=0.03,
                 fac_update_freq=1,
                 kfac_update_freq=1,
                 kfac_batch_size=16,
                 kl_clip=0.001,
                 factor_decay=0.95,
                 exclude_vocabulary_size=None,
                 hook_enabled=True,
                 exclude_parts=''):

        # For compatibility with `KFACParamScheduler`
        defaults = dict(lr=lr,
                        damping=damping,
                        fac_update_freq=fac_update_freq,
                        kfac_update_freq=kfac_update_freq) 

        super(KFAC, self).__init__(model.parameters(), lr)
        self.lr = lr
        self.damping = damping
        self.kfac_update_freq = kfac_update_freq
        self.fac_update_freq = fac_update_freq
        self.kfac_batch_size = kfac_batch_size
        self.kl_clip = kl_clip if (kl_clip is not None and kl_clip >= 0) else None
        self.factor_decay = factor_decay
        self.exclude_vocabulary_size = exclude_vocabulary_size
        self.hook_enabled = hook_enabled
        
        # generate backend (Torch)
        try:
            print(f"world size:{torch.distributed.get_world_size()}")
        except:
            return RuntimeError('Torch.distributed much be init before create TorchBackend.')
        
        self.backend=_TorchBackend()
        print(self.backend)
        # register hooks
        self.modules = []
        self.module_names = []
        self._register_module_hooks(model)

        # dictionaries keyed by `module` to storing KFs, inverse KFs, etc
        self.m_a, self.m_g = {}, {}
        self.handles = []
        
        # scheduling results
        self.module_ranks = None

        self.steps = 0

        
    ### Register hooks
    def set_hook_enabled(self, mode=True):
        self.hook_enabled = mode

    def _forward_hook_event(self, module, input):
        """Default: hook for saving input (a)"""
        # print(len(input))
        if self.hook_enabled and torch.is_grad_enabled() and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_a(input[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_a:
                    self.m_a[module] = new
                else:
                    #self.m_a[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_a[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_a[module].mul_(1-xi).add_(new, alpha=xi)
            if self.backend.size() > 1:
                self.handles.append(self.backend.allreduce_async_(self.m_a[module], op=self.backend.Average))

    def _backward_hook_event(self, module, grad_input, grad_output):
        """Default: hook for saving gradient w.r.t output (g)"""
        if self.hook_enabled and self.steps % self.fac_update_freq == 0:
            with torch.no_grad():
                new = get_vector_g(grad_output[0].data[0:self.kfac_batch_size], module)
                if module not in self.m_g:
                    self.m_g[module] = new
                else:
                    #self.m_g[module].mul_(self.factor_decay).add_(new, alpha=1-self.factor_decay)
                    self.m_g[module].mul_(1-self.factor_decay).add_(new, alpha=self.factor_decay)
                    #xi =  math.pow(self.steps+1, -self.factor_decay)
                    #self.m_g[module].mul_(1-xi).add_(new, alpha=xi)
            if self.backend.size() > 1:
                self.handles.append(self.backend.allreduce_async_(self.m_g[module], op=self.backend.Average))

    def _register_module_hooks(self, model):
        """Register forard/backward hooks to supported modules"""
        supported_modules = {'Linear', 'Conv2d'}
        name_idx = 0
        for module in model.modules():
            classname = module.__class__.__name__
            # print(classname)
            if classname in supported_modules:
                if self.exclude_vocabulary_size is not None and classname == 'Linear' and module.out_features == self.exclude_vocabulary_size:
                    continue # exclude the pre-softmax linear layer in the Transformer model
                self.modules.append(module)
                module.register_forward_pre_hook(self._forward_hook_event)
                # module.register_full_forward_hook(self._forward_hook_event)
                module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.4, and pytorch1.8 (full_backward_hook is not fired when its grad_input is None)
                #module.register_full_backward_hook(self._backward_hook_event)  # used in pytorch1.10
                module_name = 'module_name_%s_%d' % (classname, name_idx)
                self.module_names.append(module_name)
                name_idx += 1

        if self.backend.rank() == 0:
            logger.info("#register modules: %s", len(self.modules))

	### Precondition gradients
    def _precondition_grads(self):
        """Compute preconditioned gradients via Eva"""
        g_sum = 0
        v_sum = 0
        vg_sum = 0

        for module in self.modules:
            # get ma, mg, grad
            ma = self.m_a[module].view(-1, 1)
            mg = self.m_g[module].view(-1, 1)
            grad = self._get_grad(module)
            
            # print(grad)

            #if self.backend.rank() == 0:
            #    logger.info("mg: %s" % (mg))
            
            # compute intermediate states
            a = (ma.T @ ma).item()
            g = (mg.T @ mg).item()
            ag = (mg.T @ grad @ ma).item()
            
            #if self.backend.rank() == 0 and self.steps % 60 == 0:
            #    logger.info("a: %f, g: %f, ag: %f" % (a, g, ag))
            #    logger.info("beta: %f", ag/(a * g + self.damping))

            # compute preconditioned grads
            v = (mg @ ma.T).mul_(-ag/(a * g + self.damping))
            v.add_(grad)
            v.div_(self.damping)

            # weight and bias
            if module.bias is not None:
                weight = v[:, :-1].view(module.weight.grad.data.size())
                bias = v[:, -1:].view(module.bias.grad.data.size())
                # transform preconditioned gradient into gradient scale
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                        vg_sum += (bias * module.bias.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        v_sum += (bias * bias).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                        g_sum += (module.bias.grad.data * module.bias.grad.data).sum().item()

                # copy
                # print(weight)
                module.weight.grad.data.copy_(weight)
                module.bias.grad.data.copy_(bias)
                del grad
            else:
                weight = v.view(module.weight.grad.data.size())
                if self.kl_clip is not None:
                    if self.kl_clip > 0:
                        vg_sum += (weight * module.weight.grad.data * self.lr ** 2).sum().item()
                    else:
                        v_sum += (weight * weight).sum().item()
                        g_sum += (module.weight.grad.data * module.weight.grad.data).sum().item()
                # copy
                # print(weight)
                module.weight.grad.data.copy_(weight)
            del v

        # scale preconditioned gradient
        if self.kl_clip is not None:
            if self.kl_clip > 0: # kl-clip
                nu = min(1.0, math.sqrt(self.kl_clip / vg_sum)) if vg_sum > 0 else 1.0
            else: # re-scale
                nu = math.sqrt(g_sum / v_sum)
            # print(f"nu: {nu}")
            for module in self.modules:
                module.weight.grad.data.mul_(nu)
                if module.bias is not None:
                    module.bias.grad.data.mul_(nu)

    def _get_grad(self, module):
        """Get gradient with shape [output_dim, input_dim] for module"""
        if module.__class__.__name__ == 'Conv2d':
            # n_filters * (in_c * kw * kh)
            grad = module.weight.grad.data.view(module.weight.grad.data.size(0), -1)
        else:
            grad = module.weight.grad.data
        if module.bias is not None:
            grad = torch.cat([grad, module.bias.grad.data.view(-1, 1)], 1)
        return grad    


    ### Perform one K-FAC step
    @torch.no_grad()
    def step(self, closure=None, epoch=None):
        """Perform one K-FAC step"""

        # update params, used for compatibilty with `KFACParamScheduler`
        # print(self.param_groups)
        # group = self.param_groups[0]
        # self.lr = group['lr']
        # self.damping = group['damping']
        # self.fac_update_freq = group['fac_update_freq']
        # self.kfac_update_freq = group['kfac_update_freq']

        if self.steps % self.fac_update_freq == 0 and self.backend.size() > 1:
            for handle in self.handles:
                self.backend.synchronize(handle)
            self.handles = []

        # print("all synchronized")

        self._precondition_grads()

        self.steps += 1
        super().step()


class AdafactorSchedule(LambdaLR):
    """
    Since [`~optimization.Adafactor`] performs its own scheduling, if the training loop relies on a scheduler (e.g.,
    for logging), this class creates a proxy object that retrieves the current lr values from the optimizer.

    It returns `initial_lr` during startup and the actual `lr` during stepping.
    """

    def __init__(self, optimizer, initial_lr=0.0):
        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    """
    Get a proxy schedule for [`~optimization.Adafactor`]

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        initial_lr (`float`, *optional*, defaults to 0.0):
            Initial lr

    Return:
        [`~optimization.Adafactor`] proxy schedule object.


    """
    return AdafactorSchedule(optimizer, initial_lr)
