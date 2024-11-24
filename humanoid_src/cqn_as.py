from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensordict import TensorDict

import utils
from cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
)


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        low_dim: int,
        action_shape: Tuple,
        hidden_dim: int,
        gru_layers: int,
        levels: int,
        bins: int,
    ):
        super().__init__()
        self._levels = levels
        self._action_sequence, self._actor_dim = action_shape
        self._bins = bins

        self.net = nn.Sequential(
            nn.Linear(
                low_dim + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Linear(
            hidden_dim,
            self._actor_dim * bins,
        )
        self.output_shape = (self._action_sequence * self._actor_dim, bins)

        self.apply(utils.weight_init)
        self.head.weight.data.fill_(0.0)
        self.head.bias.data.fill_(0.0)

    def forward_each_level(
        self,
        level: int,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
    ):
        """
        Implementation to compute Q-values at each level.

        Inputs:
        - level: level index (integer, not one-hot)
        - obs: low-dimensional observations
        - prev_actions: actions from *all* previous levels

        Outputs:
        - q_values: (batch_size, level, action_sequence * actor_dim, bins)
        """
        h = obs

        level_id = (
            torch.eye(self._levels, device=h.device, dtype=h.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(h.shape[0], 0)
        )
        level_id = level_id.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        prev_action = prev_action.view(
            -1, self._action_sequence, self._actor_dim
        )  # [B, T, D]
        action_sequence_id = (
            torch.eye(self._action_sequence, device=h.device, dtype=h.dtype)
            .unsqueeze(0)
            .repeat_interleave(h.shape[0], 0)
        )  # [B, T, T]

        h = h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        x = torch.cat([h, prev_action, action_sequence_id, level_id], -1)  # [B, T, D]
        # Process through MLP for each action sequence step
        feats = self.net(x)
        # Process through GRU
        feats, _ = self.gru(feats)
        q_values = self.head(feats).view(-1, *self.output_shape)

        return q_values

    def forward(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
    ):
        """
        Optimized implementation to compute Q-values at all levels in parallel.
        See `forward_each_level` for implementation that processes each level.

        Inputs:
        - obs: low-dimensional observations
        - prev_actions: actions from *all* previous levels

        Outputs:
        - q_values: (batch_size, level, action_sequence * actor_dim, bins)
        """
        device, dtype = obs.device, obs.dtype
        levels = prev_actions.size(1)
        B, L, T = prev_actions.size(0), levels, self._action_sequence

        # Reshape previous actions
        prev_actions = prev_actions.view(-1, L, T, self._actor_dim)  # [B, L, T, D]

        # Action sequence id - [T, T] -> [B, L, T, T]
        action_sequence_id = torch.eye(T, device=device, dtype=dtype)[
            None, None, :, :
        ].repeat(B, L, 1, 1)

        # level id - [L, L] -> [B, L, T, L]
        level_id = torch.eye(L, device=device, dtype=dtype)[None, :, None, :].repeat(
            B, 1, T, 1
        )

        obs = obs[:, None, None, :].repeat(1, L, T, 1)
        x = torch.cat([obs, prev_actions, action_sequence_id, level_id], -1)
        feats = self.net(x)
        # Process through GRU
        feats = feats.view(B * L, T, -1)
        feats = self.gru(feats)[0]
        q_values = self.head(feats).view(B, L, *self.output_shape)
        return q_values


class C2FCritic(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        low_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        gru_layers: int,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        actor_dim = action_shape[0] * action_shape[1]  # action_sequence * action_dim
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )

        self.network = C2FCriticNetwork(
            low_dim,
            action_shape,
            hidden_dim,
            gru_layers,
            levels,
            bins,
        )

    def get_action(self, obs: torch.Tensor):
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            qs = self.network.forward_each_level(level, obs, (low + high) / 2)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)
        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        qs_per_level = []
        qs_a_per_level = []

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        prev_actions = []
        for level in range(self.levels):
            prev_actions.append((low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]
            low, high = zoom_in(low, high, argmax_q, self.bins)

        qs_all = self.network(obs, torch.stack(prev_actions, 1))
        for level in range(self.levels):
            qs = qs_all[:, level]
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # qs: [B, D, bins]
            # qs_a: [B, D]
            qs_a = torch.gather(qs, dim=-1, index=argmax_q.unsqueeze(-1))[..., 0]

            qs_per_level.append(qs)
            qs_a_per_level.append(qs_a)

        qs = torch.stack(qs_per_level, -3)
        qs_a = torch.stack(qs_a_per_level, -2)
        return qs, qs_a

    def encode_decode_action(self, continuous_action: torch.Tensor):
        """Encode and decode actions"""
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        return continuous_action


class CQNASAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        levels,
        bins,
        gru_layers,
        critic_target_tau,
        critic_target_interval,
        weight_decay,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.critic_target_interval = critic_target_interval
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule

        self.critic = C2FCritic(
            action_shape,
            obs_shape[-1],
            hidden_dim,
            levels,
            bins,
            gru_layers,
        ).to(self.device)
        self.critic_target = C2FCritic(
            action_shape,
            obs_shape[-1],
            hidden_dim,
            levels,
            bins,
            gru_layers,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.train()
        self.critic_target.eval()

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.critic.get_action(obs)
        stddev = torch.ones_like(action) * stddev
        dist = utils.TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action.cpu().numpy()[0]

    def add_noise_to_action(self, action: np.array, step: int):
        if step < self.num_expl_steps:
            action = np.random.uniform(-1.0, 1.0, size=action.shape).astype(
                action.dtype
            )
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            action = np.clip(
                action
                + np.random.normal(0, stddev, size=action.shape).astype(action.dtype),
                -1.0,
                1.0,
            )
        return action

    def update_critic(self, obs, action, reward, discount, next_obs):
        with torch.no_grad():
            next_action = self.critic.get_action(next_obs)
            target_V = self.critic_target(next_obs, next_action)[1]
            target_Q = reward.unsqueeze(-1) + discount.unsqueeze(-1) * target_V

        # Cross entropy loss for C51
        Q = self.critic(obs, action)[1]
        critic_loss = F.mse_loss(Q, target_Q)

        # optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return TensorDict(critic_loss=critic_loss.detach())

    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]
        next_obs = batch["next_obs"]

        # update critic
        metrics = self.update_critic(obs, action, reward, discount, next_obs)
        metrics["batch_reward"] = reward.mean().detach()
        metrics["batch_discount"] = discount.mean().detach()
        return metrics

    def update_target_critic(self, step):
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )