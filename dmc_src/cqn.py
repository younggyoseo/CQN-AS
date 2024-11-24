"""
Implementation of CQN that works on DeepMind Control Suite.
The difference to CQN used in RLBench:
- (i) Does not support low-dimensional inputs
- (ii) Does not support behavior cloning objective
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

import utils
from cqn_utils import (
    random_action_if_within_delta,
    zoom_in,
    encode_action,
    decode_action,
)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=False),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.GroupNorm(1, 32),
            nn.SiLU(inplace=False),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._actor_dim = action_shape[0]
        self._bins = bins

        # Advantage stream in Dueling network
        self.adv_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.adv_net = nn.Sequential(
            nn.Linear(feature_dim + self._actor_dim + levels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.adv_head = nn.Linear(hidden_dim, self._actor_dim * bins * atoms)
        self.adv_output_shape = (self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        self.value_trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim + self._actor_dim + levels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=False),
        )
        self.value_head = nn.Linear(hidden_dim, self._actor_dim * 1 * atoms)
        self.value_output_shape = (self._actor_dim, 1, atoms)

        self.apply(utils.weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def forward_each_level(
        self, level: int, obs: torch.Tensor, prev_action: torch.Tensor
    ):
        """
        Implementation that processes forward step for each level

        Inputs:
        - obs: [B, D] shaped tensor for value_h and adv_h, shared for all levels
        - prev_actions: [B, D] shaped prev action for the current level

        Outputs:
        - q_logits: [B, action_dimensions, bins, atoms]
        """
        level_id = (
            torch.eye(self._levels, device=obs.device, dtype=obs.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(obs.shape[0], 0)
        )

        value_h = self.value_trunk(obs)
        value_x = torch.cat([value_h, prev_action, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            -1, *self.value_output_shape
        )

        adv_h = self.adv_trunk(obs)
        adv_x = torch.cat([adv_h, prev_action, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(-1, *self.adv_output_shape)
        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits

    def forward(self, obs: torch.Tensor, prev_actions: torch.Tensor):
        """
        Optimized implementation that processes forward step for all the levels at once
        This is possible because we have pre-computed prev_actions for all the levels
        when we are computing Q(s,a) for given action a.
        But this is not possible when we want to get actions for given s, as we have to
        compute action for each level. In this case, use `forward_each_level`

        Inputs:
        - obs: [B, D] shaped tensor for value_h and adv_h, shared for all levels
        - prev_actions: [B, L, D]

        Outputs:
        - q_logits: [B, L, action_dimensions, bins, atoms]
        """
        device, dtype = obs.device, obs.dtype
        B, L = prev_actions.shape[:2]
        # Reshape previous actions
        prev_actions = prev_actions.view(B, L, self._actor_dim)  # [B, L, T, D]

        # level id - [L, L] -> [B, L, L]
        level_id = torch.eye(L, device=device, dtype=dtype)[None, :, :].repeat(B, 1, 1)

        adv_h = self.adv_trunk(obs)
        adv_h = adv_h[:, None, :].repeat(1, L, 1)
        adv_x = torch.cat([adv_h, prev_actions, level_id], -1)
        advs = self.adv_head(self.adv_net(adv_x)).view(B, L, *self.adv_output_shape)

        value_h = self.value_trunk(obs)
        value_h = value_h[:, None, :].repeat(1, L, 1)
        value_x = torch.cat([value_h, prev_actions, level_id], -1)
        values = self.value_head(self.value_net(value_x)).view(
            B, L, *self.value_output_shape
        )

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        repr_dim: int,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        actor_dim = action_shape[0]
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms), requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.network = C2FCriticNetwork(
            repr_dim, action_shape, feature_dim, hidden_dim, levels, bins, atoms
        )

    def get_action(self, obs: torch.Tensor):
        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        for level in range(self.levels):
            q_logits = self.network.forward_each_level(level, obs, (low + high) / 2)
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            obs: [B, F] shaped feature tensor
            continuous_action: [B, D] shaped action tensor

        Return:
            q_probs: [B, L, D, bins, atoms] for value distribution at all bins
            q_probs_a: [B, L, D, atoms] for value distribution at given bin
            log_q_probs: [B, L, D, bins, atoms] with log probabilities
            log_q_probs_a: [B, L, D, atoms] with log probabilities
        """

        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(obs.shape[0], 1).detach()
        high = self.initial_high.repeat(obs.shape[0], 1).detach()

        # Pre-compute previous actions for all the levels
        prev_actions = []
        for level in range(self.levels):
            prev_actions.append((low + high) / 2)
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]
            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_logits_all = self.network(obs, torch.stack(prev_actions, 1))
        for level in range(self.levels):
            q_logits = q_logits_all[:, level]
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # (Log) Probs [..., D, bins, atoms]
            # (Log) Probs_a [..., D, atoms]
            q_probs = F.softmax(q_logits, 3)  # [B, D, bins, atoms]
            q_probs_a = torch.gather(
                q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            q_probs_a = q_probs_a[..., 0, :]  # [B, D, atoms]

            log_q_probs = F.log_softmax(q_logits, 3)  # [B, D, bins, atoms]
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            log_q_probs_a = log_q_probs_a[..., 0, :]  # [B, D, atoms]

            q_probs_per_level.append(q_probs)
            q_probs_a_per_level.append(q_probs_a)
            log_q_probs_per_level.append(log_q_probs)
            log_q_probs_a_per_level.append(log_q_probs_a)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_probs = torch.stack(q_probs_per_level, -4)  # [B, L, D, bins, atoms]
        q_probs_a = torch.stack(q_probs_a_per_level, -3)  # [B, L, D, atoms]
        log_q_probs = torch.stack(log_q_probs_per_level, -4)
        log_q_probs_a = torch.stack(log_q_probs_a_per_level, -3)
        return q_probs, q_probs_a, log_q_probs, log_q_probs_a

    def compute_target_q_dist(
        self,
        next_obs: torch.Tensor,
        next_continuous_action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_obs: [B, F] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, atoms] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(next_obs, next_continuous_action)[1]

        shape = next_q_probs_a.shape  # [B, L, D, atoms]
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        # Compute Tz for [B, atoms]
        Tz = reward + discount * self.support.unsqueeze(0).detach()
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.v_min) / self.delta_z
        # Mask for conditions
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        lower_mask = (upper > 0) & (lower == upper)
        upper_mask = (lower < (self.atoms - 1)) & (lower == upper)
        # Apply masks separately
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)
        # Repeat Tz for (L * D) times -> [B * L * D, atoms]
        multiplier = batch_size // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        # Distribute probability of Tz
        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )
        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )  # m_u = m_u + p(s_t+n, a*)(b - l)

        m = m.view(*shape)  # [B, L, D, atoms]
        return m

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


class CQNAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        levels,
        bins,
        atoms,
        v_min,
        v_max,
        critic_target_tau,
        critic_target_interval,
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

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.critic = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(device)
        self.critic_target = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
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

    def update_critic(self, obs, action, reward, discount, next_obs):
        with torch.no_grad():
            next_action = self.critic.get_action(next_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_obs, next_action, reward, discount
            )

        # Cross entropy loss for C51
        log_q_probs_a = self.critic(obs, action)[3]
        critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return TensorDict(critic_loss=critic_loss.detach())

    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]
        next_obs = batch["next_obs"]

        # Augmentation
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

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
