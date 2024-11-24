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


class RandomShiftsAug(nn.Module):
    """
    Random shift augmentation for rgb observations
    """

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


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]
        self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded

        self.conv_nets = nn.ModuleList()
        for _ in range(self.num_views):
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                ImgChLayerNorm(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                ImgChLayerNorm(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                ImgChLayerNorm(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                ImgChLayerNorm(256),
                nn.SiLU(),
            )
            self.conv_nets.append(conv_net)

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor):
        # obs: [B, V, C, H, W]
        obs = obs / 255.0 - 0.5
        hs = []
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h


class C2FCriticNetwork(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
        gru_layers: int,
        rgb_encoder_layers: int,
        levels: int,
        bins: int,
        atoms: int,
    ):
        super().__init__()
        self._levels = levels
        self._action_sequence, self._actor_dim = action_shape
        self._bins = bins

        # Advantage stream in Dueling network
        ## RGB encoder for advantage stream
        adv_rgb_encoder_net = []
        input_dim = repr_dim
        for i in range(rgb_encoder_layers):
            adv_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        adv_rgb_encoder_net = adv_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.adv_rgb_encoder = nn.Sequential(*adv_rgb_encoder_net)

        ## Low-dimensional encoder for advantage stream
        self.adv_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        ## Main network for advantage stream
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.adv_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.adv_head = nn.Linear(
            hidden_dim,
            self._actor_dim * bins * atoms,
        )
        self.adv_output_shape = (self._action_sequence * self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        ## RGB encoder for advantage stream
        value_rgb_encoder_net = []
        input_dim = repr_dim
        for i in range(rgb_encoder_layers):
            value_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        value_rgb_encoder_net = value_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.value_rgb_encoder = nn.Sequential(*value_rgb_encoder_net)

        ## Low-dimensional encoder for advantage stream
        self.value_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        ## Main network for advantage stream
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.value_head = nn.Linear(
            hidden_dim,
            self._actor_dim * 1 * atoms,
        )
        self.value_output_shape = (self._action_sequence * self._actor_dim, 1, atoms)

        self.apply(utils.weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def encode(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        value_h = torch.cat(
            [self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs)],
            -1,
        )
        adv_h = torch.cat(
            [self.adv_rgb_encoder(rgb_obs), self.adv_low_dim_encoder(low_dim_obs)],
            -1,
        )
        return value_h, adv_h

    def forward_each_level(
        self,
        level: int,
        features: tuple[torch.Tensor, torch.Tensor],
        prev_action: torch.Tensor,
    ):
        """
        Implementation that processes forward step for each level

        Inputs:
        - features: ([B, D], [B, D]) for value_h and adv_h, shared for all levels
        - prev_actions: [B, D] shaped prev action for the current level

        Outputs:
        - q_logits: [B, action_sequence * action_dimensions, bins, atoms]
        """
        value_h, adv_h = features

        level_id = (
            torch.eye(self._levels, device=value_h.device, dtype=value_h.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )
        level_id = level_id.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        prev_action = prev_action.view(
            -1, self._action_sequence, self._actor_dim
        )  # [B, T, D]
        action_sequence_id = (
            torch.eye(self._action_sequence, device=value_h.device, dtype=value_h.dtype)
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )  # [B, T, T]

        # Value
        value_h = value_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        value_x = torch.cat(
            [value_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats, _ = self.value_gru(value_feats)
        values = self.value_head(value_feats).view(-1, *self.value_output_shape)

        # Advantage
        adv_h = adv_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        adv_x = torch.cat(
            [adv_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats, _ = self.adv_gru(adv_feats)
        advs = self.adv_head(adv_feats).view(-1, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        prev_actions: torch.Tensor,
    ):
        """
        Optimized implementation that processes forward step for all the levels at once
        This is possible because we have pre-computed prev_actions for all the levels
        when we are computing Q(s,a) for given action a.
        But this is not possible when we want to get actions for given s, as we have to
        compute action for each level. In this case, use `forward_each_level`

        Inputs:
        - rgb_obs: [B, D]
        - low_dim_obs: [B, D]
        - prev_actions: [B, L, D]

        Outputs:
        - q_logits: [B, L, action_sequence * action_dimensions, bins, atoms]
        """
        device, dtype = rgb_obs.device, rgb_obs.dtype
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

        # Encode features
        value_h, adv_h = self.encode(rgb_obs, low_dim_obs)

        # Value
        value_h = value_h[:, None, None, :].repeat(1, L, T, 1)
        value_x = torch.cat([value_h, prev_actions, action_sequence_id, level_id], -1)
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats = value_feats.view(B * L, T, -1)
        value_feats = self.value_gru(value_feats)[0]
        values = self.value_head(value_feats).view(B, L, *self.value_output_shape)

        # Advantage
        adv_h = adv_h[:, None, None, :].repeat(1, L, T, 1)
        adv_x = torch.cat([adv_h, prev_actions, action_sequence_id, level_id], -1)
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats = adv_feats.view(B * L, T, -1)
        adv_feats = self.adv_gru(adv_feats)[0]
        advs = self.adv_head(adv_feats).view(B, L, *self.adv_output_shape)

        q_logits = values + advs - advs.mean(-2, keepdim=True)
        return q_logits


class C2FCritic(nn.Module):
    def __init__(
        self,
        action_shape: tuple,
        repr_dim: int,
        low_dim: int,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
        gru_layers: int,
        rgb_encoder_layers: int,
        use_parallel_impl: bool,
    ):
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_parallel_impl = use_parallel_impl
        actor_dim = action_shape[0] * action_shape[1]  # action_sequence * action_dim
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
            repr_dim,
            low_dim,
            action_shape,
            feature_dim,
            hidden_dim,
            gru_layers,
            rgb_encoder_layers,
            levels,
            bins,
            atoms,
        )

    def get_action(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor):
        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            q_logits = self.network.forward_each_level(
                level, features, (low + high) / 2
            )
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ):
        """Compute value distributions for given obs and action.

        Args:
            rgb_obs: [B, repr_dim] shaped feature tensor
            low_dim_obs: [B, low_dim] shaped feature tensor
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

        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        if self.use_parallel_impl:
            # Pre-compute previous actions for all the levels
            prev_actions = []
            for level in range(self.levels):
                prev_actions.append((low + high) / 2)
                argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]
                low, high = zoom_in(low, high, argmax_q, self.bins)
            q_logits_all = self.network(rgb_obs, low_dim_obs, torch.stack(prev_actions, 1))
        else:
            features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            if self.use_parallel_impl:
                q_logits = q_logits_all[:, level]
            else:
                q_logits = self.network.forward_each_level(level, features, (low + high) / 2)
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
        next_rgb_obs: torch.Tensor,
        next_low_dim_obs: torch.Tensor,
        next_continuous_action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ):
        """Compute target distribution for distributional critic
        based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py implementation

        Args:
            next_rgb_obs: [B, repr_dim] shaped feature tensor
            next_low_dim_obs: [B, low_dim] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Return:
            m: [B, L, D, atoms] shaped tensor for value distribution
        """
        next_q_probs_a = self.forward(
            next_rgb_obs, next_low_dim_obs, next_continuous_action
        )[1]

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


class CQNASAgent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
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
        bc_lambda,
        bc_margin,
        gru_layers,
        rgb_encoder_layers,
        use_parallel_impl,
        critic_lambda,
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
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda

        # models
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)
        self.critic = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
            gru_layers,
            rgb_encoder_layers,
            use_parallel_impl,
        ).to(device)
        self.critic_target = C2FCritic(
            action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
            gru_layers,
            rgb_encoder_layers,
            use_parallel_impl,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

        print(self.encoder)
        print(self.critic)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        action = self.critic_target.get_action(
            rgb_obs, low_dim_obs
        )  # use critic_target
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

    def update_critic(
        self,
        rgb_obs,
        low_dim_obs,
        action,
        reward,
        discount,
        next_rgb_obs,
        next_low_dim_obs,
        demos,
    ):
        with torch.no_grad():
            next_action = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, reward, discount
            )

        # Cross entropy loss for C51
        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            rgb_obs, low_dim_obs, action
        )
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        demos = demos.float().squeeze(1)  # [B,]

        # BC - First-order stochastic dominance loss
        # q_probs: [B, L, D, bins, atoms], q_probs_a: [B, L, D, atoms]
        q_probs_cdf = torch.cumsum(q_probs, -1)
        q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
        # q_probs_{a_{i}} is stochastically dominant over q_probs_{a_{-i}}
        bc_fosd_loss = (
            (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
            .clamp(min=0)
            .sum(-1)
            .mean([-1, -2, -3])
        )
        bc_fosd_loss = (bc_fosd_loss * demos).sum() / demos.sum()
        critic_loss = critic_loss + self.bc_lambda * bc_fosd_loss

        # BC - Margin loss
        qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
        qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(-1)
        margin_loss = torch.clamp(
            self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
        ).mean([-1, -2, -3])
        margin_loss = (margin_loss * demos).sum() / demos.sum()
        critic_loss = critic_loss + self.bc_lambda * margin_loss

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return TensorDict(
            ratio_of_demos=demos.mean().detach(),
            q_critic_loss=q_critic_loss.detach(),
            bc_margin_loss=margin_loss.detach(),
            bc_fosd_loss=bc_fosd_loss.detach(),
        )

    def update(self, batch):
        rgb_obs = batch["rgb_obs"]
        low_dim_obs = batch["low_dim_obs"]
        action = batch["action"]
        reward = batch["reward"]
        discount = batch["discount"]
        next_rgb_obs = batch["next_rgb_obs"]
        next_low_dim_obs = batch["next_low_dim_obs"]
        demos = batch["demos"]

        # augment
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        # encode
        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        # update critic
        metrics = self.update_critic(
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
        )
        metrics["batch_reward"] = reward.mean().detach()
        return metrics

    def update_target_critic(self, step):
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
