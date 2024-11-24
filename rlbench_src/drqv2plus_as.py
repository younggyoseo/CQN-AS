from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


import utils


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
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.ReLU(inplace=False),
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


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._action_sequence, self._actor_dim = action_shape

        self.rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim * 2 + self._action_sequence, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.policy_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, self._actor_dim)
        self.apply(utils.weight_init)

    def forward(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor, std: float):
        """
        Inputs:
        - rgb_obs: features from visual encoder
        - low_dim_obs: low-dimensional observations

        Outputs:
        - dist: torch distribution for policy
        """
        rgb_h = self.rgb_encoder(rgb_obs)
        low_dim_h = self.low_dim_encoder(low_dim_obs)
        h = torch.cat([rgb_h, low_dim_h], -1)

        action_sequence_id = (
            torch.eye(
                self._action_sequence,
                device=low_dim_obs.device,
                dtype=low_dim_obs.dtype,
            )
            .unsqueeze(0)
            .repeat_interleave(low_dim_obs.shape[0], 0)
        )
        h = h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        h = torch.cat([h, action_sequence_id], -1)

        policy_feats = self.policy(h)
        policy_feats, _ = self.policy_gru(policy_feats)
        mu = self.policy_head(policy_feats).view(
            h.shape[0], self._action_sequence * self._actor_dim
        )
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: tuple,
        feature_dim: int,
        hidden_dim: int,
        out_shape: tuple,
    ):
        super().__init__()
        self._action_sequence, self._actor_dim = action_shape
        self._out_shape = out_shape
        out_dim = 1
        for s in out_shape:
            out_dim *= s

        self.rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.Q1 = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + self._action_sequence, hidden_dim
            ),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.Q1_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.Q1_head = nn.Linear(hidden_dim, out_dim // self._action_sequence)
        self.Q2 = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._actor_dim + self._action_sequence, hidden_dim
            ),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.Q2_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.Q2_head = nn.Linear(hidden_dim, out_dim // self._action_sequence)
        self.apply(utils.weight_init)

    def forward(
        self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor, actions: torch.Tensor
    ):
        """
        Inputs:
        - obs: features from visual encoder
        - low_dim_obs: low-dimensional observations
        - action: actions

        Outputs:
        - qs: (batch_size, 2)
        """
        actions = actions.view(-1, self._action_sequence, self._actor_dim)
        action_sequence_id = (
            torch.eye(
                self._action_sequence,
                device=low_dim_obs.device,
                dtype=low_dim_obs.dtype,
            )
            .unsqueeze(0)
            .repeat_interleave(low_dim_obs.shape[0], 0)
        )
        rgb_h = self.rgb_encoder(rgb_obs)
        low_dim_h = self.low_dim_encoder(low_dim_obs)

        rgb_h = rgb_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        low_dim_h = low_dim_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        h = torch.cat([rgb_h, low_dim_h, actions, action_sequence_id], -1)

        q1_feats = self.Q1(h)
        q1_feats, _ = self.Q1_gru(q1_feats)  # [B, T, D]
        q1 = self.Q1_head(q1_feats).view(h.shape[0], *self._out_shape)  # [B, T, 1]

        q2_feats = self.Q2(h)
        q2_feats, _ = self.Q2_gru(q2_feats)
        q2 = self.Q2_head(q2_feats).view(h.shape[0], *self._out_shape)  # [B, T, 1]

        qs = torch.cat([q1, q2], -1)
        return qs


class DistributionalCritic(Critic):
    def __init__(
        self,
        distributional_critic_limit: float,
        distributional_critic_atoms: int,
        distributional_critic_transform: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.limit = distributional_critic_limit
        self.atoms = distributional_critic_atoms
        self.transform = distributional_critic_transform

    def to_dist(self, qs):
        # qs: [B, T, 2]
        # qs_dist: [B, T, atoms, 2]
        B, T = qs.shape[:2]

        qs = qs.view(B * T, qs.size(-1))
        qs_dist = torch.cat(
            [
                utils.to_categorical(
                    qs[:, q_idx].unsqueeze(-1),
                    limit=self.limit,
                    num_atoms=self.atoms,
                    transformation=self.transform,
                )
                for q_idx in range(qs.size(-1))
            ],
            -1,
        )
        qs_dist = qs_dist.view(B, T, self.atoms, -1)
        return qs_dist

    def from_dist(self, qs_dist):
        # qs_dist: [B, T, atoms, 2]
        # qs: [B, T, 2]
        B, T = qs_dist.shape[:2]
        qs_dist = qs_dist.view(B * T, self.atoms, qs_dist.size(-1))
        qs = torch.cat(
            [
                utils.from_categorical(
                    qs_dist[..., q_idx],
                    limit=self.limit,
                    transformation=self.transform,
                )
                for q_idx in range(qs_dist.size(-1))
            ],
            dim=-1,
        )
        qs = qs.view(B, T, -1)
        return qs

    def compute_distributional_critic_loss(self, qs, target_qs):
        loss = 0.0
        for q_idx in range(qs.size(-1)):
            loss += -torch.sum(
                torch.log_softmax(qs[[..., q_idx]], -1)
                * target_qs.squeeze(-1).detach(),
                -1,
            )
        return loss.unsqueeze(-1)


class DrQV2Agent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        lr,
        weight_decay,
        feature_dim,
        hidden_dim,
        use_distributional_critic,
        distributional_critic_limit,
        distributional_critic_atoms,
        distributional_critic_transform,
        bc_lambda,
        critic_target_tau,
        critic_target_interval,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.critic_target_interval = critic_target_interval
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.bc_lambda = bc_lambda
        self.use_distributional_critic = use_distributional_critic
        self.distributional_critic_limit = distributional_critic_limit
        self.distributional_critic_atoms = distributional_critic_atoms
        self.distributional_critic_transform = distributional_critic_transform

        # models
        low_dim = low_dim_obs_shape[-1]
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)
        self.actor = Actor(
            self.encoder.repr_dim, low_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        action_sequence = action_shape[0]
        self._action_sequence = action_sequence
        if use_distributional_critic:
            self.critic = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(action_sequence, self.distributional_critic_atoms, 1),
            ).to(device)
            self.critic_target = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(action_sequence, self.distributional_critic_atoms, 1),
            ).to(device)
        else:
            self.critic = Critic(
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(
                    action_sequence,
                    1,
                ),
            ).to(device)
            self.critic_target = Critic(
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(
                    action_sequence,
                    1,
                ),
            ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay
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
        self.actor.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(rgb_obs, low_dim_obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
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
    ):
        with torch.no_grad():
            dist = self.actor(next_rgb_obs, next_low_dim_obs, self.stddev_schedule)
            next_action = dist.sample(clip=self.stddev_clip)
            target_qs = self.critic_target(next_rgb_obs, next_low_dim_obs, next_action)
            if self.use_distributional_critic:
                target_qs = self.critic_target.from_dist(target_qs)
            target_Q1, target_Q2 = target_qs[..., 0], target_qs[..., 1]
            target_V = torch.min(target_Q1, target_Q2).unsqueeze(2)  # [B, T, 1]
            reward = reward.unsqueeze(2)  # [B, 1, 1]
            discount = discount.unsqueeze(2)  # [B, 1, 1]
            target_Q = reward + (discount * target_V)  # [B, T, 1]
            if self.use_distributional_critic:
                target_Q = self.critic_target.to_dist(target_Q)

        qs = self.critic(rgb_obs, low_dim_obs, action)

        if self.use_distributional_critic:
            critic_loss = self.critic.compute_distributional_critic_loss(
                qs, target_Q
            ).mean()
        else:
            Q1, Q2 = qs[..., 0], qs[..., 1]  # [B, T]
            target_Q = target_Q.squeeze(2)
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return TensorDict(critic_loss=critic_loss.detach())

    def update_actor(self, rgb_obs, low_dim_obs, demo_action, demos):
        dist = self.actor(rgb_obs, low_dim_obs, self.stddev_schedule)
        action = dist.sample(clip=self.stddev_clip)
        qs = self.critic(rgb_obs, low_dim_obs, action)
        if self.use_distributional_critic:
            qs = self.critic.from_dist(qs)
        Q1, Q2 = qs[..., 0], qs[..., 1]
        Q = torch.min(Q1, Q2)

        base_actor_loss = -Q.mean()

        bc_loss = self.get_bc_loss(dist.mean, demo_action, demos)
        actor_loss = base_actor_loss + self.bc_lambda * bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return TensorDict(
            actor_loss=base_actor_loss.mean().detach(),
        )

    def get_bc_loss(self, predicted_action, buffer_action, demos):
        # Only apply loss to demo items
        demos = demos.float()
        bs = demos.shape[0]
        bc_loss = (
            F.mse_loss(
                predicted_action.view(bs, -1),
                buffer_action.view(bs, -1),
                reduction="none",
            )
            * demos
        )
        bc_loss = bc_loss.sum() / demos.sum()
        return bc_loss

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

        critic_metrics = self.update_critic(
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
        )
        actor_metrics = self.update_actor(
            rgb_obs.detach(),
            low_dim_obs.detach(),
            action,
            demos,
        )
        metrics = critic_metrics
        metrics.update(actor_metrics)
        metrics["batch_reward"] = reward.mean().detach()
        return metrics

    def update_target_critic(self, step):
        if step % self.critic_target_interval == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
