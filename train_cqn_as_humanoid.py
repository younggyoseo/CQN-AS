import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs


import humanoid_src.humanoid_env as humanoid_env
import utils
from logger import Logger
from humanoid_src.replay_buffer_action_sequence import (
    ReplayBufferStorage,
    make_replay_loader,
)
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, action_sequence, cfg):
    cfg.obs_shape = obs_spec.shape
    # Action sequence
    cfg.action_shape = (action_sequence, *action_spec.shape)
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.action_sequence,
            self.cfg.agent,
        )
        self.timer = utils.Timer()
        self.logger = Logger(
            self.work_dir, self.cfg.use_tb, self.cfg.use_wandb, self.cfg
        )
        self._update_step = 0
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        self.train_env = humanoid_env.make(
            self.cfg.task_name,
        )
        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble = utils.TemporalEnsembleControl(
                1000,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )
            self.eval_temporal_ensemble = utils.TemporalEnsembleControl(
                1000,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )
        # create replay buffer
        self.initialize_loader()

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def update_step(self):
        return self._update_step

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        """We use train env for evaluation, because it's convenient"""
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_step = 0
            time_step = self.train_env.reset()
            if self.cfg.temporal_ensemble:
                self.eval_temporal_ensemble.reset()
            self.video_recorder.init(self.train_env, enabled=(episode == 0))
            while not time_step.last():
                if (
                    self.cfg.temporal_ensemble
                    or episode_step % self.cfg.action_sequence == 0
                ):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                        )
                    action = action.reshape([self.cfg.action_sequence, -1])
                    if self.cfg.temporal_ensemble:
                        self.eval_temporal_ensemble.register_action_sequence(action)
                if self.cfg.temporal_ensemble:
                    sub_action = self.eval_temporal_ensemble.get_action()
                else:
                    sub_action = action[step % self.cfg.action_sequence]
                time_step = self.train_env.step(sub_action)
                self.video_recorder.record(self.train_env)
                total_reward += time_step.reward
                step += 1
                episode_step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        do_eval = False

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()

        if self.cfg.use_compile:
            update_fn = torch.compile(self.agent.update)
            act_fn = torch.compile(self.agent.act)
            torch.set_float32_matmul_precision("high")
        else:
            update_fn = self.agent.update
            act_fn = self.agent.act

        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(self.train_env.render())
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # do evaluation before resetting the environment
                if do_eval:
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    self.eval()
                    do_eval = False

                # reset env
                time_step = self.train_env.reset()
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(self.train_env.render())

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # set a flag to initate evaluation when the current episode terminates
            if self.global_step >= self.cfg.eval_every_frames and eval_every_step(
                self.global_step
            ):
                do_eval = True

            # sample action
            if (
                self.cfg.temporal_ensemble
                or episode_step % self.cfg.action_sequence == 0
            ):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    # Here, use eval_mode = True
                    action = act_fn(
                        time_step.observation,
                        self.global_step,
                        eval_mode=True,
                    )
                action = action.reshape([self.cfg.action_sequence, -1])
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.register_action_sequence(action)

            # try to update the agent
            if (
                not seed_until_step(self.global_step)
                and self.global_step % self.cfg.agent.update_every_steps == 0
            ):
                for _ in range(self.cfg.num_update_steps):
                    batch = next(self.replay_iter)
                    batch = utils.to_torch_tensor_dict(batch, self.device)
                    metrics = update_fn(batch)
                    self._update_step += 1
                    self.agent.update_target_critic(self.update_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            if self.cfg.temporal_ensemble:
                sub_action = self.train_temporal_ensemble.get_action()
            else:
                sub_action = action[episode_step % self.cfg.action_sequence]
            # Here, add noise to sub_action
            sub_action = self.agent.add_noise_to_action(sub_action, self.global_step)
            time_step = self.train_env.step(sub_action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(self.train_env.render())
            episode_step += 1
            self._global_step += 1

    def initialize_loader(self):
        data_specs = (
            self.train_env.raw_observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")
        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.action_sequence,
            fill_action="last_action",
        )
        self._replay_iter = None

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(
    config_path="cfgs",
    config_name="config_cqn_as_humanoid",
)
def main(cfg):
    from train_cqn_as_humanoid import (
        Workspace as W,
    )

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
