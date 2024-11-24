import logging

from gymnasium import spaces
import numpy as np
from typing import Union, Dict, Any, NamedTuple
from collections import deque

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX
from bigym.action_modes import JointPositionActionMode
from bigym_src.bigym_utils import TASK_MAP

from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.action_modes import PelvisDof
from demonstrations.demo import DemoStep
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from dm_env import StepType, specs


def _task_name_to_env_class(task_name: str) -> type[BiGymEnv]:
    return TASK_MAP[task_name]


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=time_step.rgb_obs,
            low_dim_obs=time_step.low_dim_obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
            demo=time_step.demo or 0.0,
        )

    def low_dim_observation_spec(self):
        return self._env.low_dim_observation_spec()

    def rgb_observation_spec(self):
        return self._env.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        return self._env.low_dim_raw_observation_spec()

    def rgb_raw_observation_spec(self):
        return self._env.rgb_raw_observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class BiGym:
    def __init__(
        self,
        task_name: str,
        enable_all_floating_dof: bool = True,
        action_mode: str = "absolute",
        episode_length: int = 3000,
        demo_down_sample_rate: int = 25,
        frame_stack: int = 1,
        camera_shape: tuple[int] = (84, 84),
        camera_keys: tuple[str] = ("head", "right_wrist", "left_wrist"),
        state_keys: tuple[str] = (
            "proprioception",
            "proprioception_grippers",
            "proprioception_floating_base",
        ),
        render_mode: str = "rgb_array",
        normalize_low_dim_obs: bool = False,
    ):
        self._task_name = task_name
        self._enable_all_floating_dof = enable_all_floating_dof
        self._action_mode = action_mode
        self._episode_length = episode_length
        self._frame_stack = frame_stack
        self._camera_shape = camera_shape
        self._camera_keys = camera_keys
        self._state_keys = state_keys
        self._render_mode = render_mode
        self._demo_down_sample_rate = demo_down_sample_rate
        self._normalize_low_dim_obs = normalize_low_dim_obs

        self._launch()
        self._initialize_frame_stack()
        self._construct_action_and_observation_spaces()

    def low_dim_observation_spec(self) -> spaces.Box:
        shape = self.low_dim_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def low_dim_raw_observation_spec(self) -> spaces.Box:
        shape = self.low_dim_raw_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def rgb_observation_spec(self) -> spaces.Box:
        shape = self.rgb_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def rgb_raw_observation_spec(self) -> spaces.Box:
        shape = self.rgb_raw_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def action_spec(self) -> spaces.Box:
        shape = self.action_space.shape
        spec = specs.Array(shape, np.float32, "action")
        return spec

    def step(self, action):
        action = self._convert_action_to_raw(action)
        bigym_obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self._extract_obs(bigym_obs)
        self._step_counter += 1

        # Timelimit
        if self._step_counter >= (self._episode_length // self._demo_down_sample_rate):
            truncated = True
        else:
            truncated = False

        # Handle bootstrap
        if terminated or truncated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        discount = float(1 - terminated)

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=reward,
            discount=discount,
            demo=0.0,
        )

    def reset(self, **kwargs):
        # Clear deques used for frame stacking
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        bigym_obs, info = self._env.reset(**kwargs)
        obs = self._extract_obs(bigym_obs)
        self._step_counter = 0

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            demo=0.0,
        )

    def render(self) -> Union[None, np.ndarray]:
        return self._env.render()

    def get_demos(self, num_demos):
        """
        1. Collect or fetch demonstrations
        2. Compute action stats from demonstrations, override self._action_stats
        3. Rescale actions in demonstrations to [-1, 1] space
        """
        demo_store = DemoStore()
        assert num_demos != 0
        raw_demos = demo_store.get_demos(
            Metadata.from_env(self._env),
            amount=num_demos,
            frequency=CONTROL_FREQUENCY_MAX // self._demo_down_sample_rate,
        )

        for raw_demo in raw_demos:
            for ts in raw_demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32) for k, v in ts.observation.items()
                }

        # Need to filter out states after the first rewarding state
        new_raw_demos = []
        for raw_demo in raw_demos:
            new_raw_demo = []
            for demostep in raw_demo.timesteps:
                new_raw_demo.append(demostep)
                if demostep.reward > 0:
                    break
            new_raw_demos.append(new_raw_demo)
        raw_demos = new_raw_demos

        # compute low_dim stats from demonstrations
        # NOTE: this needs to be done before convert_demo_to_timesteps
        if self._normalize_low_dim_obs:
            self._low_dim_obs_stats = self.extract_low_dim_obs_stats(raw_demos)

        demos = []
        num_successful = 0
        for raw_demo in raw_demos:
            demo, successful = self.convert_demo_to_timesteps(raw_demo)
            num_successful += float(successful)
            demos.append(demo)
        print(f"Number of successful demos: {num_successful}/{len(raw_demos)}")

        # override action stats with demonstration-based stats
        self._action_stats = self.extract_action_stats(demos)
        # rescale actions with action stats
        demos = [self.rescale_demo_actions(demo) for demo in demos]
        return demos

    def extract_action_stats(self, demos: list[list[ExtendedTimeStep]]):
        actions = []
        for demo in demos:
            for timestep in demo:
                actions.append(timestep.action)
        actions = np.stack(actions)

        # Two Gripper one-hot actions' stats are hard-coded
        action_max = np.hstack([np.max(actions, 0)[:-2], 1, 1])
        action_min = np.hstack([np.min(actions, 0)[:-2], 0, 0])

        assert np.all(action_min >= self._env.action_space.low) and np.all(
            action_max <= self._env.action_space.high
        )

        action_stats = {
            "max": action_max,
            "min": action_min,
        }
        return action_stats

    def extract_low_dim_obs_stats(self, demos: list[list[ExtendedTimeStep]]):
        low_dim_obses = []
        for demo in demos:
            for timestep in demo:
                low_dim_obs = np.hstack(
                    [timestep.observation[key] for key in self._state_keys],
                    dtype=np.float32,
                )
                low_dim_obses.append(low_dim_obs)
        low_dim_obses = np.stack(low_dim_obses)

        # Two Gripper one-hot actions' stats are hard-coded
        low_dim_obs_mean = np.mean(low_dim_obses, 0)
        low_dim_obs_std = np.std(low_dim_obses, 0)

        low_dim_obs_stats = {
            "mean": low_dim_obs_mean,
            "std": low_dim_obs_std,
        }
        return low_dim_obs_stats

    def convert_demo_to_timesteps(self, demo):
        timesteps = []

        # Clear deques used for frame stacking
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        # Detect whether the demo is successful or not
        # This is inefficient but not that slow
        rewards = []
        for i in range(len(demo)):
            demostep = demo[i]
            reward = demostep.reward
            rewards.append(reward)
        successful_demo = sum(rewards) > 0.25

        last_timestep = False
        for i in range(len(demo)):
            demostep = demo[i]

            obs = self._extract_obs(demostep.observation)
            reward = demostep.reward
            discount = 1.0
            term, trunc = demostep.termination, demostep.truncation
            action = demostep.info["demo_action"].astype(np.float32)

            if i == 0:
                step_type = StepType.FIRST
            else:
                if (i == len(demo) - 1) or reward > 0:
                    if not (term or trunc):
                        # Timelimit
                        trunc = True
                    step_type = StepType.LAST
                    if term:
                        # Discount becomes 0.0 only when term is True
                        discount = 0.0
                    last_timestep = True
                else:
                    step_type = StepType.MID

            timestep = ExtendedTimeStep(
                rgb_obs=obs["rgb_obs"],
                low_dim_obs=obs["low_dim_obs"],
                step_type=step_type,
                action=action,
                reward=reward,
                discount=discount,
                demo=int(successful_demo),
            )
            timesteps.append(timestep)

            if last_timestep:
                break

        return timesteps, successful_demo

    def rescale_demo_actions(
        self, demo: list[ExtendedTimeStep]
    ) -> list[ExtendedTimeStep]:
        new_timesteps = []
        for timestep in demo:
            action = self._convert_action_from_raw(timestep.action)
            new_timesteps.append(timestep._replace(action=action))
        return new_timesteps

    def close(self) -> None:
        self._env.close()

    def _launch(self):
        bigym_class = _task_name_to_env_class(self._task_name)
        camera_configs = [
            CameraConfig(
                name=camera_name,
                rgb=True,
                depth=False,
                resolution=self._camera_shape,
            )
            for camera_name in self._camera_keys
        ]

        if self._enable_all_floating_dof:
            action_mode = JointPositionActionMode(
                absolute=self._action_mode == "absolute",
                floating_base=True,
                floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
            )
        else:
            action_mode = JointPositionActionMode(
                absolute=self._action_mode == "absolute",
                floating_base=True,
            )

        self._env = bigym_class(
            render_mode=self._render_mode,
            action_mode=action_mode,
            observation_config=ObservationConfig(
                cameras=camera_configs,
                proprioception=True,
                privileged_information=False,
            ),
            control_frequency=CONTROL_FREQUENCY_MAX // self._demo_down_sample_rate,
        )

        # Episode length counter
        self._step_counter = 0

    def _initialize_frame_stack(self):
        # Create deques for frame stacking
        self._low_dim_obses = deque([], maxlen=self._frame_stack)
        self._frames = {
            camera_key: deque([], maxlen=self._frame_stack)
            for camera_key in self._camera_keys
        }

    def _construct_action_and_observation_spaces(self):
        # Setup action/observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_space.shape
        )

        # Compute dimension of low_dim_obs
        low_dim = 0
        for state_key in self._state_keys:
            low_dim += self._env.observation_space[state_key].shape[-1]
        self.low_dim_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_dim * self._frame_stack,),
            dtype=np.float32,
        )
        self.low_dim_raw_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(low_dim,), dtype=np.float32
        )  # without frame stacking
        self.rgb_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self._camera_keys), 3 * self._frame_stack, *self._camera_shape),
            dtype=np.uint8,
        )
        self.rgb_raw_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self._camera_keys), 3, *self._camera_shape),
            dtype=np.uint8,
        )  # without frame stacking

        # Set default action stats, which will be overridden by demonstration action stats
        # Required for a case we don't use demonstrations
        action_min = -np.ones(self.action_space.shape, dtype=self.action_space.dtype)
        action_max = np.ones(self.action_space.shape, dtype=self.action_space.dtype)
        # Two grippers for bimanual setups
        action_min[-2:] = 0
        action_max[-2:] = 1
        self._action_stats = {"min": action_min, "max": action_max}

    def _convert_action_to_raw(self, action):
        """Convert [-1, 1] action to raw joint space using action stats"""
        assert (max(action) <= 1) and (min(action) >= -1)
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        _action_min = action_min
        _action_max = action_max
        new_action = (action + 1) / 2.0  # to [0, 1]
        new_action = (
            new_action * (_action_max - _action_min + 1e-8) + _action_min
        )  # original
        return new_action.astype(action.dtype, copy=False)

    def _convert_action_from_raw(self, action):
        """Convert raw action in joint space to [-1, 1] using action stats"""
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        _action_min = action_min
        _action_max = action_max

        new_action = (action - _action_min) / (
            _action_max - _action_min + 1e-8
        )  # to [0, 1]
        new_action = new_action * 2 - 1  # to [-1, 1]
        return new_action.astype(action.dtype, copy=False)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        out = dict()

        # Get low-dimensional state with stacking
        low_dim_obs = np.hstack(
            [obs[key] for key in self._state_keys], dtype=np.float32
        )
        if self._normalize_low_dim_obs:
            mean, std = self._low_dim_obs_stats["mean"], self._low_dim_obs_stats["std"]
            low_dim_obs = (low_dim_obs - mean) / (std + 1e-8)

        if len(self._low_dim_obses) == 0:
            for _ in range(self._frame_stack):
                self._low_dim_obses.append(low_dim_obs)
        else:
            self._low_dim_obses.append(low_dim_obs)
        out["low_dim_obs"] = np.concatenate(list(self._low_dim_obses), axis=0)

        # Get rgb observations with stacking
        for camera_key in self._camera_keys:
            pixels = obs[f"rgb_{camera_key}"].copy().astype(np.uint8)
            if len(self._frames[camera_key]) == 0:
                for _ in range(self._frame_stack):
                    self._frames[camera_key].append(pixels)
            else:
                self._frames[camera_key].append(pixels)
        out["rgb_obs"] = np.stack(
            [
                np.concatenate(list(self._frames[camera_key]), axis=0)
                for camera_key in self._camera_keys
            ],
            0,
        )
        return out

    def __del__(
        self,
    ) -> None:
        self.close()


def make(
    task_name,
    enable_all_floating_dof,
    action_mode,
    demo_down_sample_rate,
    episode_length,
    frame_stack,
    camera_shape,
    camera_keys,
    state_keys,
    render_mode,
    normalize_low_dim_obs,
):
    env = BiGym(
        task_name,
        enable_all_floating_dof=enable_all_floating_dof,
        action_mode=action_mode,
        episode_length=episode_length,
        demo_down_sample_rate=demo_down_sample_rate,
        frame_stack=frame_stack,
        camera_shape=camera_shape,
        camera_keys=camera_keys,
        state_keys=state_keys,
        render_mode=render_mode,
        normalize_low_dim_obs=normalize_low_dim_obs,
    )
    env = ExtendedTimeStepWrapper(env)
    return env
