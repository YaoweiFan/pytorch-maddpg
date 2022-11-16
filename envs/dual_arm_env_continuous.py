import os
from os.path import dirname
from absl import logging  # information print
from robosuite.controllers import load_controller_config
import robosuite as suite
import numpy as np
import pandas as pd

from robosuite.environments.robot_env import RobotEnv

from .multi_agent_env import MultiAgentEnv


class DualArmContinuousEnv(MultiAgentEnv):
    """Dual-arm assemble environment for decentralised multi-agent coordination scenarios."""

    def __init__(
            self,
            seed,
            reward_shaping,
            reward_mimic,
            reward_success,
            reward_defeat,
            reward_scale,
            episode_limit,
            obs_timestep_number,
            debug,

            joint_pos_size,
            joint_vel_size,
            eef_pos_size,
            eef_quat_size,
            ft_size,
            peg_pos_size,
            peg_to_hole_size,
            robot_state_size,
            object_state_size,
            n_agents,
            action_dim,
            obs_choose,
            trajectory_data_path,

            has_renderer,
            has_offscreen_renderer,
            ignore_done,
            use_camera_obs,
            control_freq,
            camera_name,
            camera_heights,
            camera_widths,
    ):
        super().__init__()

        # MuJoCo
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.ignore_done = ignore_done
        self.use_camera_obs = use_camera_obs
        self.control_freq = control_freq
        self.camera_name = camera_name
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

        # Observation and state
        self.joint_pos_size = joint_pos_size
        self.joint_vel_size = joint_vel_size
        self.eef_pos_size = eef_pos_size
        self.eef_quat_size = eef_quat_size
        self.ft_size = ft_size
        self.peg_pos_size = peg_pos_size
        self.peg_to_hole_size = peg_to_hole_size
        self.robot_state_size = robot_state_size
        self.object_state_size = object_state_size
        self.obs_choose = obs_choose
        self.obs = None

        # Action
        self.last_action = None
        self.action_dim = action_dim

        # Agents
        self.n_agents = n_agents

        # Rewards
        self.reward_shaping = reward_shaping
        self.reward_mimic = reward_mimic
        self.reward_success = reward_success
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale

        # Algorithm
        self.obs_timestep_number = obs_timestep_number
        self.episode_limit = episode_limit
        self.debug = debug
        self.seed = seed
        self.episode_count = 0
        self.episode_steps = 0
        self.total_steps = 0
        self.timeouts = 0
        self.assemble_success = 0
        self.assemble_game = 0

        # Mimic trajectory
        absl_path = os.path.join(dirname(dirname(__file__)), trajectory_data_path)
        trajectory_data = pd.read_csv(absl_path)
        self.robot0_trajectory = np.array([[trajectory_data.Px.array[i],
                                            trajectory_data.Py.array[i],
                                            trajectory_data.Pz.array[i]] for i in range(len(trajectory_data.Px.array))])

        self.robot1_trajectory = np.array([[trajectory_data.Qx.array[i],
                                            trajectory_data.Qy.array[i],
                                            trajectory_data.Qz.array[i]] for i in range(len(trajectory_data.Qx.array))])

    def _launch(self):
        """Launch the Dual-arm assemble environment."""
        options = {"env_name": "TwoArmAssemble", "env_configuration": "single-arm-parallel", "robots": []}
        for i in range(self.n_agents):
            options["robots"].append("Panda")
        controller_name = "OSC_POSITION"
        options["controller_configs"] = load_controller_config(default_controller=controller_name)

        self.env = suite.make(
            **options,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            ignore_done=self.ignore_done,
            use_camera_obs=self.use_camera_obs,
            control_freq=self.control_freq,
            horizon=self.episode_limit,
            reward_shaping=self.reward_shaping,
            reward_scale=self.reward_scale,
            camera_names=self.camera_name,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
        )
        return True

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self.episode_steps = 0
        if self.episode_count == 0:
            self._launch()
        self.obs = self.env.reset()

        self.last_action = np.zeros((self.n_agents, self.action_dim))
        return self.get_state(), self.get_obs()

    def _mimic_reward(self):
        """Return mimic reward."""
        mimic_reward = 0

        for robot_id in range(self.n_agents):
            min_dis = 1000000000

            check_data = None
            pos_curr = None
            if robot_id == 0:
                pos_curr = self.obs["robot0_eef_pos"]
                check_data = self.robot0_trajectory[self.episode_steps:]
            elif robot_id == 1:
                pos_curr = self.obs["robot1_eef_pos"]
                check_data = self.robot1_trajectory[self.episode_steps:]

            for i in range(len(check_data)):
                tmp_dis = np.linalg.norm(check_data[i] - pos_curr)
                if tmp_dis < min_dis:
                    min_dis = tmp_dis

            mimic_reward -= min_dis

        return 10 * mimic_reward

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info.
        actions: np.ndarray (n_agents, action_dim)
        """
        # last_action 更新
        self.last_action = actions.copy()

        grab_actions = np.array([[1], [1]])
        united_actions = np.concatenate((actions, grab_actions), axis=1).reshape(-1)
        self.obs, reward, terminated, info = self.env.step(united_actions)
        self.render()

        if self.reward_mimic:
            reward += self._mimic_reward()

        self.total_steps += 1
        self.episode_steps += 1

        if self.episode_steps >= self.episode_limit:
            self.timeouts += 1

        if terminated:
            self.episode_count += 1
            self.assemble_game += 1
            if info["success"]:
                self.assemble_success += 1
                reward += self.reward_success * (1 - self.reward_shaping)
            if info["defeat"]:
                reward += self.reward_defeat * (1 - self.reward_shaping)

        return self.get_state(), self.get_obs(), [reward, reward], terminated, info

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - joint_pos
           - joint_vel
           - eef_pos
           - eef_quat
           - ft
           - peg_pos
           - peg_to_hole

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. 

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        assert isinstance(self.env, RobotEnv)
        prefix = self.env.robots[agent_id].robot_model.naming_prefix

        if self.obs_choose == "noft":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_hole"],
                )
            )
        elif self.obs_choose == "noeef":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_ft"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_hole"],
                )
            )
        elif self.obs_choose == "nopegpos":
            agent_obs = np.concatenate(
                (
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    self.obs[prefix + "eef_ft"],
                    self.obs[prefix + "peg_to_hole"],
                )
            )
        else:
            agent_obs = np.concatenate(
                (
                    # self.obs[prefix + "joint_pos"],
                    # self.obs[prefix + "joint_vel"],
                    self.obs[prefix + "eef_pos"],
                    self.obs[prefix + "eef_quat"],
                    # self.obs[prefix + "eef_ft"],
                    np.array([0, 0, 0, 0, 0, 0]),
                    # self.obs[prefix + "ft"],
                    self.obs[prefix + "peg_pos"],
                    self.obs[prefix + "peg_to_hole"],
                )
            )

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs, self.episode_steps / self.episode_limit)

        return agent_obs

    def get_state(self):
        """Returns the global state.
        This function assumes that self.obs is up-to-date.
        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.concatenate(
            (
                self.obs["robot0_robot-state"],
                self.obs["robot1_robot-state"],
                self.obs["object-state"],
            )
        )
        return state

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.robot_state_size * 2 + self.object_state_size

    def get_obs_size(self):
        """Returns the size of the observation."""
        if self.obs_choose == "noft":
            return self.eef_pos_size + self.eef_quat_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "noeef":
            return self.ft_size + self.peg_pos_size + self.peg_to_hole_size
        if self.obs_choose == "nopegpos":
            return self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_to_hole_size
        return self.eef_pos_size + self.eef_quat_size + self.ft_size + self.peg_pos_size + self.peg_to_hole_size

    def get_avail_actions(self):
        # Continuous environment, forbid to use this method
        raise Exception("Continuous environment, forbid to use this method!")

    def get_avail_agent_actions(self, agent_id):
        # Continuous environment, forbid to use this method
        raise Exception("Continuous environment, forbid to use this method!")

    def get_total_actions(self):
        """Returns the dim of actions an agent could ever take."""
        return self.action_dim

    def render(self):
        """Render"""
        if self.has_renderer:
            self.env.render()

    def close(self):
        """Close the environment"""
        self.env.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self.seed

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "action_dim": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "assemble_success": self.assemble_success,
            "assemble_games": self.assemble_game,
            "win_rate": self.assemble_success / self.assemble_game,
            "timeouts": self.timeouts,
        }
        return stats

    def get_frame(self):
        return self.obs[self.camera_name + "_image"][::-1]
