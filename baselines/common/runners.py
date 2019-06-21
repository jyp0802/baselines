import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

        overcooked = 'env_name' in env.__dict__.keys() and env.env_name == "Overcooked-v0"
        if overcooked:
            self.obs0 = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs1 = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)

            both_obs_and_state_and_other_idx = env.reset()

            ##### STARTING HERE: This portion of the code can most likely be improved a lot
            transp_shape = list(range(len(both_obs_and_state_and_other_idx.shape)))
            transp_shape[0], transp_shape[1] = 1, 0
            both_obs_and_state_and_other_idx = np.transpose(both_obs_and_state_and_other_idx, transp_shape)
            
            both_obs, state_and_other_idx = both_obs_and_state_and_other_idx
            state_and_other_idx = np.array(state_and_other_idx)
            both_obs = np.array(both_obs)

            threads, players = np.array(both_obs).shape
            
            obs = []
            for y in range(players):
                sub_obs = []
                for x in range(threads):
                    sub_obs.append(both_obs[x][y])
                obs.append(sub_obs)

            obs0, obs1 = obs

            self.obs0[:] = np.array(obs0)
            self.obs1[:] = np.array(obs1)
            self.curr_state, self.other_agent_idx = state_and_other_idx[:, 0], state_and_other_idx[:, 1]
            ##### ENDING here
        else:
            self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

