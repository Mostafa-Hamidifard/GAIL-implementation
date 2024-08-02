import gymnasium as gym
import pickle
import os
from gymnasium.utils.play import play
import numpy as np


class MDP_logger:
    """_summary_
    This class stores the tuples (obs_t, obs_tp1, action, rew, terminated, truncated, info) in order for each step
    """

    def __init__(self) -> None:
        self.recorded_list = []

    def add_tuple(self, tuple):
        """_summary_
        adding a MDP entry
        Args:
            tuple (tuple): (obs_t, obs_tp1, action, rew, terminated, truncated, info)
        """
        if len(tuple) != 7:
            raise Exception(
                "number of data in the tuple is not 7: (obs_t, obs_tp1, action, rew, terminated, truncated, info)"
            )
        self.recorded_list.append(tuple)

    def save(self, path, name):
        """_summary_
        it saves the list of tuples (obs_t, obs_tp1, action, rew, terminated, truncated, info) in a pickle style
        Args:
            path (string): where to save
            name (string): the name to save
        """
        name = name + ".pkl"
        file_path = os.path.join(path, name)
        with open(file_path, "wb") as file:
            pickle.dump(
                {
                    "data": self.recorded_list,
                    "columns": (
                        "obs_t",
                        "obs_tp1",
                        "action",
                        "rew",
                        "terminated",
                        "truncated",
                        "info",
                    ),
                },
                file,
            )


# creating a logger instance
logger = MDP_logger()


# defining callback for each step to save player data
def play_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    info_tuple = (obs_t, obs_tp1, action, rew, terminated, truncated, info)
    logger.add_tuple(info_tuple)


# start playing env and recording data
play(
    gym.make("CarRacing-v2", render_mode="rgb_array", domain_randomize=True),
    keys_to_action={
        "w": np.array([0, 0.5, 0]),
        "a": np.array([-0.5, 0, 0]),
        "s": np.array([0, 0, 0.5]),
        "d": np.array([0.5, 0, 0]),
        "wa": np.array([-0.5, 0.5, 0]),
        "dw": np.array([0.5, 0.5, 0]),
        "ds": np.array([0.5, 0, 0.5]),
        "as": np.array([-0.5, 0, 0.5]),
    },
    noop=np.array([0, 0, 0]),
    callback=play_callback,
)

# Define the filename
player = "Mostafa"
filename = f"car_racing_{player}"

# Get the current directory of the script
current_directory = os.path.dirname(__file__)

# Define the parent directory (one level up)
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# Define the relative path to the data directory in the parent directory
relative_directory = os.path.join(parent_directory, r"data\expert_data")

# Create the directory if it doesn't exist
os.makedirs(relative_directory, exist_ok=True)

logger.save(relative_directory, filename)
