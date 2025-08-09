import gymnasium as gym;
from gym_snakegame.snake_game import SnakeGameEnv

gym.register(
    id="gym_snakegame/SnakeGame-v0",
    entry_point=SnakeGameEnv,
)