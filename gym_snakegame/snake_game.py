import numpy as np
from collections import deque
import pygame

import gymnasium as gym
from gymnasium import spaces

def compare(a, b):
  if a == b: return 0;
  if a > b:  return -1;
  if a < b:  return 1;

class SnakeGameEnv(gym.Env):
  metadata = {
    "render_modes": ["human", "rgb_array", "ansi"],
    "render_fps": 20,
  }

  def __init__(self, render_mode=None, board_size=15, n_food=1):
    assert board_size >= 5
    assert n_food > 0

    # constants used to identify items on the game grid
    self.BLANK = 0
    self.HEAD = 1
    self.FOOD = board_size**2 + 1
    
    # in the original code, n_channel=(1, 2 or 4) but
    # in this version, only a single channel is allowed
    self.n_channel = 1
    
    # game state
    self.board_size = board_size  # The size of the square grid
    self.n_food = n_food
    self.snake = deque() # snake is represented by a list of (x,y) points 
    self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint32)
    self.clock = None

    # rendering data
    self.window = None
    self.render_mode = render_mode
    self.color_gradient = (255 - 100) / (board_size**2)
    self.window_width = 600  # The size of the PyGame window
    self.window_height = 700
    self.window_diff = self.window_height - self.window_width
    
    # every gymnasium environment must observation and action spaces
    
    # Actions:  RIGHT, UP, LEFT, DOWN
    self.action_space = spaces.Discrete(4)
    self._action_to_direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    
    # Observations:
    # * 8 binary values, one per direction (N, NE, E, SE, S, SW, W, NW)
    #     - True=Walkable, False=Blocked
    # * 1 integer between [1,8] representing direction (N, NE, E, SE, S, SW, W, NW)
    self.observation_space = spaces.Dict({
      "vision": spaces.MultiBinary(n=(3,3)),
      "smell": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int8),
    })

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    # reset
    self.board.fill(0)
    self.snake.clear()
    for i in range(3):
      self.snake.appendleft(np.array([self.board_size // 2, self.board_size // 2 - i]))
    for i, (x, y) in enumerate(self.snake):
      self.board[x, y] = len(self.snake) - i

    self._place_target(initial=True)

    # update iteration
    self._n_step = 0
    self._score = 0
    self.prev_action = 1

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self.render()

    return observation, info

  def _place_target(self, initial: bool = False) -> None:
    target_candidate = np.argwhere(self.board == self.BLANK)
    if initial:
      target_list = target_candidate[self.np_random.choice(len(target_candidate), self.n_food)]
      for x, y in target_list:
        self.board[x, y] = self.FOOD
    else:
      if target_candidate.size == 0:
        return
      else:
        new_target = target_candidate[self.np_random.choice(len(target_candidate))]
        self.board[new_target[0], new_target[1]] = self.FOOD

  def _get_obs(self):
    """
    Use the current game state to construct an `observation` for the agent.
    Should agree with the format specified by `self.observation_space`.
    """
    
    # create a 3x3 binary array centered on the snake's head
    vision = np.ones((3,3), dtype=np.bool)
    head_pos = self.snake[-1]
    head_c, head_r = head_pos;
    
    # loop over all 8 grid cells around the snake's head
    for dr in [-1, 0, 1]:
      for dc in [-1, 0, 1]:
        board_c = head_c + dc;
        board_r = head_r + dr;
        
        is_inside_grid = (
          board_c >= 0              and
          board_r >= 0              and
          board_c < self.board_size and
          board_r < self.board_size
        );
        
        if is_inside_grid:
          board_value = self.board[board_c][board_r];
          is_blank = (board_value == self.BLANK);
          is_food  = (board_value == self.FOOD);
          is_walkable = is_food or is_blank;
          
          vision[1+dc, 1+dr] = (1 if is_walkable else 0);
        else:
          vision[1+dc, 1+dr] = 0;
    
    # get coordinates of nearest food to snake's head,
    # and compute a smell vector in that direction
    food_pos = self.get_nearest_food(self.board, head_pos);
    smell_c = compare(head_c, food_pos[0]);
    smell_r = compare(head_r, food_pos[1]);
    smell = np.array([smell_c, smell_r], dtype=np.int8);
    
    return {
      "vision": vision,
      "smell": smell
    }

  def get_nearest_food(self, grid, head_pos):
    foods = np.array(np.where(grid == self.FOOD)).T
    foods_dist = np.pow(foods - head_pos, 2).sum(axis=1)

    nearest_food_idx = np.argmin(foods_dist)
    return foods[nearest_food_idx]

  def _get_info(self):
    return {"snake_length": len(self.snake), "prev_action": self.prev_action}

  def step(self, action: int):
    direction = self._action_to_direction[action]

    # update iteration
    self._n_step += 1

    current_head = self.snake[-1]
    current_tail = self.snake[0]
    next_head = current_head + direction

    if np.array_equal(next_head, self.snake[-2]):
      next_head = current_head - direction

    # get out the board
    if not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size):
      reward = -1
      terminated = True
    # hit the snake
    elif 0 < self.board[next_head[0], next_head[1]] < self.FOOD:
      reward = -1
      terminated = True
    else:
      # blank
      if self.board[next_head[0], next_head[1]] == self.BLANK:
        self.board[current_tail[0], current_tail[1]] = self.BLANK
        self.snake.popleft()
        reward = 0
        terminated = False
      # target
      # self.board[next_head[0], next_head[1]] == self.FOOD
      else:
        self._score += 1
        reward = 1
        self._place_target()
        self.board[next_head[0], next_head[1]] = 0
        if len(self.snake) == self.board_size**2:
          terminated = True
        else:
          terminated = False
      self.snake.append(next_head)
      for x, y in self.snake:
        self.board[x][y] += 1

    observation = self._get_obs()
    info = self._get_info()

    if self.render_mode == "human":
      self.render()

    self.prev_action = action

    truncated = False
    return observation, reward, terminated, truncated, info

  def render(self):
    if self.render_mode is None:
      assert self.spec is not None
      gym.logger.warn(
        "You are calling render method without specifying any render mode. "
        "You can specify the render_mode at initialization, "
        f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
      )
      return

    if self.render_mode in {"rgb_array", "human"}:
      return self._render_frame()

    if self.render_mode in {"ansi"}:
      return self._render_ansi()
    
  
  def _render_char(self, c):
    if   c == self.FOOD:  return "●"; # food
    elif c == self.HEAD:  return "█"; # snake head
    elif c == self.BLANK: return "░"; # empty space
    else:                 return "█"; # snake body

  def _render_ansi(self):
    result = "";
    result += f"[ Score {self._score} ]\n"

    horiz_line = "-"*self.board_size + '\n';
    result += horiz_line;

    for r in range(self.board_size):
      for c in range(self.board_size):
        result += self._render_char(self.board[r][c])
      result += "\n"
    
    result += horiz_line;
    result += "\nVision:";
    
    obs = self._get_obs();
    vision = obs["vision"];
    for r in range(3):
      result += "\n";
      for c in range(3):
        if vision[r][c] == 1: result += "░";
        else:                 result += "█";
    
    return result;
  
  # ---- PYGAME RENDERING CODE BELOW ---- #

  def get_body_color(self, r: int, c: int):
    color = 255 - self.color_gradient * self.board[r][c]
    return (color, color, color)
  
  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()

  def _render_frame(self):
    pygame.font.init()
    if self.window is None:
      pygame.init()
      self.square_size = self.window_width // self.board_size
      self.font_size = self.window_diff // 3
      if self.render_mode == "human":
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
      else:
        self.window = pygame.Surface((self.window_width, self.window_height))

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_width, self.window_height))
    canvas.fill((0, 0, 0))
    myFont = pygame.font.SysFont("consolas", self.font_size, bold=True)
    score_render_text = myFont.render(f"score: {self._score}", True, (255, 255, 255))
    n_step_render_text = myFont.render(f"step: {self._n_step}", True, (255, 255, 255))

    canvas.blit(
      score_render_text,
      (self.window_width // 30 * 1, self.window_diff // 2 - self.font_size // 2),
    )
    canvas.blit(
      n_step_render_text,
      (self.window_width // 30 * 15, self.window_diff // 2 - self.font_size // 2),
    )

    for r in range(self.board_size):
      for c in range(self.board_size):
        if self.board[r, c] == self.BLANK:
          pygame.draw.rect(
            canvas,
            (200, 200, 200),
            pygame.Rect(
              self.square_size * c,
              self.window_diff + self.square_size * r,
              self.square_size,
              self.square_size,
            ),
            1,
          )
        # head
        elif self.board[r, c] == self.HEAD:
          pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
              self.square_size * c,
              self.window_diff + self.square_size * r,
              self.square_size,
              self.square_size,
            ),
          )
        elif self.board[r, c] == self.FOOD:
          pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
              self.square_size * c,
              self.window_diff + self.square_size * r,
              self.square_size,
              self.square_size,
            ),
          )
        # body
        else:
          pygame.draw.rect(
            canvas,
            self.get_body_color(r, c),
            pygame.Rect(
              self.square_size * c,
              self.window_diff + self.square_size * r,
              self.square_size,
              self.square_size,
            ),
          )
        # blank

    if self.render_mode == "human":
      # The following line copies our drawings from `canvas` to the visible window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()

      # We need to ensure that human-rendering occurs at the predefined framerate.
      # The following line will automatically add a delay to keep the framerate stable.
      self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
      return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))