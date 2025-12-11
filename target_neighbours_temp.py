from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Token:
  id: str
  kind: str
  row: int
  col: int
  label: str = ""


class GameLogic:
  def __init__(self, grid_size: int) -> None:
    self.grid_size = grid_size

  def neighbours(self, row: int, col: int) -> List[Tuple[int, int]]:
    options = [
      (row - 1, col),
      (row + 1, col),
      (row, col - 1),
      (row, col + 1),
    ]
    return [
      (r, c)
      for r, c in options
      if 0 <= r < self.grid_size and 0 <= c < self.grid_size
    ]

  def reconstruct_path(
    self,
    parents: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
  ) -> List[Tuple[int, int]]:
    path = [goal]
    node = goal
    while node != start:
      node = parents[node]
      path.append(node)
    return path[::-1]

  def find_path(
    self,
    board: List[List[Optional[str]]],
    tokens: Dict[str, Token],
    start: Tuple[int, int],
    goal: Tuple[int, int],
  ) -> Optional[List[Tuple[int, int]]]:
    queue: List[Tuple[int, int]] = [start]
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    while queue:
      row, col = queue.pop(0)
      if (row, col) == goal:
        return self.reconstruct_path(parents, start, goal)
      for nr, nc in self.neighbours(row, col):
        if (nr, nc) in visited:
          continue
        occupant = board[nr][nc]
        if occupant:
          if (nr, nc) != goal:
            occupant_token = tokens[occupant]
            if occupant_token.kind in ("wall", "side"):
              continue
        visited.add((nr, nc))
        parents[(nr, nc)] = (row, col)
        queue.append((nr, nc))
    return None


