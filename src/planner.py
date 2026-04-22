import heapq
import math

import numpy as np


class ThetaStar:
    def __init__(self, cost_grid):
        self.grid = np.asarray(cost_grid)
        self.H, self.W = self.grid.shape

    def _neighbors(self, p):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = p[0] + dr, p[1] + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] < 999:
                    yield (nr, nc)

    def _step_cost(self, a, b):
        d = math.sqrt(2) if (a[0] != b[0] and a[1] != b[1]) else 1.0
        return float(self.grid[b[0], b[1]]) * d

    def _h(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def line_of_sight(self, a, b):
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if self.grid[r, c] >= 999:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def find_path(self, start, goal):
        if self.grid[start[0], start[1]] >= 999 or self.grid[goal[0], goal[1]] >= 999:
            return []
        open_heap = [(self._h(start, goal), start)]
        parent = {start: start}
        g = {start: 0.0}
        closed = set()
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while parent[path[-1]] != path[-1]:
                    path.append(parent[path[-1]])
                return path[::-1]
            if cur in closed:
                continue
            closed.add(cur)
            for nb in self._neighbors(cur):
                if nb in closed:
                    continue
                par = parent[cur]
                if par != cur and self.line_of_sight(par, nb):
                    new_g = g[par] + math.hypot(nb[0] - par[0], nb[1] - par[1]) * float(self.grid[nb[0], nb[1]])
                    new_par = par
                else:
                    new_g = g[cur] + self._step_cost(cur, nb)
                    new_par = cur
                if nb not in g or new_g < g[nb]:
                    g[nb] = new_g
                    parent[nb] = new_par
                    heapq.heappush(open_heap, (new_g + self._h(nb, goal), nb))
        return []


class DStarLite:
    def __init__(self, cost_grid):
        self.grid = np.asarray(cost_grid, dtype=float)
        self.H, self.W = self.grid.shape
        self.start = None
        self.goal = None
        self.g = {}

    def _neighbors(self, p):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = p[0] + dr, p[1] + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] < 999:
                    yield (nr, nc)

    def _step_cost(self, a, b):
        d = math.sqrt(2) if (a[0] != b[0] and a[1] != b[1]) else 1.0
        return float(self.grid[b[0], b[1]]) * d

    def _h(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _plan(self, start, goal):
        if self.grid[start[0], start[1]] >= 999 or self.grid[goal[0], goal[1]] >= 999:
            return []
        open_heap = [(self._h(start, goal), start)]
        parent = {start: start}
        g = {start: 0.0}
        closed = set()
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while parent[path[-1]] != path[-1]:
                    path.append(parent[path[-1]])
                self.g = g
                return path[::-1]
            if cur in closed:
                continue
            closed.add(cur)
            for nb in self._neighbors(cur):
                if nb in closed:
                    continue
                new_g = g[cur] + self._step_cost(cur, nb)
                if nb not in g or new_g < g[nb]:
                    g[nb] = new_g
                    parent[nb] = cur
                    heapq.heappush(open_heap, (new_g + self._h(nb, goal), nb))
        return []

    def find_path(self, start, goal):
        self.start = start
        self.goal = goal
        return self._plan(start, goal)

    def update_cell(self, pos, new_cost):
        self.grid[pos[0], pos[1]] = new_cost
        if self.start is None or self.goal is None:
            return []
        return self._plan(self.start, self.goal)
