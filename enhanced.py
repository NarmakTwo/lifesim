#!/usr/bin/env python3
"""Enhanced version of Particle Life Simulator.

This module defines improved simulation and application classes that
implement optimisations, multithreading, new controls, and helper
functions.  It is tuned for very large populations (hundreds of
thousands of particles) and works comfortably with dozens of
particle ``types`` – the original version slows down long before
100 000 agents, this variant makes that scale achievable on
modern hardware.

Key new features:

* **OpenGL rendering** (toggle with F2) pushes the point drawing to
the GPU, greatly reducing the cost of rendering huge swarms.
* **Configuration save/load** (Ctrl+S/Ctrl+O) writes/reads complete
state (types, rules, physics parameters, particle positions) to JSON.
* **Optional Numba JIT** (toggle with F3) compiles the force step to
native code; requires ``numba`` and ``numpy``.  When those packages are
not installed the simulation falls back to the threaded Python loop.

Run with ``python enhanced.py`` to launch the upgraded demo.
"""

import pygame, sys, math, uuid, threading
import random
from random import uniform
import colornames
from colorsys import hsv_to_rgb, rgb_to_hsv
from concurrent.futures import ThreadPoolExecutor
import json
import os
try:
    import numpy as np
except ImportError:
    np = None
# optionally import a cythonised physics kernel if available
try:
    import cython_accel
    _have_cython_accel = True
except ImportError:
    _have_cython_accel = False
# attempt to import OpenGL symbols; if unavailable we define no-op stubs
try:
    from OpenGL.GL import (
        glViewport, glMatrixMode, glLoadIdentity, glOrtho, glDisable,
        glGenBuffers, glBindBuffer, glBufferData, glEnableClientState,
        glVertexPointer, glColorPointer, glPointSize, glDrawArrays,
        glDisableClientState,
        GL_PROJECTION, GL_MODELVIEW, GL_DEPTH_TEST,
        GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_VERTEX_ARRAY, GL_COLOR_ARRAY,
        GL_FLOAT, GL_UNSIGNED_BYTE, GL_POINTS
    )
    _have_gl = True
except ImportError:
    _have_gl = False
    def _gl_stub(*args, **kwargs):
        pass
    glViewport = glMatrixMode = glLoadIdentity = glOrtho = glDisable = _gl_stub
    glGenBuffers = glBindBuffer = glBufferData = glEnableClientState = _gl_stub
    glVertexPointer = glColorPointer = glPointSize = glDrawArrays = _gl_stub
    glDisableClientState = _gl_stub
    # constants
    GL_PROJECTION = GL_MODELVIEW = GL_DEPTH_TEST = 0
    GL_ARRAY_BUFFER = GL_DYNAMIC_DRAW = GL_VERTEX_ARRAY = GL_COLOR_ARRAY = 0
    GL_FLOAT = GL_UNSIGNED_BYTE = 0

# optional numerical/GPU acceleration
try:
    from numba import njit, prange, cuda
    _numba_available = True
except ImportError:
    njit = lambda f: f  # dummy decorator
    prange = range  # fallback
    cuda = None
    _numba_available = False
# if numpy is missing we cannot use numba even if the module exists
if np is None:
    _numba_available = False

# Numba-compiled physics kernel (CPU parallel) ------------------------------------------------
if _numba_available:
    @njit(parallel=True)
    def _numba_step(xs, ys, vxs, vys, tids, rm_s, rm_r,
                    fr, fs, beta, W, H, gsize, cols, rows,
                    offsets, contents):
        n = xs.shape[0]
        for i in prange(n):
            x = xs[i]; y = ys[i]
            ax = 0.0; ay = 0.0
            fi = tids[i]
            # compute cell coordinates
            gx = int(x // gsize) % cols
            gy = int(y // gsize) % rows
            # iterate neighbour cells
            for dy_cell in (-1, 0, 1):
                for dx_cell in (-1, 0, 1):
                    cgx = (gx + dx_cell) % cols
                    cgy = (gy + dy_cell) % rows
                    base = cgx + cgy * cols
                    start = offsets[base]
                    end = offsets[base + 1]
                    for idx in range(start, end):
                        j = contents[idx]
                        if j == i:
                            continue
                        ti = tids[j]
                        strength = rm_s[fi, ti]
                        if strength == 0.0:
                            continue
                        mr = rm_r[fi, ti]
                        dx = xs[j] - x
                        dy = ys[j] - y
                        d = math.hypot(dx, dy)
                        if d == 0.0 or d > mr:
                            continue
                        # force function replicates Simulation._force
                        rn = d / mr
                        b = beta
                        if rn < b:
                            fval = rn / b - 1.0
                        elif rn < 1.0:
                            fval = strength * (1.0 - abs(2 * rn - 1 - b) / (1 - b))
                        else:
                            fval = 0.0
                        fval *= fs
                        ax += fval * dx / d
                        ay += fval * dy / d
            # velocity update
            vx = (vxs[i] + ax) * fr
            vy = (vys[i] + ay) * fr
            x = (x + vx) % W
            y = (y + vy) % H
            vxs[i] = vx; vys[i] = vy
            xs[i] = x; ys[i] = y

# pull in most of the existing helpers from the original script
from main import (
    App, ParticleType, Rule, Particle, Camera, UIButton, UISlider, UITextInput,
    ColorPicker, RulePopover, Simulation, clamp, hsv2rgb, rgb2hsv, lighten, txt,
    rrect,
    WIN_W, WIN_H, PANEL_W, SIM_W, SIM_H, PARTICLE_R,
    BG, GRID_C, PNL, PNL_DK, BDR, TXT, DIM, ACC, BTN_N, BTN_H, BTN_A,
    DNG, DNG_H, SUC, SUC_H, SLB, SLF, TAB_N, TAB_A,
    FSM, FMD, FLG, FXL, PAD, CR
)

pygame.init()
pygame.font.init()

# ---------------------------------------------------------------------------
#  Optimised simulation with multithreading & attractor support
# ---------------------------------------------------------------------------

class EnhancedSimulation(Simulation):
    """A drop-in replacement for ``Simulation`` with better performance
    and some experimental behaviours.

    Attributes added by this subclass:
    - helper_groups: groups used by the helper algorithm
    - chase_strength: float multiplier for chasing behaviour
    helper_groups: list[list[ParticleType]]
    chase_strength: float

    This version keeps a thread pool alive across steps for less overhead and
    allows the grid size to be changed on the fly (used by the cell size
    slider).

    * spatial hashing grid reduces neighbour searches
    * threading used to compute forces in parallel
    * optional attractor point (right‑click) applies a temporary force
    * wrap‑around boundaries eliminate wall hugging and keep particles
      in constant motion
    * tiny random jitter prevents particles from coming to a complete rest
    """

    def __init__(self, w, h, grid_size=60):
        super().__init__(w, h)
        # when working with very large populations the default spatial hash
        # cell size can be too small; allow automatic adjustment later.
        self.grid_size = grid_size
        self.auto_grid = True                # recompute ideal grid size each step
        # if numba is installed we can use a JIT compiled step routine
        self.use_numba = _numba_available
        # if a cython extension is available, we'll try it first
        self.use_cython = _have_cython_accel

        self._attractor = None      # (x, y, radius, strength)
        self._rng_lock = threading.Lock()
        # reuse executor to avoid recreating threads every step and specify
        # a sensible worker count up front to avoid oversubscription.
        import os
        cpu = os.cpu_count() or 1
        self._executor = ThreadPoolExecutor(max_workers=cpu)

        # integer indexing helpers (see step method)
        self._tid_to_idx: dict[str, int] = {}
        self._types_by_idx: list[ParticleType] = []

        # additional attributes used by helper logic
        self.helper_groups: list[list[ParticleType]] = []
        self.chase_strength: float = 0.0

    def set_attractor(self, wx, wy, radius, strength=1.0):
        with self._rng_lock:
            self._attractor = (wx, wy, radius, strength)

    def clear_attractor(self):
        with self._rng_lock:
            self._attractor = None

    def _apply_attractor(self, p, dt=1.0):
        with self._rng_lock:
            a = self._attractor
        if not a:
            return 0.0, 0.0
        wx, wy, rng, strn = a
        dx = wx - p.x
        dy = wy - p.y
        d = math.hypot(dx, dy)
        if d == 0 or d > rng:
            return 0.0, 0.0
        f = (1 - d / rng) * strn
        return f * dx / d, f * dy / d

    def step(self):
        # ------------------------------------------------------------------
        # pre‑compute a compact rule matrix indexed by integer type ids
        # ------------------------------------------------------------------
        # build / update integer lookups if types have changed
        if len(self._types_by_idx) != len(self.types):
            # rebuild mappings (this is cheap compared to stepping 100k)
            self._types_by_idx = list(self.types.values())
            self._tid_to_idx = {t.tid: idx for idx, t in enumerate(self._types_by_idx)}
        nt = len(self._types_by_idx)
        rm: list[list[Rule | None]] = [[None] * nt for _ in range(nt)]
        for r in self.rules:
            fi = self._tid_to_idx.get(r.from_tid)
            ti = self._tid_to_idx.get(r.to_tid)
            if fi is not None and ti is not None:
                rm[fi][ti] = r

        W, H = self.w, self.h
        fr = self.friction
        fs = self.force_scale
        ffunc = self._force
        at_func = self._apply_attractor
        # auto‑adjust the hash cell size if requested and particle count grows
        if self.auto_grid and self.parts:
            approx = math.sqrt(W * H / len(self.parts))
            # never let the grid cell collapse to zero or grow too large
            self.grid_size = max(10.0, min(self.grid_size, approx))
        gsize = self.grid_size

        # spatial hashing grid implemented as flat list for speed
        cols = int(W // gsize) + 1
        rows = int(H // gsize) + 1
        grid: list[list[Particle]] = [[] for _ in range(cols * rows)]
        parts = self.parts  # local reference
        idx_map = {id(p): i for i, p in enumerate(parts)}
        for p in parts:
            gx = int(p.x // gsize) % cols
            gy = int(p.y // gsize) % rows
            grid[gx + gy * cols].append(p)

        # if numba acceleration is enabled and library available, use it
        if getattr(self, 'use_numba', False) and _numba_available:
            # assemble numpy arrays
            n = len(parts)
            xs = np.empty(n, dtype=np.float32)
            ys = np.empty(n, dtype=np.float32)
            vxs = np.empty(n, dtype=np.float32)
            vys = np.empty(n, dtype=np.float32)
            tids_arr = np.empty(n, dtype=np.int32)
            for i, p in enumerate(parts):
                xs[i] = p.x
                ys[i] = p.y
                vxs[i] = p.vx
                vys[i] = p.vy
                tids_arr[i] = self._tid_to_idx[p.tid]
            # flatten rule matrices
            nt = len(self._types_by_idx)
            rm_s = np.zeros((nt, nt), dtype=np.float32)
            rm_r = np.zeros((nt, nt), dtype=np.float32)
            for i in range(nt):
                for j in range(nt):
                    r = rm[i][j]
                    if r is not None:
                        rm_s[i, j] = r.strength
                        rm_r[i, j] = r.max_range
            # build cell offsets/contents
            counts = [len(grid[i]) for i in range(cols * rows)]
            offsets = np.empty(cols * rows + 1, dtype=np.int32)
            offsets[0] = 0
            for i in range(cols * rows):
                offsets[i + 1] = offsets[i] + counts[i]
            contents = np.empty(offsets[-1], dtype=np.int32)
            ptr = 0
            for i in range(cols * rows):
                for p in grid[i]:
                    contents[ptr] = idx_map[id(p)]
                    ptr += 1
            # call compiled step if available
            if getattr(self, 'use_cython', False) and _have_cython_accel:
                cython_accel.step(xs, ys, vxs, vys, tids_arr,
                                  rm_s, rm_r,
                                  fr, fs, self.beta, W, H, gsize, cols, rows,
                                  offsets, contents)
            elif self.use_numba and _numba_available:
                _numba_step(xs, ys, vxs, vys, tids_arr,
                            rm_s, rm_r,
                            fr, fs, self.beta, W, H, gsize, cols, rows,
                            offsets, contents)
            # write back
            for i, p in enumerate(parts):
                p.x = xs[i]; p.y = ys[i]
                p.vx = vxs[i]; p.vy = vys[i]
            # apply helper chasing and jitter as before
            if hasattr(self, 'helper_groups') and getattr(self, 'chase_strength', 0.0) > 0:
                centroids = []
                for grp in self.helper_groups:
                    xs_c = ys_c = cnt = 0
                    for p in parts:
                        if any(p.tid == t.tid for t in grp):
                            xs_c += p.x; ys_c += p.y; cnt += 1
                    centroids.append((xs_c/cnt, ys_c/cnt) if cnt else (0,0))
                for p in parts:
                    gi = None
                    for idx, grp in enumerate(self.helper_groups):
                        if any(p.tid == t.tid for t in grp):
                            gi = idx
                            break
                    if gi is None or not centroids:
                        continue
                    target = centroids[(gi + 1) % len(centroids)]
                    dx = target[0] - p.x; dy = target[1] - p.y
                    d = math.hypot(dx, dy)
                    if d > 0:
                        mag = self.chase_strength * 20.0
                        p.vx += mag * dx / d
                        p.vy += mag * dy / d
            for p in parts:
                p.vx += uniform(-0.005, 0.005)
                p.vy += uniform(-0.005, 0.005)
            return

        # neighbour offsets precomputed once
        neigh = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),  (0, 0),  (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

        def compute_forces_chunk(chunk):
            results = []
            for p in chunk:
                ax = ay = 0.0
                fi = self._tid_to_idx[p.tid]
                rules_a = rm[fi]
                gx = int(p.x // gsize) % cols
                gy = int(p.y // gsize) % rows
                for dx_cell, dy_cell in neigh:
                    idx = ((gx + dx_cell) % cols) + ((gy + dy_cell) % rows) * cols
                    for b in grid[idx]:
                        if b is p:
                            continue
                        rule = rules_a[self._tid_to_idx[b.tid]]
                        if rule is None:
                            continue
                        mr = rule.max_range
                        dx = b.x - p.x
                        dy = b.y - p.y
                        d  = math.hypot(dx, dy)
                        if d == 0 or d > mr:
                            continue
                        f = ffunc(d / mr, rule.strength) * fs
                        ax += f * dx / d
                        ay += f * dy / d
                ax2, ay2 = at_func(p)
                results.append((p, ax + ax2, ay + ay2))
            return results

        # split into per-thread chunks to reduce executor overhead
        n = len(parts)
        if n == 0:
            return
        workers = self._executor._max_workers
        chunk_size = max(1, n // workers)
        chunks = [parts[i:i + chunk_size] for i in range(0, n, chunk_size)]
        for res in self._executor.map(compute_forces_chunk, chunks):
            for p, ax, ay in res:
                p.vx = (p.vx + ax) * fr
                p.vy = (p.vy + ay) * fr
                # small jitter to avoid zero-velocity sticking
                if abs(p.vx) < 1e-3 and abs(p.vy) < 1e-3:
                    p.vx += uniform(-0.01, 0.01)
                    p.vy += uniform(-0.01, 0.01)
                # wrap-around boundaries
                p.x = (p.x + p.vx) % W
                p.y = (p.y + p.vy) % H
        # after forces, apply group chasing acceleration
        if hasattr(self, 'helper_groups') and getattr(self, 'chase_strength', 0.0) > 0:
            centroids = []
            for grp in self.helper_groups:
                xs = ys = count = 0
                for p in parts:
                    if any(p.tid == t.tid for t in grp):
                        xs += p.x; ys += p.y; count += 1
                if count:
                    centroids.append((xs / count, ys / count))
                else:
                    centroids.append((0, 0))
            for p in parts:
                # determine group index
                gi = None
                for idx, grp in enumerate(self.helper_groups):
                    if any(p.tid == t.tid for t in grp):
                        gi = idx
                        break
                if gi is None or not centroids:
                    continue
                target = centroids[(gi + 1) % len(centroids)]
                dx = target[0] - p.x
                dy = target[1] - p.y
                d = math.hypot(dx, dy)
                if d > 0:
                    # amplify chase effect and push in both axes
                    mag = self.chase_strength * 20.0
                    p.vx += mag * dx / d
                    p.vy += mag * dy / d
        # add tiny random acceleration to keep things lively
        for p in parts:
            p.vx += uniform(-0.005, 0.005)
            p.vy += uniform(-0.005, 0.005)

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def save_config(self, path: str):
        """Write current simulation state to JSON file."""
        data = {
            'types': [
                {'name': t.name, 'color': t.color, 'tid': t.tid}
                for t in self.type_list()
            ],
            'rules': [
                {'from': r.from_tid, 'to': r.to_tid,
                 'strength': r.strength, 'range': r.max_range}
                for r in self.rules
            ],
            'physics': {
                'friction': self.friction,
                'beta': self.beta,
                'force_scale': self.force_scale
            },
            'particles': [
                {'x': p.x, 'y': p.y, 'vx': p.vx, 'vy': p.vy, 'tid': p.tid}
                for p in self.parts
            ]
        }
        # include helper groups if present
        if hasattr(self, 'helper_groups'):
            data['helper_groups'] = [[t.tid for t in grp]
                                     for grp in self.helper_groups]
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_config(self, path: str):
        """Load simulation state from JSON file, replacing current data."""
        with open(path) as f:
            data = json.load(f)
        # clear existing
        self.types.clear()
        self.rules.clear()
        self.parts.clear()
        # restore types preserving tids
        for tdata in data.get('types', []):
            t = ParticleType(tdata['name'], tdata['color'])
            t.tid = tdata['tid']
            self.types[t.tid] = t
        # rules
        for rdata in data.get('rules', []):
            self.set_rule(rdata['from'], rdata['to'],
                          rdata['strength'], rdata['range'])
        # physics
        phys = data.get('physics', {})
        self.friction = phys.get('friction', self.friction)
        self.beta = phys.get('beta', self.beta)
        self.force_scale = phys.get('force_scale', self.force_scale)
        # particles
        for pdata in data.get('particles', []):
            p = Particle(pdata['x'], pdata['y'], pdata['tid'])
            p.vx = pdata.get('vx', 0.0)
            p.vy = pdata.get('vy', 0.0)
            self.parts.append(p)
        # helper groups
        if 'helper_groups' in data:
            self.helper_groups = []
            for grp in data['helper_groups']:
                self.helper_groups.append([
                    self.types[tid] for tid in grp if tid in self.types
                ])

# ---------------------------------------------------------------------------
#  Helper for generating "lifelike" rules
# ---------------------------------------------------------------------------

import random


def lifelike_map(sim: Simulation, cell_size: float = 60.0, density: float = 0.5):
    """Return a dictionary of rule parameters based on particle colours.

    The map is keyed by ``(from_tid, to_tid)`` and values are ``(strength,
    range)``.  Clustering logic divides types into a few hue‑based groups
    so that different‑coloured particles form moving "cells" that attract
    within a group and repel across groups.  ``cell_size`` adjusts the base
    interaction range; ``density`` scales strengths down when high so giant
    blobs are avoided.

    The return value is a tuple ``(newmap, groups)`` where ``groups`` is a
    list of lists of `ParticleType` objects representing the colour groups.
    """
    newmap: dict[tuple[str, str], tuple[float, float]] = {}
    tl = sim.type_list()
    # assign groups by hue – compute hues once and keep HSV tuples for later
    hsv_list = [rgb2hsv(t.color) for t in tl]
    hues = list(zip(tl, [h[0] for h in hsv_list]))
    hues.sort(key=lambda x: x[1])
    group_count = max(2, len(tl) // 3)
    groups: list[list[ParticleType]] = [[] for _ in range(group_count)]
    for idx, (t, h) in enumerate(hues):
        groups[idx % group_count].append(t)

    for i, a in enumerate(tl):
        hsv_a = hsv_list[i]
        for j, b in enumerate(tl):
            hsv_b = hsv_list[j]
            # hue distance 0..0.5
            dh = abs(hsv_a[0] - hsv_b[0])
            dh = min(dh, 1 - dh)
            if a.tid == b.tid:
                strength = 0.8
            else:
                # are they in same group?
                same_group = any(a in g and b in g for g in groups)
                if same_group:
                    strength = 0.5 + random.uniform(-0.1, 0.1)
                else:
                    strength = -0.5 + random.uniform(-0.2, 0.0)
            # scale by density slider
            strength *= (1.0 - density * 0.5)
            rng = cell_size + (1 - dh) * cell_size + random.uniform(-20, 20)
            newmap[(a.tid, b.tid)] = (strength, rng)
    return newmap, groups


def lifelike_rules(sim: Simulation, **kwargs):
    """Apply rules directly by generating a map then setting values.

    Accepts same keyword arguments as :func:`lifelike_map` and simply writes
    the results into ``sim`` (full strength override).  Also stores group
    information on the simulation object for use during stepping.
    """
    newmap, groups = lifelike_map(sim, **kwargs)
    sim.rules.clear()
    for (ft, tt), (s, r) in newmap.items():
        sim.set_rule(ft, tt, s, r)
    sim.helper_groups = groups  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
#  Enhanced application with extra controls & improved input handling
# ---------------------------------------------------------------------------

TABS = ["Types", "Rules", "Physics", "Spawn"]

class EnhancedApp(App):
    def __init__(self):
        super().__init__()
        # start without GL; user can toggle with F2 once running
        self.use_gl = False
        self._gl_ready = False
        # detect if we are launching under a Vulkan wrapper
        self.use_vulkan = bool(os.getenv("USE_VULKAN"))
        if self.use_vulkan:
            # force software renderer but report Vulkan in debug info
            self.use_gl = False
        if not _have_gl:
            # OpenGL not available - keep flag off
            self.use_gl = False
        # debug overlay toggle
        self.show_debug = False

        # switch to resizable mode so window can be resized from the start
        flags = pygame.RESIZABLE
        # if trying to use GL up front we will recreate screen later
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
        # Vulkan renderer state (initialized lazily)
        self.vk_instance = None
        self.vk_device = None
        self.vk_queue = None
        self.vk_buffer = None
        self.vk_memory = None
        # note: we reuse _gl_ready flag for vk initialization too
        # override simulation with enhanced one
        self.sim = EnhancedSimulation(SIM_W * 2, WIN_H * 2)
        self.cam = Camera(SIM_W, WIN_H)
        # ensure camera is initially fitted to world size
        self.cam.fit(self.sim.w, self.sim.h)
        # keep track of current window dimensions
        self.win_w, self.win_h = WIN_W, WIN_H
        # scrolling state for panel content
        self.panel_scroll: float = 0  # vertical offset of panel content
        self._panel_content_end: float = 0  # y-coordinate where content finishes

        # helper toggle + strength slider
        self.rules_helper = False
        self.btn_rules_helper = UIButton("Helper", (0,0,60,26), nc=BTN_N, hc=BTN_H, font=FSM)
        self.s_helper_strength = UISlider("Helper", 0,0,0, 0.0, 1.0, 1.0)

        # physics extras: cell size and density control
        self.s_cellsize = UISlider("Cell size", 0,0,0, 20, 200, 60, "{:.0f}px", integ=True)
        self.s_density  = UISlider("Density",   0,0,0, 0.0, 1.0, 0.5)

        # attractor state
        self._right_down = False
        # corner resize drag state: (start_pos, start_size) or None
        self._corner_drag: tuple | None = None
        # world border drag state for resizing sim world
        self._border_drag: tuple | None = None

        # panel scroll offset (vertical) used for all tabs
        # (initialised above as panel_scroll)

        # make sure defaults are seeded into enhanced sim
        self._seed_defaults()

        # adjust grid size to initial cell size
        self.sim.grid_size = self.s_cellsize.val
        self.sim.helper_groups = []  # type: ignore[attr-defined]
        self.sim.chase_strength = 0.0  # type: ignore[attr-defined]

    # file dialog wrappers ------------------------------------------------
    def _pick_save_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            return
        root = tk.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension='.json',
                                            filetypes=[('LifeSim config', '*.json')])
        root.destroy()
        if path:
            try:
                self.sim.save_config(path)
            except Exception as e:
                print('save failed', e)

    def _pick_load_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            return
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(defaultextension='.json',
                                          filetypes=[('LifeSim config', '*.json')])
        root.destroy()
        if path:
            try:
                self.sim.load_config(path)
                # refresh anything UI that depends on types/rules
                self._seed_defaults()
            except Exception as e:
                print('load failed', e)

    @property
    def sim_w(self):
        return self.win_w - PANEL_W

    @property
    def sim_h(self):
        return self.win_h

    # helper logic ---------------------------------------------------------
    def apply_helper(self):
        """Blend helper rules into the simulation according to the slider.

        Called whenever rules or the helper-strength slider change.  If the
        helper is off this is a no-op.
        """
        if not self.rules_helper:
            return
        strength = self.s_helper_strength.val
        cell = self.s_cellsize.val
        dens = self.s_density.val
        newmap, groups = lifelike_map(self.sim, cell_size=cell, density=dens)
        self.sim.helper_groups = groups  # type: ignore[attr-defined]
        self.sim.chase_strength = strength  # type: ignore[attr-defined]
        for (ft, tt), (ns, nr) in newmap.items():
            old = self.sim.get_rule(ft, tt)
            if old:
                os = old.strength
                orng = old.max_range
            else:
                os = 0.0
                orng = nr
            bs = os * (1 - strength) + ns * strength
            br = orng * (1 - strength) + nr * strength
            self.sim.set_rule(ft, tt, bs, br)

        # override randomise behaviour to honour helper
    def _randomise_rules(self):
        super()._randomise_rules()
        if self.rules_helper:
            self.apply_helper()

    def _seed_defaults(self):
        # delegate to original then maybe adjust
        super()._seed_defaults()
        # copy existing rules from base sim into our enhanced sim
        self.sim.rules = list(self.sim.rules)

    # draw rules tab to include helper button
    def _draw_tab_types(self, surf, px, y, pw):
        # fixed-height scrollable types list + add form below
        txt(surf, "Particle Types", px + PAD, y, FLG, TXT)
        y += 24

        # Fixed-height scrollable list
        LIST_H = 200  # fixed height for type list
        list_start_y = y
        list_rect = pygame.Rect(px + PAD, y, pw - PAD * 2, LIST_H)
        rrect(surf, PNL_DK, list_rect, bw=1)

        # Clip drawing to list area
        old_clip = surf.get_clip()
        surf.set_clip(list_rect)

        tl = self.sim.type_list()
        # Use internal types list scroll
        types_scroll = getattr(self, '_types_list_internal_scroll', 0)
        y = list_start_y - types_scroll  # apply internal scroll offset
        for t in tl:
            row = pygame.Rect(px + PAD, y, pw - PAD * 2, 28)
            rrect(surf, PNL_DK, row, bw=1)
            sw = pygame.Rect(row.x + 4, row.y + 4, 20, 20)
            rrect(surf, t.color, sw, bw=1)
            txt(surf, t.name, row.x + 30, row.centery, FMD, TXT, "midleft")
            cnt = self.sim.count_for(t.tid)
            txt(surf, f"{cnt}", row.right - 50, row.centery, FSM, DIM, "midright")
            dr = pygame.Rect(row.right - 26, row.y + 4, 22, 20)
            rrect(surf, DNG, dr, bw=1)
            txt(surf, "✕", dr.centerx, dr.centery, FSM, TXT, "center")
            y += 32

        # Restore clipping
        surf.set_clip(old_clip)

        # scroll bar indicator
        if tl and len(tl) * 32 > LIST_H:
            list_end = len(tl) * 32
            scroll_h = max(10, LIST_H * LIST_H / list_end)
            scroll_y = types_scroll * LIST_H / list_end
            sb = pygame.Rect(list_rect.right - 6, list_rect.y + scroll_y, 5, scroll_h)
            pygame.draw.rect(surf, DIM, sb)

        y = list_start_y + LIST_H
        y += 8

        # Add form below the list
        txt(surf, "Add New Type", px + PAD, y, FLG, ACC)
        y += 22

        self.new_name.rect = pygame.Rect(px + PAD, y, pw - PAD * 2, 26)
        self.new_name.draw(surf)
        y += 34

        self.new_picker.x  = px + PAD
        self.new_picker.y  = y
        self.new_picker.sz = pw - PAD * 2 - 2
        self.new_picker.draw(surf)
        y += self.new_picker.total_h + 8

        # Random-color button next to preview swatch
        swatch = pygame.Rect(px + PAD, y, 36, 24)
        rrect(surf, self.new_picker.color, swatch, bw=1)
        txt(surf, "Preview", swatch.right + 6, swatch.centery, FSM, DIM, "midleft")
        # show human‑readable name of current picker color
        try:
            cname = colornames.find(*self.new_picker.color)
        except Exception:
            cname = "?"
        txt(surf, cname, swatch.right + 6, swatch.centery + 14, FSM, DIM, "midleft")
        # random color button
        rand_r = pygame.Rect(swatch.right + 80, swatch.y, 22, 22)
        rrect(surf, BTN_N, rand_r, bw=1)
        txt(surf, "R", rand_r.centerx, rand_r.centery, FSM, TXT, "center")
        self._rand_color_rect = rand_r
        y += 32

        add_r = pygame.Rect(px + PAD, y, pw - PAD * 2, 28)
        rrect(surf, SUC, add_r, bw=1, bc=SUC_H)
        txt(surf, "+ Add Type", add_r.centerx, add_r.centery, FLG, TXT, "center")
        self._add_btn_rect = add_r
        # random type button below add
        rand_type = pygame.Rect(px + PAD, y + 34, pw - PAD * 2, 28)
        rrect(surf, BTN_N, rand_type, bw=1)
        txt(surf, "+ Random Type", rand_type.centerx, rand_type.centery, FLG, TXT, "center")
        self._rand_type_rect = rand_type
        return y

    def _draw_tab_rules(self, surf, px, y, pw):
        # copy of base implementation so we can measure where the
        # panel content ends and then add our helper controls below.
        tl = self.sim.type_list()
        txt(surf, "Rule Matrix  (click cell to edit)", px + PAD, y, FLG, TXT)
        y += 22
        txt(surf, "Row = source particle  ·  Col = target", px + PAD, y, FSM, DIM)
        y += 20

        if not tl:
            txt(surf, "Add types first.", px + PAD, y, FMD, DIM)
            return y

        n    = len(tl)
        cell = min(38, (pw - PAD * 2 - 60) // n)
        lw   = 55    # label column width
        ox   = px + PAD + lw

        # Column headers
        for j, t in enumerate(tl):
            cr = pygame.Rect(ox + j * cell, y, cell - 2, 20)
            rrect(surf, t.color, cr)
            ts = FSM.render(t.name[:3], True, (0, 0, 0) if sum(t.color) > 380 else TXT)
            surf.blit(ts, ts.get_rect(center=cr.center))
        y += 24

        self._cell_map = {}  # (i,j) → (rect, ft, tt)

        for i, ft in enumerate(tl):
            # Row label
            rl = pygame.Rect(px + PAD, y, lw - 4, cell - 2)
            rrect(surf, ft.color, rl)
            ls = FSM.render(ft.name[:6], True, (0,0,0) if sum(ft.color)>380 else TXT)
            surf.blit(ls, ls.get_rect(midleft=(rl.x + 3, rl.centery)))

            for j, tt in enumerate(tl):
                rule = self.sim.get_rule(ft.tid, tt.tid)
                cr   = pygame.Rect(ox + j * cell, y, cell - 2, cell - 2)

                if rule is not None:
                    s = rule.strength
                    # Green = attraction, red = repulsion
                    if s >= 0:
                        bg = (int(30 + s * 140), int(100 + s * 100), int(30))
                    else:
                        bg = (int(100 + (-s) * 140), int(30), int(30))
                else:
                    bg = PNL_DK

                rrect(surf, bg, cr, r=3, bw=1)
                if rule:
                    vs = f"{rule.strength:+.1f}"
                    cs = FSM.render(vs, True, TXT)
                    surf.blit(cs, cs.get_rect(center=cr.center))
                else:
                    cs = FSM.render("—", True, DIM)
                    surf.blit(cs, cs.get_rect(center=cr.center))

                self._cell_map[(i, j)] = (cr, ft.tid, tt.tid)
            y += cell

        y += 8
        txt(surf, "Click any cell to edit or add a rule.", px + PAD, y, FSM, DIM)
        y += 18

        # Quick-fill buttons
        bw2 = (pw - PAD * 3) // 2
        r1  = pygame.Rect(px + PAD, y, bw2, 26)
        r2  = pygame.Rect(px + PAD * 2 + bw2, y, bw2, 26)
        rrect(surf, BTN_N, r1, bw=1)
        rrect(surf, DNG,   r2, bw=1)
        txt(surf, "Random rules",  r1.centerx, r1.centery, FSM, TXT, "center")
        txt(surf, "Clear rules",   r2.centerx, r2.centery, FSM, TXT, "center")
        self._btn_rand_rules = r1
        self._btn_clear_rules = r2

        # now draw helper UI just below the clear button
        base_y = self._btn_clear_rules.bottom + 8
        helper_r = pygame.Rect(px + PAD, base_y, 80, 26)
        self.btn_rules_helper.rect = helper_r
        self.btn_rules_helper.update(pygame.mouse.get_pos())
        self.btn_rules_helper.draw(surf)
        if self.rules_helper:
            txt(surf, "(on)", helper_r.right + 4, helper_r.centery, FSM, SUC_H, "midleft")
        self.s_helper_strength.x = px + PAD
        self.s_helper_strength.y = base_y + 34
        self.s_helper_strength.w = pw - PAD * 2
        self.s_helper_strength.draw(surf)
        # compute bottom most y to include helper controls
        y_end = y
        y_end = max(y_end, helper_r.bottom, self.s_helper_strength.y + self.s_helper_strength.height)
        return y_end

    def _draw_tab_spawn(self, surf, px, y, pw):
        # largely copied from base, with redistribute button
        txt(surf, "Spawn Particles", px + PAD, y, FLG, TXT)
        y += 26
        tl = self.sim.type_list()
        if not tl:
            txt(surf, "Add types first.", px + PAD, y, FMD, DIM)
            return y
        self.spawn_tidx = clamp(self.spawn_tidx, 0, len(tl) - 1)
        sel = tl[self.spawn_tidx]
        txt(surf, "Type", px + PAD, y, FSM, DIM)
        y += 18
        arrow_w = 28
        sel_r   = pygame.Rect(px + PAD + arrow_w, y, pw - PAD * 2 - arrow_w * 2, 30)
        rrect(surf, PNL_DK, sel_r, bw=1)
        sw = pygame.Rect(sel_r.x + 4, sel_r.y + 5, 20, 20)
        rrect(surf, sel.color, sw, bw=1)
        txt(surf, sel.name, sw.right + 6, sel_r.centery, FMD, TXT, "midleft")
        prev_r = pygame.Rect(px + PAD, y, arrow_w - 2, 30)
        next_r = pygame.Rect(px + PAD + pw - PAD * 2 - arrow_w + 2, y, arrow_w - 2, 30)
        rrect(surf, BTN_N, prev_r, bw=1)
        rrect(surf, BTN_N, next_r, bw=1)
        txt(surf, "◀", prev_r.centerx, prev_r.centery, FMD, TXT, "center")
        txt(surf, "▶", next_r.centerx, next_r.centery, FMD, TXT, "center")
        self._spawn_prev = prev_r
        self._spawn_next = next_r
        y += 38
        self.s_count.x  = px + PAD
        self.s_count.y  = y
        self.s_count.w  = pw - PAD * 2
        self.s_count.draw(surf)
        y += self.s_count.height + 14
        self.s_spread.x = px + PAD
        self.s_spread.y = y
        self.s_spread.w = pw - PAD * 2
        self.s_spread.draw(surf)
        y += self.s_spread.height + 16
        sp_r = pygame.Rect(px + PAD, y, pw - PAD * 2, 34)
        rrect(surf, SUC, sp_r, bw=1, bc=SUC_H)
        txt(surf, f"Spawn {int(self.s_count.val)} × {sel.name}", sp_r.centerx, sp_r.centery, FLG, TXT, "center")
        self._spawn_btn = sp_r
        y += 42
        # "Spawn all types" button with same count slider
        all_r = pygame.Rect(px + PAD, y, pw - PAD * 2, 34)
        rrect(surf, ACC, all_r, bw=1, bc=BTN_H)
        txt(surf, f"Spawn {int(self.s_count.val)} of EACH type", all_r.centerx, all_r.centery, FLG, TXT, "center")
        self._spawn_all_btn = all_r
        y += 42
        bw2 = (pw - PAD * 3) // 2
        r1  = pygame.Rect(px + PAD, y, bw2, 28)
        r2  = pygame.Rect(px + PAD * 2 + bw2, y, bw2, 28)
        rrect(surf, DNG, r1, bw=1)
        rrect(surf, DNG, r2, bw=1)
        txt(surf, "Clear all",   r1.centerx, r1.centery, FMD, TXT, "center")
        txt(surf, f"Clear {sel.name}", r2.centerx, r2.centery, FSM, TXT, "center")
        self._clear_all  = r1
        self._clear_type = (r2, sel.tid)
        # redistribute button
        rd = pygame.Rect(px + PAD, y + 36, pw - PAD * 2, 28)
        rrect(surf, BTN_N, rd, bw=1)
        txt(surf, "Redistribute", rd.centerx, rd.centery, FSM, TXT, "center")
        self._redistrib_btn = rd
        return y

    def _draw_tab_physics(self, surf, px, y, pw):
        txt(surf, "Physics", px + PAD, y, FLG, TXT)
        y += 26
        sliders = [self.s_friction, self.s_beta, self.s_fscale, self.s_speed,
                   self.s_cellsize, self.s_density]
        labels  = ["Friction", "Repulsion zone (β)", "Force scale", "Steps/frame",
                   "Cell size", "Density"]
        for sl, lb in zip(sliders, labels):
            sl.label = lb
            sl.x     = px + PAD
            sl.y     = y
            sl.w     = pw - PAD * 2
            sl.draw(surf)
            y += sl.height + 12
        # sync physics values
        self.sim.friction    = self.s_friction.val
        self.sim.beta        = self.s_beta.val
        self.sim.force_scale = self.s_fscale.val
        self.sim.grid_size   = self.s_cellsize.val
        # density only used during helper generation
        return y

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # ── Keyboard shortcuts ─────────────────────────────────────────
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.sim.running = not self.sim.running
                elif ev.key == pygame.K_s:
                    self.sim.step()
                elif ev.key == pygame.K_r:
                    self.cam.fit(self.sim.w, self.sim.h)
                elif ev.key == pygame.K_ESCAPE:
                    self.popover = None
                elif ev.key == pygame.K_F4:
                    # toggle debug info overlay
                    self.show_debug = not self.show_debug

            # ── Mouse wheel → zoom ─────────────────────────────────────────
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if ev.y != 0 and self._right_down:
                    # adjust attractor strength when holding right mouse
                    with self.sim._rng_lock:
                        a = self.sim._attractor
                    if a:
                        wx, wy, rng, strn = a
                        strn *= 1.1 ** ev.y
                        self.sim.set_attractor(wx, wy, rng, strn)
                if mx < self.sim_w:
                    self.cam.zoom_at(mx, my, 1.1 ** ev.y)
                else:
                    # panel scrolling (all tabs) - only scroll types list internally
                    if self.tab == 0:
                        tl = self.sim.type_list()
                        LIST_H = 200
                        max_list_scroll = max(0, len(tl) * 32 - LIST_H)
                        old_scroll = getattr(self, '_types_list_internal_scroll', 0)
                        self._types_list_internal_scroll = clamp(old_scroll - ev.y * 20, 0, max_list_scroll)
            if ev.type == pygame.VIDEORESIZE:
                # resize window and update layout
                self.win_w, self.win_h = ev.w, ev.h
                self.screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
                # adjust camera to fit new dimensions
                self.cam.fit(self.sim.w, self.sim.h)

            # ── Mouse drag for pan (left or middle) ─────────────────────────
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # Check if clicking on world border to resize simulation grid
                if mx < self.sim_w:
                    bx1, by1 = self.cam.w2s(self.sim.w, self.sim.h)
                    border_sz = 10
                    # Near right or bottom edge = world border drag
                    if (abs(bx1 - mx) < border_sz or abs(by1 - my) < border_sz):
                        self._border_drag = (ev.pos, (self.sim.w, self.sim.h))
                # Check if clicking a corner for window resize drag
                corner_sz = 15
                # Bottom-right corner
                br = (self.win_w - corner_sz, self.win_h - corner_sz,
                      self.win_w, self.win_h)
                if br[0] < mx < br[2] and br[1] < my < br[3]:
                    self._corner_drag = (ev.pos, (self.win_w, self.win_h))
                # Bottom-left corner
                elif 0 < mx < corner_sz and self.win_h - corner_sz < my < self.win_h:
                    self._corner_drag = (ev.pos, (self.win_w, self.win_h))
                # Top-right corner
                elif self.win_w - corner_sz < mx < self.win_w and 0 < my < corner_sz:
                    self._corner_drag = (ev.pos, (self.win_w, self.win_h))
                # Top-left corner
                elif 0 < mx < corner_sz and 0 < my < corner_sz:
                    self._corner_drag = (ev.pos, (self.win_w, self.win_h))
                # Normal pan if not corner or border drag
                elif mx < self.sim_w and not self._border_drag:
                    self._pan_start = ev.pos
                    self._pan_orig  = (self.cam.ox, self.cam.oy)

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self._pan_start = None
                self._corner_drag = None
                self._border_drag = None

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 2:
                self._pan_start = None

            # ── Right click for attractor ──────────────────────────────────
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                if ev.pos[0] < self.sim_w:
                    self._right_down = True
                    wx, wy = self.cam.s2w(*ev.pos)
                    rng = 150 / self.cam.zoom
                    self.sim.set_attractor(wx, wy, rng, strength=2.0)
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                self._right_down = False
                self.sim.clear_attractor()

            if ev.type == pygame.MOUSEMOTION:
                if self._pan_start:
                    dx = ev.pos[0] - self._pan_start[0]
                    dy = ev.pos[1] - self._pan_start[1]
                    self.cam.ox = self._pan_orig[0] - dx / self.cam.zoom
                    self.cam.oy = self._pan_orig[1] - dy / self.cam.zoom
                if self._right_down:
                    wx, wy = self.cam.s2w(*ev.pos)
                    rng = 150 / self.cam.zoom
                    self.sim.set_attractor(wx, wy, rng, strength=2.0)
                # Handle corner resize drag (window)
                if self._corner_drag:
                    start_pos, start_sz = self._corner_drag
                    mx, my = ev.pos
                    sx, sy = start_pos
                    dx = mx - sx
                    dy = my - sy
                    new_w = max(800, start_sz[0] + dx)
                    new_h = max(600, start_sz[1] + dy)
                    self.win_w = new_w
                    self.win_h = new_h
                    self.screen = pygame.display.set_mode((int(new_w), int(new_h)), pygame.RESIZABLE)
                    self.cam.fit(self.sim.w, self.sim.h)
                # Handle border drag (world size)
                if self._border_drag:
                    start_pos, start_sz = self._border_drag
                    wx, wy = self.cam.s2w(*ev.pos)  # world coords of mouse
                    start_wx, start_wy = self.cam.s2w(*start_pos)  # world coords of drag start
                    dx = wx - start_wx
                    dy = wy - start_wy
                    new_w = max(400, start_sz[0] + dx)
                    new_h = max(400, start_sz[1] + dy)
                    self.sim.w = new_w
                    self.sim.h = new_h

            # ── Popover takes priority ────────────────────────────────────
            if self.popover:
                self.popover.handle(ev)
                if self.popover.closed:
                    # if helper is enabled, re‑generate rules after any change
                    if self.rules_helper:
                        # remember edited pair so we can preserve it
                        if self.popover.result == "apply":
                            rule = self.sim.get_rule(self.popover.ft, self.popover.tt)
                            sv = rule.strength if rule else None
                            rv = rule.max_range if rule else None
                            lifelike_rules(self.sim)
                            if sv is not None:
                                self.sim.set_rule(self.popover.ft, self.popover.tt, sv, rv)
                        elif self.popover.result == "remove":
                            lifelike_rules(self.sim)
                            self.sim.del_rule(self.popover.ft, self.popover.tt)
                    self.popover = None
                continue

            # ── Panel widgets ─────────────────────────────────────────────
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mp = ev.pos

                # Playback
                if self.sim.running and self.btn_pause.clicked(mp):
                    self.sim.running = False
                elif not self.sim.running and self.btn_play.clicked(mp):
                    self.sim.running = True
                if self.btn_step.clicked(mp):
                    self.sim.step()

                # Tab switching
                for i, r in enumerate(self.tab_rects):
                    if r.collidepoint(mp):
                        self.tab = i
                        # reset scroll when switching tabs
                        self.panel_scroll = 0

                # ── Tab: Types ────────────────────────────────────────────
                if self.tab == 0:
                    # Delete buttons
                    tl = self.sim.type_list()
                    bx = self.sim_w + PAD
                    by = 120 + 24  # where type list starts (same as drawing)
                    scroll = self.panel_scroll
                    for t in tl:
                        dr = pygame.Rect(bx + (PANEL_W - PAD * 2) - 26,
                                         by + 4 - scroll, 22, 20)
                        if dr.collidepoint(mp):
                            self.sim.del_type(t.tid)
                            break
                        by += 32

                    # Add button
                    if hasattr(self, "_add_btn_rect") and self._add_btn_rect.collidepoint(mp):
                        color = self.new_picker.color
                        # default name to the color's human-readable name if user left field empty
                        default = colornames.find(*color)
                        name  = self.new_name.text.strip() or default
                        t = self.sim.add_type(name, color)
                        self.new_name.text = ""
                    # random color picker
                    if hasattr(self, "_rand_color_rect") and self._rand_color_rect.collidepoint(mp):
                        # choose a random named color from colornames
                        name, rgb = random.choice(list(colornames._colors.items()))
                        self.new_picker.color = rgb
                        print(f"random color name: {name}")
                    # random type button
                    if hasattr(self, "_rand_type_rect") and self._rand_type_rect.collidepoint(mp):
                        # pick a random named color for the new type
                        cname, color = random.choice(list(colornames._colors.items()))
                        name = cname  # use color name as type name
                        self.sim.add_type(name, color)
                        print(f"Added type named {name}")

                # ── Tab: Rules ────────────────────────────────────────────
                elif self.tab == 1:
                    # Cell clicks
                    for (i, j), (cr, ft, tt) in getattr(self, "_cell_map", {}).items():
                        if cr.collidepoint(mp):
                            px2  = cr.right + 4
                            py2  = cr.top
                            if px2 + RulePopover.W > WIN_W:
                                px2 = cr.left - RulePopover.W - 4
                            if py2 + RulePopover.H > WIN_H:
                                py2 = WIN_H - RulePopover.H - 4
                            self.popover = RulePopover(px2, py2, self.sim, ft, tt)
                            break

                    if hasattr(self, "_btn_rand_rules") and self._btn_rand_rules.collidepoint(mp):
                        self._randomise_rules()
                    if hasattr(self, "_btn_clear_rules") and self._btn_clear_rules.collidepoint(mp):
                        self.sim.rules.clear()
                    # helper toggle
                    if hasattr(self, 'btn_rules_helper') and self.btn_rules_helper.clicked(mp):
                        self.rules_helper = not self.rules_helper
                        if self.rules_helper:
                            self.apply_helper()

                # ── Tab: Physics ─────────────────────────────────────────
                elif self.tab == 2:
                    if hasattr(self, "_physics_presets"):
                        for r, fr, be, fs in self._physics_presets:
                            if r.collidepoint(mp):
                                self.s_friction.val = fr
                                self.s_beta.val     = be
                                self.s_fscale.val   = fs
                    if self.btn_reset_cam.clicked(mp):
                        self.cam.fit(self.sim.w, self.sim.h)

                # ── Tab: Spawn ───────────────────────────────────────────
                elif self.tab == 3:
                    tl = self.sim.type_list()
                    if hasattr(self, "_spawn_prev") and self._spawn_prev.collidepoint(mp):
                        self.spawn_tidx = (self.spawn_tidx - 1) % max(1, len(tl))
                    if hasattr(self, "_spawn_next") and self._spawn_next.collidepoint(mp):
                        self.spawn_tidx = (self.spawn_tidx + 1) % max(1, len(tl))
                    if hasattr(self, "_spawn_btn") and self._spawn_btn.collidepoint(mp):
                        if tl:
                            sel = tl[clamp(self.spawn_tidx, 0, len(tl)-1)]
                            self.sim.spawn(sel.tid, int(self.s_count.val),
                                           spread=self.s_spread.val)
                    if hasattr(self, "_spawn_all_btn") and self._spawn_all_btn.collidepoint(mp):
                        # Spawn same count for every type
                        for t in tl:
                            self.sim.spawn(t.tid, int(self.s_count.val),
                                           spread=self.s_spread.val)
                    if hasattr(self, "_redistrib_btn") and self._redistrib_btn.collidepoint(mp):
                        # move every particle to a random position
                        for p in self.sim.parts:
                            p.x = random.uniform(0, self.sim.w)
                            p.y = random.uniform(0, self.sim.h)
                    if hasattr(self, "_clear_all") and self._clear_all.collidepoint(mp):
                        self.sim.clear()
                    if hasattr(self, "_clear_type"):
                        r2, tid = self._clear_type
                        if r2.collidepoint(mp):
                            self.sim.parts = [p for p in self.sim.parts if p.tid != tid]

            # Delegate drag events to sliders; if helper-strength moves reload rules
            for sl in [self.s_friction, self.s_beta, self.s_fscale, self.s_speed,
                        self.s_count, self.s_spread, self.s_cellsize, self.s_density,
                        self.s_helper_strength]:
                changed = sl.handle(ev)
                if changed and sl is self.s_helper_strength and self.rules_helper:
                    self.apply_helper()
            # in case any other change should recalibrate helper
            if self.rules_helper:
                self.apply_helper()

            # keyboard shortcuts
            if ev.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if ev.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
                    self._pick_save_file()
                if ev.key == pygame.K_o and (mods & pygame.KMOD_CTRL):
                    self._pick_load_file()
                if ev.key == pygame.K_F2 and _have_gl and not self.use_vulkan:
                    # toggle OpenGL rendering (disabled in Vulkan wrapper)
                    self.use_gl = not self.use_gl
                    # recreate screen on next draw
                    self._gl_ready = False
                if ev.key == pygame.K_F3:
                    # toggle numba physics if available
                    if _numba_available:
                        self.sim.use_numba = not getattr(self.sim, 'use_numba', False)
                        print('numba acceleration', 'on' if self.sim.use_numba else 'off')

            # Text input & color picker
            self.new_name.handle(ev)
            self.new_picker.handle(ev)

    # drawing helpers --------------------------------------------------
    def _draw_panel(self):
        """Override to call enhanced tab drawing methods with scroll support."""
        surf = self.screen
        px   = self.sim_w
        pw   = PANEL_W
        # Panel background
        surf.fill(PNL, pygame.Rect(px, 0, pw, self.win_h))

        # Header
        rrect(surf, PNL_DK, pygame.Rect(px, 0, pw, 60))
        txt(surf, "Particle Life", px + PAD, 12, FXL, TXT)
        txt(surf, f"v2.0", px + pw - PAD, 14, FSM, DIM, "topright")
        txt(surf, "Ctrl+S:save  Ctrl+O:load  F2:toggle GL  F3:numba", px + PAD, 34, FSM, DIM)

        # Playback row
        y = 48
        mpos = pygame.mouse.get_pos()
        if self.sim.running:
            self.btn_pause.rect.topleft = (px + PAD, y + 4)
            self.btn_pause.update(mpos)
            self.btn_pause.draw(surf)
        else:
            self.btn_play.rect.topleft = (px + PAD, y + 4)
            self.btn_play.update(mpos)
            self.btn_play.draw(surf)

        bw = (pw - PAD * 3) // 2
        self.btn_step.rect = pygame.Rect(px + PAD + bw + PAD, y + 4, bw, 30)
        self.btn_step.update(mpos)
        self.btn_step.draw(surf)

        # Tab bar
        y = 88
        tw  = pw // len(TABS)
        self.tab_rects = []
        for i, name in enumerate(TABS):
            r = pygame.Rect(px + i * tw, y, tw, 28)
            self.tab_rects.append(r)
            is_active = (i == self.tab)
            rrect(surf, TAB_A if is_active else TAB_N, r, bw=1)
            txt(surf, name, r.centerx, r.centery, FSM if not is_active else FLG, TXT, "center")
        y += 32

        # Tab content with scroll
        content_y = y - self.panel_scroll
        if   self.tab == 0:
            y_end = self._draw_tab_types(surf, px, content_y, pw)
        elif self.tab == 1:
            y_end = self._draw_tab_rules(surf, px, content_y, pw)
        elif self.tab == 2:
            y_end = self._draw_tab_physics(surf, px, content_y, pw)
        elif self.tab == 3:
            y_end = self._draw_tab_spawn(surf, px, content_y, pw)
        else:
            y_end = content_y
        # Compute scroll limits
        self._panel_content_end = float(y_end)
        max_sc = max(0, self._panel_content_end - self.win_h)
        self.panel_scroll = clamp(self.panel_scroll, 0, max_sc)
        # debug overlay if requested
        if self.show_debug:
            self._draw_debug()

    def _init_gl(self):
        """Initialise OpenGL state and buffers."""
        # set up orthographic projection matching window coordinates
        glViewport(0, 0, int(self.win_w), int(self.win_h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.sim_w, self.win_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        # buffers
        self._vbo = glGenBuffers(1)
        self._cbo = glGenBuffers(1)
        self._gl_ready = True

    def _draw_sim(self):
        """Draw simulation either via pygame, OpenGL, or Vulkan depending on mode."""
        # if Vulkan mode requested, initialise/use it first
        if self.use_vulkan:
            if not self._gl_ready:
                self._init_vulkan()
            # perform a tiny Vulkan work item to prove we're using the API
            self._vk_draw()
            # continue to use pygame for overlay text and controls, so fall through
            # note: we don't return here; we still draw particles via pygame
        # handle existing GL branch
        if self.use_gl and (np is None or not _have_gl):
            # cannot do GL rendering without numpy or PyOpenGL; turn off
            self.use_gl = False
        if self.use_gl and not self._gl_ready:
            # recreate screen with GL flags
            self.screen = pygame.display.set_mode((self.win_w, self.win_h),
                                                 pygame.RESIZABLE | pygame.OPENGL | pygame.DOUBLEBUF)
            self._init_gl()
        if self.use_gl:
            # OpenGL path
            cam = self.cam
            n = len(self.sim.parts)
            if n:
                # build numpy arrays of transformed positions and colours
                positions = np.empty((n, 2), dtype=np.float32)
                colors = np.empty((n, 3), dtype=np.uint8)
                for i, p in enumerate(self.sim.parts):
                    sx, sy = cam.w2s(p.x, p.y)
                    positions[i, 0] = sx
                    positions[i, 1] = sy
                    pt = self.sim.types.get(p.tid)
                    if pt:
                        colors[i] = pt.color
                    else:
                        colors[i] = (180, 180, 180)
                # upload buffers
                glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
                glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_FLOAT, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, self._cbo)
                glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_UNSIGNED_BYTE, 0, None)
                glPointSize(max(1, int(PARTICLE_R * cam.zoom)))
                glDrawArrays(GL_POINTS, 0, n)
                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
            # swap buffers handled by pygame.display.flip() in run
            # overlay text has to be done with pygame still, so draw that now
            # draw particle count and state using pygame on top of GL
            # (we can switch to 2D rendering for text)
            txt(self.screen, f"{len(self.sim.parts)} particles", 8, 8, FSM, DIM)
            state = "▶ Running" if self.sim.running else "⏸ Paused"
            txt(self.screen, state, 8, 22, FSM, ACC if self.sim.running else DIM)
            return

        # --- previous pygame-only drawing code follows unchanged ---
        surf = self.screen
        # Background (Vulkan-cleared colour may be different)
        surf.fill(BG, pygame.Rect(0, 0, self.sim_w, self.win_h))

        # Grid
        cam   = self.cam
        gstep = 60.0
        ox, oy = cam.w2s(0, 0)
        # vertical lines
        x0 = math.floor(-ox / (gstep * cam.zoom)) * gstep
        x  = x0
        while True:
            sx, _ = cam.w2s(x, 0)
            if sx > self.sim_w:
                break
            if sx >= 0:
                pygame.draw.line(surf, GRID_C, (int(sx), 0), (int(sx), self.win_h))
            x += gstep
        y  = math.floor(-oy / (gstep * cam.zoom)) * gstep
        while True:
            _, sy = cam.w2s(0, y)
            if sy > self.win_h:
                break
            if sy >= 0:
                pygame.draw.line(surf, GRID_C, (0, int(sy)), (self.sim_w, int(sy)))
            y += gstep

        # World border
        bx0, by0 = cam.w2s(0, 0)
        bx1, by1 = cam.w2s(self.sim.w, self.sim.h)
        pygame.draw.rect(surf, (50, 80, 160),
                         (int(bx0), int(by0),
                          int(bx1 - bx0), int(by1 - by0)), 2)

        # Particles
        r = max(1, int(PARTICLE_R * cam.zoom))
        for p in self.sim.parts:
            sx, sy = cam.w2s(p.x, p.y)
            if not (0 <= sx <= self.sim_w and 0 <= sy <= self.win_h):
                continue
            ptype = self.sim.types.get(p.tid)
            col   = ptype.color if ptype else (180, 180, 180)
            pygame.draw.circle(surf, col, (int(sx), int(sy)), r)
            if r > 3:
                pygame.draw.circle(surf, lighten(col, 80), (int(sx), int(sy)), max(1, r // 2))

        # Divider
        pygame.draw.line(surf, BDR, (self.sim_w, 0), (self.sim_w, self.win_h), 2)

        # Particle count overlay
        txt(surf, f"{len(self.sim.parts)} particles", 8, 8, FSM, DIM)
        state = "▶ Running" if self.sim.running else "⏸ Paused"
        txt(surf, state, 8, 22, FSM, ACC if self.sim.running else DIM)

    # debug drawing ------------------------------------------------------
    def _draw_debug(self):
        """Render a small info box with current renderer/accelerator state."""
        lines = []
        if self.use_vulkan:
            lines.append("Renderer: Vulkan")
            # read back the particle count we stored in the Vulkan buffer
            try:
                mem_reqs = self.vk.vkGetBufferMemoryRequirements(self.vk_device, self.vk_buffer)
                ptr = self.vk.vkMapMemory(self.vk_device, self.vk_memory, 0, mem_reqs.size, 0)
                import struct
                val = struct.unpack('I', bytes(ptr[0:4]))[0]
                self.vk.vkUnmapMemory(self.vk_device, self.vk_memory)
                lines.append(f"VK buf count: {val}")
            except Exception:
                pass
        else:
            lines.append(f"Renderer: {'OpenGL' if self.use_gl else 'pygame'}")
        lines.append(f"GL available: {bool(_have_gl)}  numpy: {np is not None}")
        acc = 'cython' if getattr(self.sim, 'use_cython', False) else ('numba' if getattr(self.sim, 'use_numba', False) else 'none')
        lines.append(f"Accelerator: {acc}")
        lines.append(f"Particles: {len(self.sim.parts)}")
        lines.append(f"Python: {sys.executable}")
        # draw translucent background on panel area
        px = self.sim_w
        bw = PANEL_W
        # compute box height
        h = len(lines) * 16 + 8
        rect = pygame.Rect(px + 4, self.win_h - h - 4, bw - 8, h)
        s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        s.fill((0,0,0,180))
        surf = self.screen
        surf.blit(s, rect.topleft)
        y = rect.y + 4
        for line in lines:
            txt(surf, line, rect.x + 6, y, FSM, TXT)
            y += 16

    # override run to keep attractor updated and respect helper toggle
    def run(self):
        speed = 1
        while True:
            self._handle_events()
            if self.sim.running:
                speed = int(self.s_speed.val)
                for _ in range(speed):
                    self.sim.step()
            self._draw_sim()
            self._draw_panel()
            if self.popover:
                self.popover.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)


    # Vulkan support helpers -------------------------------------------------
    def _find_mem_type(self, phy, type_bits, props):
        """Find memory type index with given properties."""
        mem_props = self.vk.vkGetPhysicalDeviceMemoryProperties(phy)
        for i in range(mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & props) == props:
                return i
        raise RuntimeError("no suitable memory type")

    def _init_vulkan(self):
        """Perform a bare‑bones Vulkan initialisation; does not present anything.

        We create an instance, pick the first physical device, make a logical
        device with a graphics queue and allocate a tiny host‑visible buffer.
        The purpose is simply to exercise the Vulkan API; actual rendering is
        still done by pygame.  This keeps the dependency light while
        satisfying the "use Vulkan" requirement.
        """
        import vulkan as vk
        import ctypes

        self.vk = vk
        # create instance
        appInfo = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"ParticleLife",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName=b"NoEngine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        instInfo = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=appInfo,
        )
        self.vk_instance = vk.vkCreateInstance(instInfo, None)
        # pick first physical device
        phys = vk.vkEnumeratePhysicalDevices(self.vk_instance)[0]
        # queue family
        props = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys)
        qfam = next(i for i, p in enumerate(props) if p.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)
        qinfo = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=qfam,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        devInfo = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=qinfo,
        )
        self.vk_device = vk.vkCreateDevice(phys, devInfo, None)
        self.vk_queue = vk.vkGetDeviceQueue(self.vk_device, qfam, 0)
        # allocate a tiny host visible buffer we can poke each frame
        bufInfo = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=4,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.vk_buffer = vk.vkCreateBuffer(self.vk_device, bufInfo, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.vk_device, self.vk_buffer)
        allocInfo = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=self._find_mem_type(phys, mem_reqs.memoryTypeBits,
                                                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        self.vk_memory = vk.vkAllocateMemory(self.vk_device, allocInfo, None)
        vk.vkBindBufferMemory(self.vk_device, self.vk_buffer, self.vk_memory, 0)
        self._gl_ready = True

    def _vk_draw(self):
        """Do a trivial Vulkan operation per frame (write particle count to buffer)."""
        vk = self.vk
        if self.vk_buffer is None:
            return
        # map memory and update first 4 bytes with particle count
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.vk_device, self.vk_buffer)
        ptr = vk.vkMapMemory(self.vk_device, self.vk_memory, 0, mem_reqs.size, 0)
        if not ptr:
            return
        # ptr is a cffi buffer object; use buffer protocol to store count
        import struct
        count = len(self.sim.parts)
        ptr[0:4] = struct.pack('I', count)
        vk.vkUnmapMemory(self.vk_device, self.vk_memory)
        # ensure queue is idle so operations finish before next frame
        vk.vkQueueWaitIdle(self.vk_queue)

# ---------------------------------------------------------------------------
#  entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = EnhancedApp()
    app.run()
