#!/usr/bin/env python3
"""Enhanced version of Particle Life Simulator.

This module defines improved simulation and application classes that
implement optimisations, multithreading, new controls, and helper
functions. It is kept separate from ``main.py`` so that the base
version remains unchanged.

Run with ``python enhanced.py`` to launch the upgraded demo.
"""

import pygame, sys, math, uuid, threading
import random
from random import uniform
from colorsys import hsv_to_rgb, rgb_to_hsv
from concurrent.futures import ThreadPoolExecutor

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
        self.grid_size = grid_size
        self._attractor = None      # (x, y, radius, strength)
        self._rng_lock = threading.Lock()
        # reuse executor to avoid recreating threads every step
        self._executor = ThreadPoolExecutor()
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
        # build rule map for fast access
        rm = {}
        for r in self.rules:
            rm.setdefault(r.from_tid, {})[r.to_tid] = r

        W, H = self.w, self.h
        fr = self.friction
        fs = self.force_scale
        ffunc = self._force
        at_func = self._apply_attractor
        gsize = self.grid_size

        # spatial hashing grid
        grid: dict[tuple[int,int], list[Particle]] = {}
        parts = self.parts  # local reference
        for p in parts:
            gx = int(p.x // gsize)
            gy = int(p.y // gsize)
            grid.setdefault((gx, gy), []).append(p)

        def compute_forces(p):
            ax = ay = 0.0
            rules_a = rm.get(p.tid, {})
            gx = int(p.x // gsize)
            gy = int(p.y // gsize)
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    cell = grid.get((gx + dx_cell, gy + dy_cell), [])
                    for b in cell:
                        if b is p:
                            continue
                        rule = rules_a.get(b.tid)
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
            return p, ax + ax2, ay + ay2

        parts = self.parts
        # parallel execution using cached executor
        for p, ax, ay in self._executor.map(compute_forces, parts):
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
    # assign groups by hue
    hues = [(t, rgb2hsv(t.color)[0]) for t in tl]
    hues.sort(key=lambda x: x[1])
    group_count = max(2, len(tl) // 3)
    groups: list[list[ParticleType]] = [[] for _ in range(group_count)]
    for idx, (t, h) in enumerate(hues):
        groups[idx % group_count].append(t)

    for a in tl:
        hsv_a = rgb2hsv(a.color)
        for b in tl:
            hsv_b = rgb2hsv(b.color)
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
        # switch to resizable mode so window can be resized from the start
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
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
                        name  = self.new_name.text.strip() or f"Type {len(self.sim.types)+1}"
                        color = self.new_picker.color
                        t = self.sim.add_type(name, color)
                        self.new_name.text = ""
                    # random color picker
                    if hasattr(self, "_rand_color_rect") and self._rand_color_rect.collidepoint(mp):
                        # pick new random hue
                        h = random.random()
                        s = 1.0
                        v = 0.85
                        self.new_picker.hsv = (h, s, v)
                    # random type button
                    if hasattr(self, "_rand_type_rect") and self._rand_type_rect.collidepoint(mp):
                        name = f"Type {len(self.sim.types)+1}"
                        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                        self.sim.add_type(name, color)

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
        rrect(surf, PNL_DK, pygame.Rect(px, 0, pw, 48))
        txt(surf, "Particle Life", px + PAD, 12, FXL, TXT)
        txt(surf, f"v2.0", px + pw - PAD, 14, FSM, DIM, "topright")

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

    def _draw_sim(self):
        """Copy of base draw with dynamic dimensions."""
        surf = self.screen
        # Background
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

# ---------------------------------------------------------------------------
#  entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = EnhancedApp()
    app.run()
