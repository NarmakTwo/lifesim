#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║         PARTICLE LIFE  —  Full-Featured Sim          ║
║  Zoom/Pan · Rule Matrix · Color Pickers · Live Edit  ║
╚══════════════════════════════════════════════════════╝

Controls:
  Scroll         Zoom in / out
  Right-drag     Pan
  Space          Play / Pause
  S              Step one frame
  R              Reset camera
  Escape         Close popover / deselect
"""

import pygame, sys, math, uuid
from random import uniform
from colorsys import hsv_to_rgb, rgb_to_hsv

pygame.init()
pygame.font.init()

# ── Window ─────────────────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1400, 900
PANEL_W       = 370
SIM_W         = WIN_W - PANEL_W
SIM_H         = WIN_H

# ── Physics defaults ───────────────────────────────────────────────────────────
DEFAULT_FRICTION    = 0.85
DEFAULT_BETA        = 0.3
DEFAULT_FORCE_SCALE = 1.0
PARTICLE_R          = 3

# ── Colour palette ─────────────────────────────────────────────────────────────
BG       = (  6,  23,  59)
GRID_C   = ( 15,  38,  80)
PNL      = (  9,  22,  56)
PNL_DK   = (  5,  14,  38)
BDR      = ( 32,  62, 128)
TXT      = (195, 215, 255)
DIM      = ( 95, 125, 170)
ACC      = ( 70, 130, 255)
BTN_N    = ( 20,  48, 108)
BTN_H    = ( 40,  78, 148)
BTN_A    = ( 62, 112, 215)
DNG      = (160,  45,  45)
DNG_H    = (205,  70,  70)
SUC      = ( 40, 148,  75)
SUC_H    = ( 60, 185,  95)
SLB      = ( 16,  40,  88)
SLF      = ( 62, 122, 232)
TAB_N    = ( 16,  40,  90)
TAB_A    = ( 45,  95, 190)

# ── Fonts ──────────────────────────────────────────────────────────────────────
FSM = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12)
FMD = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 14)
FLG = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 16, bold=True)
FXL = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 20, bold=True)

PAD = 8
CR  = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rrect(surf, col, rect, r=CR, bw=0, bc=None):
    pygame.draw.rect(surf, col, rect, border_radius=r)
    if bw:
        pygame.draw.rect(surf, bc or BDR, rect, bw, border_radius=r)


def txt(surf, s, x, y, font=FMD, col=TXT, anchor="topleft"):
    img = font.render(str(s), True, col)
    rct = img.get_rect(**{anchor: (x, y)})
    surf.blit(img, rct)
    return rct


def hsv2rgb(h, s, v):
    r, g, b = hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb2hsv(c):
    return rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lighten(col, amt=60):
    return tuple(min(255, c + amt) for c in col)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

class ParticleType:
    def __init__(self, name, color):
        self.name  = name
        self.color = tuple(color)
        self.tid   = str(uuid.uuid4())[:8]


class Rule:
    __slots__ = ("from_tid", "to_tid", "strength", "max_range")

    def __init__(self, ft, tt, strength, max_range):
        self.from_tid  = ft
        self.to_tid    = tt
        self.strength  = strength
        self.max_range = max_range


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "tid")

    def __init__(self, x, y, tid):
        self.x  = float(x)
        self.y  = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.tid = tid


class Simulation:
    def __init__(self, w, h):
        self.w = float(w)
        self.h = float(h)
        self.types:   dict[str, ParticleType] = {}
        self.rules:   list[Rule]              = []
        self.parts:   list[Particle]          = []
        self.friction     = DEFAULT_FRICTION
        self.beta         = DEFAULT_BETA
        self.force_scale  = DEFAULT_FORCE_SCALE
        self.running      = False

    # ── Types ──────────────────────────────────────────────────────────────────
    def add_type(self, name, color):
        t = ParticleType(name, color)
        self.types[t.tid] = t
        return t

    def del_type(self, tid):
        self.types.pop(tid, None)
        self.rules = [r for r in self.rules
                      if r.from_tid != tid and r.to_tid != tid]
        self.parts = [p for p in self.parts if p.tid != tid]

    # ── Rules ──────────────────────────────────────────────────────────────────
    def get_rule(self, ft, tt):
        for r in self.rules:
            if r.from_tid == ft and r.to_tid == tt:
                return r
        return None

    def set_rule(self, ft, tt, strength, max_range):
        r = self.get_rule(ft, tt)
        if r:
            r.strength  = strength
            r.max_range = max_range
        else:
            self.rules.append(Rule(ft, tt, strength, max_range))

    def del_rule(self, ft, tt):
        self.rules = [r for r in self.rules
                      if not (r.from_tid == ft and r.to_tid == tt)]

    # ── Spawning ───────────────────────────────────────────────────────────────
    def spawn(self, tid, count, cx=None, cy=None, spread=None):
        cx = cx if cx is not None else self.w / 2
        cy = cy if cy is not None else self.h / 2
        sp = spread if spread is not None else min(self.w, self.h) * 0.4
        # pre‑allocate local names for loop speed
        parts = []
        c1 = clamp
        u1 = uniform
        w, h = self.w, self.h
        for _ in range(count):
            x = c1(cx + u1(-sp, sp), 0, w)
            y = c1(cy + u1(-sp, sp), 0, h)
            parts.append(Particle(x, y, tid))
        self.parts.extend(parts)

    def clear(self):
        self.parts.clear()

    # ── Physics ────────────────────────────────────────────────────────────────
    def _force(self, rn, a):
        """Particle-life force function. rn = normalised distance [0,1]."""
        b = self.beta
        if rn < b:
            return rn / b - 1.0          # always repel at short range
        if rn < 1.0:
            return a * (1.0 - abs(2 * rn - 1 - b) / (1 - b))
        return 0.0

    def step(self):
        rm: dict[str, dict[str, Rule]] = {}
        for r in self.rules:
            rm.setdefault(r.from_tid, {})[r.to_tid] = r

        fs = self.force_scale
        fr = self.friction
        W, H = self.w, self.h
        parts = self.parts

        for a in parts:
            ax = ay = 0.0
            rules_a = rm.get(a.tid, {})
            for b in parts:
                if b is a:
                    continue
                rule = rules_a.get(b.tid)
                if rule is None:
                    continue
                mr = rule.max_range
                dx = b.x - a.x
                dy = b.y - a.y
                d  = math.hypot(dx, dy)
                if d == 0 or d > mr:
                    continue
                f   = self._force(d / mr, rule.strength) * fs
                ax += f * dx / d
                ay += f * dy / d

            a.vx = (a.vx + ax) * fr
            a.vy = (a.vy + ay) * fr
            a.x  = clamp(a.x + a.vx, 0, W)
            a.y  = clamp(a.y + a.vy, 0, H)

    # ── Convenience ────────────────────────────────────────────────────────────
    def type_list(self):
        """Return types in stable order."""
        return list(self.types.values())

    def count_for(self, tid):
        return sum(1 for p in self.parts if p.tid == tid)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

class Camera:
    def __init__(self, sw, sh):
        self.sw = sw
        self.sh = sh
        self.zoom = 1.0
        self.ox   = 0.0
        self.oy   = 0.0

    def w2s(self, wx, wy):
        return (wx - self.ox) * self.zoom, (wy - self.oy) * self.zoom

    def s2w(self, sx, sy):
        return sx / self.zoom + self.ox, sy / self.zoom + self.oy

    def zoom_at(self, sx, sy, factor):
        wx, wy   = self.s2w(sx, sy)
        self.zoom = clamp(self.zoom * factor, 0.04, 40.0)
        self.ox   = wx - sx / self.zoom
        self.oy   = wy - sy / self.zoom

    def pan(self, dx, dy):
        self.ox -= dx / self.zoom
        self.oy -= dy / self.zoom

    def fit(self, ww, wh):
        self.zoom = min(self.sw / ww, self.sh / wh)
        self.ox   = -(self.sw - ww * self.zoom) / (2 * self.zoom)
        self.oy   = -(self.sh - wh * self.zoom) / (2 * self.zoom)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class UIButton:
    def __init__(self, label, rect, nc=None, hc=None, font=FMD, icon=None):
        self.label = label
        self.rect  = pygame.Rect(rect)
        self.nc    = nc or BTN_N
        self.hc    = hc or BTN_H
        self.font  = font
        self.icon  = icon
        self._hov  = False

    def draw(self, surf):
        c = self.hc if self._hov else self.nc
        rrect(surf, c, self.rect, bw=1)
        s = self.font.render(self.label, True, TXT)
        surf.blit(s, s.get_rect(center=self.rect.center))

    def update(self, mpos):
        self._hov = self.rect.collidepoint(mpos)

    def clicked(self, mpos, button=1):
        return self.rect.collidepoint(mpos)


class UISlider:
    def __init__(self, label, x, y, w, lo, hi, val, fmt="{:.2f}", integ=False):
        self.label = label
        self.x, self.y, self.w = x, y, w
        self.lo, self.hi = lo, hi
        self.val   = val
        self.fmt   = fmt
        self.integ = integ
        self._drag = False
        self.H     = 12

    @property
    def track(self):
        return pygame.Rect(self.x, self.y + 16, self.w, self.H)

    def draw(self, surf):
        txt(surf, self.label, self.x, self.y, FSM, DIM)
        txt(surf, self.fmt.format(self.val), self.x + self.w, self.y, FSM, TXT, "topright")
        tr = self.track
        rrect(surf, SLB, tr, 4)
        t  = (self.val - self.lo) / (self.hi - self.lo)
        fw = max(self.H, int(t * self.w))
        rrect(surf, SLF, pygame.Rect(tr.x, tr.y, fw, tr.h), 4)
        cx = tr.x + int(t * self.w)
        pygame.draw.circle(surf, TXT, (cx, tr.centery), 8)
        pygame.draw.circle(surf, ACC, (cx, tr.centery), 6)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.track.inflate(20, 20).collidepoint(ev.pos):
                self._drag = True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._drag = False
        if ev.type == pygame.MOUSEMOTION and self._drag:
            t = clamp((ev.pos[0] - self.x) / self.w, 0.0, 1.0)
            v = self.lo + t * (self.hi - self.lo)
            self.val = round(v) if self.integ else v
            return True
        return False

    @property
    def height(self):
        return 16 + self.H + 4


class UITextInput:
    def __init__(self, rect, placeholder="", maxlen=24):
        self.rect        = pygame.Rect(rect)
        self.placeholder = placeholder
        self.text        = ""
        self.maxlen      = maxlen
        self.active      = False
        self._t          = 0

    def draw(self, surf):
        bc = ACC if self.active else BDR
        rrect(surf, PNL_DK, self.rect, bw=1, bc=bc)
        disp, col = (self.text, TXT) if self.text else (self.placeholder, DIM)
        s = FMD.render(disp, True, col)
        surf.blit(s, (self.rect.x + 6, self.rect.centery - s.get_height() // 2))
        if self.active:
            self._t += 1
            if self._t % 60 < 30:
                cx = self.rect.x + 6 + FMD.size(self.text)[0]
                pygame.draw.line(surf, TXT,
                                 (cx, self.rect.y + 4), (cx, self.rect.bottom - 4))

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(ev.pos)
        if ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.active = False
            elif len(self.text) < self.maxlen and ev.unicode.isprintable():
                self.text += ev.unicode


class ColorPicker:
    """Compact SV square + hue bar color picker."""

    def __init__(self, x, y, size=150):
        self.x, self.y, self.sz = x, y, size
        self.hsv  = (0.0, 1.0, 0.85)
        self._sv  = None
        self._hue = None
        self._dirty = True
        self._dsv   = False
        self._dhue  = False
        self.HUE_H  = 16
        self.GAP    = 6

    @property
    def sv_rect(self):
        return pygame.Rect(self.x, self.y, self.sz, self.sz)

    @property
    def hue_rect(self):
        return pygame.Rect(self.x, self.y + self.sz + self.GAP, self.sz, self.HUE_H)

    @property
    def total_h(self):
        return self.sz + self.GAP + self.HUE_H

    @property
    def color(self):
        return hsv2rgb(*self.hsv)

    @color.setter
    def color(self, rgb):
        self.hsv    = rgb2hsv(rgb)
        self._dirty = True

    def _build(self):
        sz  = self.sz
        sv  = pygame.Surface((sz, sz))
        h   = self.hsv[0]
        for xi in range(sz):
            sat = xi / max(sz - 1, 1)
            for yi in range(sz):
                val = 1.0 - yi / max(sz - 1, 1)
                sv.set_at((xi, yi), hsv2rgb(h, sat, val))
        self._sv = sv

        hw  = self.sz
        hh  = self.HUE_H
        hs  = pygame.Surface((hw, hh))
        for xi in range(hw):
            c = hsv2rgb(xi / max(hw - 1, 1), 1.0, 1.0)
            pygame.draw.line(hs, c, (xi, 0), (xi, hh - 1))
        self._hue   = hs
        self._dirty = False

    def draw(self, surf):
        if self._dirty or self._sv is None:
            self._build()
        surf.blit(self._sv, (self.x, self.y))
        pygame.draw.rect(surf, BDR, self.sv_rect, 1)
        surf.blit(self._hue, self.hue_rect.topleft)
        pygame.draw.rect(surf, BDR, self.hue_rect, 1)
        # SV cursor
        sx = int(self.hsv[1] * self.sz) + self.x
        sy = int((1.0 - self.hsv[2]) * self.sz) + self.y
        pygame.draw.circle(surf, (0, 0, 0), (sx, sy), 6, 2)
        pygame.draw.circle(surf, (255, 255, 255), (sx, sy), 5, 2)
        # Hue cursor
        hx = int(self.hsv[0] * self.sz) + self.x
        hy = self.hue_rect.centery
        pygame.draw.circle(surf, (0, 0, 0), (hx, hy), 7, 2)
        pygame.draw.circle(surf, (255, 255, 255), (hx, hy), 6, 2)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._dsv  = self.sv_rect.collidepoint(ev.pos)
            self._dhue = not self._dsv and self.hue_rect.collidepoint(ev.pos)
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._dsv = self._dhue = False
        if ev.type == pygame.MOUSEMOTION:
            if self._dsv:
                sat = clamp((ev.pos[0] - self.x) / self.sz, 0.0, 1.0)
                val = 1.0 - clamp((ev.pos[1] - self.y) / self.sz, 0.0, 1.0)
                self.hsv = (self.hsv[0], sat, val)
                return True
            if self._dhue:
                h = clamp((ev.pos[0] - self.x) / self.sz, 0.0, 1.0)
                self.hsv    = (h, self.hsv[1], self.hsv[2])
                self._dirty = True
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE-EDIT POPOVER
# ═══════════════════════════════════════════════════════════════════════════════

class RulePopover:
    """Floating popover to edit/create a rule between two types."""

    W = 230
    H = 180

    def __init__(self, x, y, sim, from_tid, to_tid):
        self.rect  = pygame.Rect(x, y, self.W, self.H)
        self.sim   = sim
        self.ft    = from_tid
        self.tt    = to_tid
        rule       = sim.get_rule(from_tid, to_tid)
        sv         = rule.strength  if rule else 0.0
        rv         = rule.max_range if rule else 100.0
        self.s_str = UISlider("Strength", x + PAD, y + 36,
                              self.W - PAD * 2, -1.0, 1.0, sv)
        self.s_rng = UISlider("Range",    x + PAD, y + 80,
                              self.W - PAD * 2, 10.0, 400.0, rv, "{:.0f}px")
        self.btn_apply = UIButton("Apply",  (x + PAD,           y + self.H - 38, 95, 28), nc=BTN_A, hc=BTN_H)
        self.btn_del   = UIButton("Remove", (x + self.W - 108,  y + self.H - 38, 100, 28), nc=DNG, hc=DNG_H)
        self.closed    = False
        self.result    = None   # "apply" | "remove" | None

    def draw(self, surf):
        rrect(surf, PNL_DK, self.rect, bw=1)
        ft = self.sim.types.get(self.ft)
        tt = self.sim.types.get(self.tt)
        fn = ft.name if ft else "?"
        tn = tt.name if tt else "?"
        label = f"{fn}  →  {tn}"
        txt(surf, label, self.rect.x + PAD, self.rect.y + PAD, FLG, TXT)
        self.s_str.draw(surf)
        self.s_rng.draw(surf)
        self.btn_apply.draw(surf)
        self.btn_del.draw(surf)

    def handle(self, ev):
        self.s_str.handle(ev)
        self.s_rng.handle(ev)
        mp = pygame.mouse.get_pos()
        self.btn_apply.update(mp)
        self.btn_del.update(mp)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.btn_apply.clicked(ev.pos):
                self.sim.set_rule(self.ft, self.tt,
                                  self.s_str.val, self.s_rng.val)
                self.result = "apply"
                self.closed = True
            elif self.btn_del.clicked(ev.pos):
                self.sim.del_rule(self.ft, self.tt)
                self.result = "remove"
                self.closed = True
            elif not self.rect.collidepoint(ev.pos):
                self.closed = True


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

TABS = ["Types", "Rules", "Physics", "Spawn"]

class App:
    def __init__(self):
        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Particle Life Simulator")
        self.clock   = pygame.time.Clock()

        self.sim     = Simulation(SIM_W * 2, WIN_H * 2)  # large world
        self.cam     = Camera(SIM_W, WIN_H)
        self.cam.fit(self.sim.w, self.sim.h)

        # Panel state
        self.tab         = 0          # current tab index
        self.tab_rects   = []
        self.popover: RulePopover | None = None

        # Dragging (camera pan)
        self._pan_start  = None
        self._pan_orig   = None

        # ── Types tab ─────────────────────────────────────────────────────────
        self.new_name    = UITextInput((SIM_W + PAD, 0, PANEL_W - PAD * 2, 26),
                                       placeholder="Type name…")
        self.new_picker  = ColorPicker(SIM_W + PAD, 0, PANEL_W - PAD * 2 - 2)

        # ── Physics tab ───────────────────────────────────────────────────────
        sx = SIM_W + PAD
        sw = PANEL_W - PAD * 2
        self.s_friction  = UISlider("Friction",    sx, 0, sw, 0.50, 1.00, DEFAULT_FRICTION)
        self.s_beta      = UISlider("Repulsion zone (β)", sx, 0, sw, 0.05, 0.80, DEFAULT_BETA)
        self.s_fscale    = UISlider("Force scale", sx, 0, sw, 0.10, 5.0,  DEFAULT_FORCE_SCALE)
        self.s_speed     = UISlider("Steps/frame", sx, 0, sw, 1, 10, 1, "{:.0f}", integ=True)

        # ── Spawn tab ─────────────────────────────────────────────────────────
        self.spawn_tidx  = 0    # index into sim.type_list()
        self.s_count     = UISlider("Count",  sx, 0, sw, 1, 500, 50, "{:.0f}", integ=True)
        self.s_spread    = UISlider("Spread", sx, 0, sw, 20, 800, 300, "{:.0f}px")

        # Playback buttons (always visible at top)
        bw = (PANEL_W - PAD * 3) // 2
        self.btn_play  = UIButton("▶  Play",  (SIM_W + PAD,          54, bw, 30), nc=SUC,  hc=SUC_H)
        self.btn_pause = UIButton("⏸  Pause", (SIM_W + PAD,          54, bw, 30), nc=BTN_N, hc=BTN_H)
        self.btn_step  = UIButton("Step →",   (SIM_W + PAD + bw + PAD, 54, bw, 30))
        self.btn_reset_cam = UIButton("Reset camera", (SIM_W + PAD, 0, PANEL_W - PAD * 2, 26))

        # seed some defaults
        self._seed_defaults()

    def _seed_defaults(self):
        """Populate a quick demo scene."""
        presets = [
            ("Red",    (220,  60,  60)),
            ("Green",  ( 60, 200,  80)),
            ("Blue",   ( 60, 120, 240)),
            ("Yellow", (230, 210,  40)),
        ]
        for name, col in presets:
            t = self.sim.add_type(name, col)
            self.sim.spawn(t.tid, 80)

        tl = self.sim.type_list()
        import random
        for a in tl:
            for b in tl:
                s = random.uniform(-0.8, 0.8)
                r = random.uniform(60, 200)
                self.sim.set_rule(a.tid, b.tid, s, r)

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAW: SIMULATION VIEWPORT
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_sim(self):
        surf = self.screen
        # Background
        surf.fill(BG, pygame.Rect(0, 0, SIM_W, WIN_H))

        # Grid
        cam   = self.cam
        gstep = 60.0
        ox, oy = cam.w2s(0, 0)
        # vertical lines
        x0 = math.floor(-ox / (gstep * cam.zoom)) * gstep
        x  = x0
        while True:
            sx, _ = cam.w2s(x, 0)
            if sx > SIM_W:
                break
            if sx >= 0:
                pygame.draw.line(surf, GRID_C, (int(sx), 0), (int(sx), WIN_H))
            x += gstep
        y  = math.floor(-oy / (gstep * cam.zoom)) * gstep
        while True:
            _, sy = cam.w2s(0, y)
            if sy > WIN_H:
                break
            if sy >= 0:
                pygame.draw.line(surf, GRID_C, (0, int(sy)), (SIM_W, int(sy)))
            y += gstep

        # World border
        bx0, by0 = cam.w2s(0, 0)
        bx1, by1 = cam.w2s(self.sim.w, self.sim.h)
        pygame.draw.rect(surf, (50, 80, 160),
                         (int(bx0), int(by0),
                          int(bx1 - bx0), int(by1 - by0)), 2)

        # Particles
        r = max(1, int(PARTICLE_R * cam.zoom))
        clip = pygame.Rect(0, 0, SIM_W, WIN_H)
        for p in self.sim.parts:
            sx, sy = cam.w2s(p.x, p.y)
            if not (0 <= sx <= SIM_W and 0 <= sy <= WIN_H):
                continue
            ptype = self.sim.types.get(p.tid)
            col   = ptype.color if ptype else (180, 180, 180)
            pygame.draw.circle(surf, col, (int(sx), int(sy)), r)
            if r > 3:
                pygame.draw.circle(surf, lighten(col, 80), (int(sx), int(sy)), max(1, r // 2))

        # Divider
        pygame.draw.line(surf, BDR, (SIM_W, 0), (SIM_W, WIN_H), 2)

        # Particle count overlay
        txt(surf, f"{len(self.sim.parts)} particles", 8, 8, FSM, DIM)
        state = "▶ Running" if self.sim.running else "⏸ Paused"
        txt(surf, state, 8, 22, FSM, ACC if self.sim.running else DIM)

    # ──────────────────────────────────────────────────────────────────────────
    #  DRAW: PANEL
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_panel(self):
        surf = self.screen
        px   = SIM_W
        pw   = PANEL_W

        # Panel background
        surf.fill(PNL, pygame.Rect(px, 0, pw, WIN_H))

        # ── Header ────────────────────────────────────────────────────────────
        rrect(surf, PNL_DK, pygame.Rect(px, 0, pw, 48))
        txt(surf, "Particle Life", px + PAD, 12, FXL, TXT)
        txt(surf, f"v2.0", px + pw - PAD, 14, FSM, DIM, "topright")

        # ── Playback row ──────────────────────────────────────────────────────
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

        # ── Tab bar ───────────────────────────────────────────────────────────
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

        # ── Tab content ───────────────────────────────────────────────────────
        content_y = y
        if   self.tab == 0: self._draw_tab_types(surf, px, content_y, pw)
        elif self.tab == 1: self._draw_tab_rules(surf, px, content_y, pw)
        elif self.tab == 2: self._draw_tab_physics(surf, px, content_y, pw)
        elif self.tab == 3: self._draw_tab_spawn(surf, px, content_y, pw)

    # ── Tab: Types ────────────────────────────────────────────────────────────

    def _draw_tab_types(self, surf, px, y, pw):
        txt(surf, "Particle Types", px + PAD, y, FLG, TXT)
        y += 24

        tl = self.sim.type_list()
        for t in tl:
            row = pygame.Rect(px + PAD, y, pw - PAD * 2, 28)
            rrect(surf, PNL_DK, row, bw=1)
            # colour swatch
            sw = pygame.Rect(row.x + 4, row.y + 4, 20, 20)
            rrect(surf, t.color, sw, bw=1)
            # name
            txt(surf, t.name, row.x + 30, row.centery, FMD, TXT, "midleft")
            # count
            cnt = self.sim.count_for(t.tid)
            txt(surf, f"{cnt}", row.right - 50, row.centery, FSM, DIM, "midright")
            # delete button
            dr = pygame.Rect(row.right - 26, row.y + 4, 22, 20)
            rrect(surf, DNG, dr, bw=1)
            txt(surf, "✕", dr.centerx, dr.centery, FSM, TXT, "center")
            y += 32

        # Add-type form
        sep = pygame.Rect(px + PAD, y, pw - PAD * 2, 1)
        pygame.draw.rect(surf, BDR, sep)
        y += 8

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

        # Colour preview swatch
        swatch = pygame.Rect(px + PAD, y, 36, 24)
        rrect(surf, self.new_picker.color, swatch, bw=1)
        txt(surf, "Preview", swatch.right + 6, swatch.centery, FSM, DIM, "midleft")
        y += 32

        add_r = pygame.Rect(px + PAD, y, pw - PAD * 2, 28)
        rrect(surf, SUC, add_r, bw=1, bc=SUC_H)
        txt(surf, "+ Add Type", add_r.centerx, add_r.centery, FLG, TXT, "center")
        self._add_btn_rect = add_r

    # ── Tab: Rules ────────────────────────────────────────────────────────────

    def _draw_tab_rules(self, surf, px, y, pw):
        tl = self.sim.type_list()
        txt(surf, "Rule Matrix  (click cell to edit)", px + PAD, y, FLG, TXT)
        y += 22
        txt(surf, "Row = source particle  ·  Col = target", px + PAD, y, FSM, DIM)
        y += 20

        if not tl:
            txt(surf, "Add types first.", px + PAD, y, FMD, DIM)
            return

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

    # ── Tab: Physics ──────────────────────────────────────────────────────────

    def _draw_tab_physics(self, surf, px, y, pw):
        txt(surf, "Physics", px + PAD, y, FLG, TXT)
        y += 26

        sliders = [self.s_friction, self.s_beta, self.s_fscale, self.s_speed]
        labels  = ["Friction", "Repulsion zone (β)", "Force scale", "Steps/frame"]
        for sl, lb in zip(sliders, labels):
            sl.label = lb
            sl.x     = px + PAD
            sl.y     = y
            sl.w     = pw - PAD * 2
            sl.draw(surf)
            y += sl.height + 12

        # Sync to sim
        self.sim.friction    = self.s_friction.val
        self.sim.beta        = self.s_beta.val
        self.sim.force_scale = self.s_fscale.val

        y += 8
        pygame.draw.rect(surf, BDR, (px + PAD, y, pw - PAD * 2, 1))
        y += 10

        txt(surf, "Presets", px + PAD, y, FLG, TXT)
        y += 26

        presets = [
            ("Chaos",    0.85, 0.3, 1.0),
            ("Smooth",   0.95, 0.2, 0.6),
            ("Jitter",   0.60, 0.4, 2.0),
            ("Crystals", 0.92, 0.15, 0.8),
        ]
        bw = (pw - PAD * 2 - PAD * (len(presets)-1)) // len(presets)
        for k, (name, fr, be, fs) in enumerate(presets):
            r = pygame.Rect(px + PAD + k * (bw + PAD), y, bw, 28)
            rrect(surf, BTN_N, r, bw=1)
            txt(surf, name, r.centerx, r.centery, FSM, TXT, "center")
            setattr(self, f"_preset_{k}", (r, fr, be, fs))
        self._physics_presets = [(getattr(self, f"_preset_{k}")) for k in range(len(presets))]
        y += 36

        # Reset camera
        self.btn_reset_cam.rect = pygame.Rect(px + PAD, y, pw - PAD * 2, 26)
        self.btn_reset_cam.update(pygame.mouse.get_pos())
        self.btn_reset_cam.draw(surf)

    # ── Tab: Spawn ────────────────────────────────────────────────────────────

    def _draw_tab_spawn(self, surf, px, y, pw):
        txt(surf, "Spawn Particles", px + PAD, y, FLG, TXT)
        y += 26

        tl = self.sim.type_list()
        if not tl:
            txt(surf, "Add types first.", px + PAD, y, FMD, DIM)
            return

        self.spawn_tidx = clamp(self.spawn_tidx, 0, len(tl) - 1)
        sel = tl[self.spawn_tidx]

        # Type selector
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

        # Count + spread sliders
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

        # Spawn button
        sp_r = pygame.Rect(px + PAD, y, pw - PAD * 2, 34)
        rrect(surf, SUC, sp_r, bw=1, bc=SUC_H)
        txt(surf, f"Spawn {int(self.s_count.val)} × {sel.name}", sp_r.centerx, sp_r.centery, FLG, TXT, "center")
        self._spawn_btn = sp_r
        y += 42

        # Clear / clear type buttons
        bw2 = (pw - PAD * 3) // 2
        r1  = pygame.Rect(px + PAD, y, bw2, 28)
        r2  = pygame.Rect(px + PAD * 2 + bw2, y, bw2, 28)
        rrect(surf, DNG, r1, bw=1)
        rrect(surf, DNG, r2, bw=1)
        txt(surf, "Clear all",   r1.centerx, r1.centery, FMD, TXT, "center")
        txt(surf, f"Clear {sel.name}", r2.centerx, r2.centery, FSM, TXT, "center")
        self._clear_all  = r1
        self._clear_type = (r2, sel.tid)

    # ──────────────────────────────────────────────────────────────────────────
    #  EVENT HANDLING
    # ──────────────────────────────────────────────────────────────────────────

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
                if mx < SIM_W:
                    self.cam.zoom_at(mx, my, 1.1 ** ev.y)

            # ── Right mouse drag → pan ─────────────────────────────────────
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                if ev.pos[0] < SIM_W:
                    self._pan_start = ev.pos
                    self._pan_orig  = (self.cam.ox, self.cam.oy)

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                self._pan_start = None

            if ev.type == pygame.MOUSEMOTION and self._pan_start:
                dx = ev.pos[0] - self._pan_start[0]
                dy = ev.pos[1] - self._pan_start[1]
                self.cam.ox = self._pan_orig[0] - dx / self.cam.zoom
                self.cam.oy = self._pan_orig[1] - dy / self.cam.zoom

            # ── Popover takes priority ────────────────────────────────────
            if self.popover:
                self.popover.handle(ev)
                if self.popover.closed:
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

                # ── Tab: Types ────────────────────────────────────────────
                if self.tab == 0:
                    # Delete buttons
                    tl = self.sim.type_list()
                    bx = SIM_W + PAD
                    by = 120 + 24  # where type list starts
                    for t in tl:
                        dr = pygame.Rect(bx + (PANEL_W - PAD * 2) - 26, by + 4, 22, 20)
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

                # ── Tab: Physics ──────────────────────────────────────────
                elif self.tab == 2:
                    if hasattr(self, "_physics_presets"):
                        for r, fr, be, fs in self._physics_presets:
                            if r.collidepoint(mp):
                                self.s_friction.val = fr
                                self.s_beta.val     = be
                                self.s_fscale.val   = fs
                    if self.btn_reset_cam.clicked(mp):
                        self.cam.fit(self.sim.w, self.sim.h)

                # ── Tab: Spawn ────────────────────────────────────────────
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
                    if hasattr(self, "_clear_all") and self._clear_all.collidepoint(mp):
                        self.sim.clear()
                    if hasattr(self, "_clear_type"):
                        r2, tid = self._clear_type
                        if r2.collidepoint(mp):
                            self.sim.parts = [p for p in self.sim.parts if p.tid != tid]

            # Delegate drag events to sliders
            for sl in [self.s_friction, self.s_beta, self.s_fscale, self.s_speed,
                        self.s_count, self.s_spread]:
                sl.handle(ev)

            # Text input & color picker
            self.new_name.handle(ev)
            self.new_picker.handle(ev)

    def _randomise_rules(self):
        import random
        tl = self.sim.type_list()
        for a in tl:
            for b in tl:
                s = random.uniform(-1.0, 1.0)
                r = random.uniform(40, 250)
                self.sim.set_rule(a.tid, b.tid, s, r)

    # ──────────────────────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        speed = 1
        while True:
            self._handle_events()

            # Simulate
            if self.sim.running:
                speed = int(self.s_speed.val)
                for _ in range(speed):
                    self.sim.step()

            # Render
            self._draw_sim()
            self._draw_panel()

            # Popover on top
            if self.popover:
                self.popover.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(60)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.run()