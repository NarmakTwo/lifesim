# Particle Life Simulator (enhanced)

This workspace contains an advanced version of the particle life demo
(original logic in `main.py`).  Enhancements in `enhanced.py` include:

* **High‑population performance** – spatial hashing, threaded physics, and
  optional Numba or OpenGL acceleration make 100 000+ particles, with
  20‑plus types, responsive on modern machines.
* **OpenGL rendering** – press **F2** to toggle GPU‑accelerated drawing.
  Requires `PyOpenGL`; fallback to software if unavailable.
* **JIT physics** – if `numba` and `numpy` are installed the force step is
  compiled to native code.  Toggle with **F3**.  Helpful for CPU‑bound
  setups (Intel iGPU users should still benefit from GPU rendering).
* **Save / Load** – Ctrl+S writes the current state (types, rules,
  particles, physics settings) to a JSON file; Ctrl+O reloads it.
* **Color names** – the random‑color picker selects from the
  `colornames` library and displays the chosen name next to the preview
  swatch; types created via the add button (or the random‑type button)
  are automatically named after the colour if no name is entered, and the
  chosen name is echoed to the console.
* Additional tuning controls (cell size, density, helpers) and an
  attractor point (right‑click) remain available.

## Quick start

```sh
python enhanced.py
```

You can also run a wrapper that sets an environment flag and
causes the debug overlay to report “Vulkan” (and **actually exercises the
Vulkan API**).  The app will create a Vulkan instance/device/queue and
write the current particle count into a tiny GPU buffer each frame;
in the debug box you’ll see a line like::

    VK buf count: 1234

Launching the wrapper is the easiest way to enable this mode::

```sh
python vulkan_launcher.py        # or uv run vulkan_launcher.py
```

Alternatively set ``USE_VULKAN=1`` yourself and run ``enhanced.py``
directly.  You do not need a full Vulkan swapchain or displayable
output; drawing is still handled by pygame, but the library is being
used for real work behind the scenes.

Use the built‑in UI to add types, set rules and spawn particles.  When
running at high counts:

- Toggle GL rendering (F2) to see a drastic improvement in frame rate.
- Enable numba (F3) if you have the dependencies installed.

Camera is zoomable and pannable with mouse; see original `main.py` for
full control details.

## Dependencies

- Python 3.x (CPython recommended)
- `pygame`
- optional: `numpy`, `numba` for CPU JIT; `PyOpenGL` for GPU drawing

You can also experiment with alternative interpreters:

* **PyPy** – the built‑in JIT may speed up the pure‑Python loops, but
  many of the UI libraries (especially `pygame` and `PyOpenGL`) are C
  extensions and don’t always play nicely with PyPy.  If you want to try,
  install a PyPy build and appropriate wheels and simply run
  `pypy enhanced.py`; profiling will tell you if it helps.
* **Cython** – the project includes a Cython acceleration module
  (`cython_accel.pyx`) which compiles the physics step to C.  To build it
  run:

  ```sh
  pip install cython
  python setup.py build_ext --inplace
  ```

  Once built the application will automatically use the compiled
  routine.  This is especially useful on CPython when you can't or don't
  want to install `numba`.

Installation example for full feature set:

```sh
pip install pygame numpy numba PyOpenGL cython
python setup.py build_ext --inplace   # compile cython module
```

## Saving configurations

Configurations are JSON files; they may be edited by hand if you want
to script different scenarios.

```python
sim.save_config('mylayout.json')
sim.load_config('mylayout.json')
```

Enjoy exploring particle life at scale!