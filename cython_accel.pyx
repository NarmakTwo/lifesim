# cython_accel.pyx
# Cython implementation of the particle physics step; invoked from
# EnhancedSimulation when available.

from libc.math cimport hypot
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def step(double[:] xs, double[:] ys,
         double[:] vxs, double[:] vys,
         int[:] tids,
         double[:, :] rm_s, double[:, :] rm_r,
         double fr, double fs, double beta,
         double W, double H, double gsize,
         int cols, int rows,
         int[:] offsets, int[:] contents):
    cdef int n = xs.shape[0]
    cdef int i, j, fi, ti, base, start, end, idx, gx, gy, cgx, cgy
    cdef double x, y, ax, ay, strength, mr, dx, dy, d, rn, b, fval, vx, vy
    for i in range(n):
        x = xs[i]; y = ys[i]
        ax = 0.0; ay = 0.0
        fi = tids[i]
        gx = int(x // gsize) % cols
        gy = int(y // gsize) % rows
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cgx = (gx + dx) % cols
                cgy = (gy + dy) % rows
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
                    d = hypot(dx, dy)
                    if d == 0.0 or d > mr:
                        continue
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
        vx = (vxs[i] + ax) * fr
        vy = (vys[i] + ay) * fr
        x = (x + vx) % W
        y = (y + vy) % H
        vxs[i] = vx; vys[i] = vy
        xs[i] = x; ys[i] = y
