# Copyright 2026 National Research Nuclear University MEPhI (NRNU MEPhI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator

    def prange(*args):
        return range(*args)

try:
    import triangle as tr
except ImportError:
    tr = None

from .riemann import compute_shock_wave_speeds_batch, riemann_solve_batch


def _build_node_edge_adjacency(edges: np.ndarray, n_points: int):
    counts = np.zeros(n_points, dtype=np.intp)
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(counts, edges[:, 1], 1)

    offsets = np.empty(n_points + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    edge_indices = np.empty(offsets[-1], dtype=np.intp)
    cursor = offsets[:-1].copy()

    for edge_idx in range(len(edges)):
        p0 = int(edges[edge_idx, 0])
        p1 = int(edges[edge_idx, 1])
        edge_indices[cursor[p0]] = edge_idx
        cursor[p0] += 1
        edge_indices[cursor[p1]] = edge_idx
        cursor[p1] += 1

    return offsets, edge_indices


@njit(cache=True)
def _solve_symmetric_2x2_pinv_rhs(g11, g12, g22, b1, b2):
    trace = g11 + g22
    if trace <= 0.0:
        return 0.0, 0.0

    gap = np.sqrt((g11 - g22) * (g11 - g22) + 4.0 * g12 * g12)
    lam1 = 0.5 * (trace + gap)
    lam2 = 0.5 * (trace - gap)

    if lam1 < 0.0:
        lam1 = 0.0
    if lam2 < 0.0:
        lam2 = 0.0

    if g12 != 0.0:
        c1x = g12
        c1y = lam1 - g11
        c2x = lam1 - g22
        c2y = g12
        if c1x * c1x + c1y * c1y >= c2x * c2x + c2y * c2y:
            v1x = c1x
            v1y = c1y
        else:
            v1x = c2x
            v1y = c2y
    else:
        if g11 >= g22:
            v1x = 1.0
            v1y = 0.0
        else:
            v1x = 0.0
            v1y = 1.0

    norm_v1 = np.sqrt(v1x * v1x + v1y * v1y)
    if norm_v1 == 0.0:
        v1x = 1.0
        v1y = 0.0
        norm_v1 = 1.0
    v1x /= norm_v1
    v1y /= norm_v1

    v2x = -v1y
    v2y = v1x

    tol = np.finfo(np.float64).eps * max(trace, 1.0)
    inv1 = 1.0 / lam1 if lam1 > tol else 0.0
    inv2 = 1.0 / lam2 if lam2 > tol else 0.0

    c1 = v1x * b1 + v1y * b2
    c2 = v2x * b1 + v2y * b2

    x = inv1 * c1 * v1x + inv2 * c2 * v2x
    y = inv1 * c1 * v1y + inv2 * c2 * v2y
    return x, y


@njit(cache=True, parallel=True)
def _compute_node_velocities_pinv(Q_edges, normals, node_edge_offsets, node_edge_indices, n_points):
    node_vel = np.zeros((n_points, 2), dtype=np.float64)

    for node_idx in prange(n_points):
        start = node_edge_offsets[node_idx]
        end = node_edge_offsets[node_idx + 1]

        g11 = 0.0
        g12 = 0.0
        g22 = 0.0
        b1 = 0.0
        b2 = 0.0

        for pos in range(start, end):
            edge_idx = node_edge_indices[pos]
            nx = normals[edge_idx, 0]
            ny = normals[edge_idx, 1]
            q = Q_edges[edge_idx]

            g11 += nx * nx
            g12 += nx * ny
            g22 += ny * ny
            b1 += nx * q
            b2 += ny * q

        vx, vy = _solve_symmetric_2x2_pinv_rhs(g11, g12, g22, b1, b2)
        node_vel[node_idx, 0] = vx
        node_vel[node_idx, 1] = vy

    return node_vel


class BoundaryType(Enum):
    CIRCLE = "circle"
    DIAGONAL = "diagonal"


class BoundaryCondition(Enum):
    EXACT = "exact"
    REFLECTING = "reflecting"


@dataclass
class Geometry:
    points: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray | None = None
    normals: np.ndarray | None = None
    centers: np.ndarray | None = None
    edge_cells: np.ndarray | None = None
    edge_cell_signs: np.ndarray | None = None
    cell_L: np.ndarray | None = None
    cell_R: np.ndarray | None = None
    initial_distances: np.ndarray | None = None


class ShockMesh:
    def __init__(
        self,
        circle_center: tuple[float, float] = (0.0, 0.0),
        circle_radius: float = 1.0,
        line_point: tuple[float, float] = (0.0, 0.0),
        normal: tuple[float, float] = (1.0, 1.0),
        target_num_cells: int = 500,
        min_angle: float = 20.0,
        g: float = 1.0,
        alpha: float = 0.1,
        eps_rotated: float = 1e-8,
        boundary_condition: str = "reflecting",
        exact_solution=None,
    ):
        self.circle_center = tuple(float(v) for v in circle_center)
        self.circle_radius = float(circle_radius)
        if self.circle_radius <= 0.0:
            raise ValueError("circle_radius must be positive")

        cx, cy = self.circle_center
        r = self.circle_radius
        self.x_min = cx - r
        self.x_max = cx + r
        self.y_min = cy - r
        self.y_max = cy + r

        self.target_num_cells = int(target_num_cells)
        self.min_angle = float(min_angle)
        self.g = float(g)
        self.alpha = float(alpha)
        self.eps_rotated = float(eps_rotated)
        self.boundary_condition = BoundaryCondition(boundary_condition)
        self.exact_solution = exact_solution
        self.line_point = tuple(float(v) for v in line_point)
        self.normal = tuple(float(v) for v in normal)

        normal_arr = np.asarray(self.normal, dtype=float)
        normal_norm = np.linalg.norm(normal_arr)
        if normal_norm <= 0.0:
            raise ValueError("normal must be non-zero")
        self.unit_normal = normal_arr / normal_norm
        self.unit_tangent = np.array([-self.unit_normal[1], self.unit_normal[0]])

        self.max_area = self._estimate_max_area()

        self.geo: Geometry | None = None
        self.boundary_types: dict[int, BoundaryType] = {}
        self._bc_circle_idx = np.empty(0, dtype=np.intp)
        self._node_edge_offsets = np.empty(0, dtype=np.intp)
        self._node_edge_indices = np.empty(0, dtype=np.intp)
        self.initial_areas: np.ndarray | None = None
        self.cell_vars: np.ndarray | None = None
        self.time = 0.0
        self._prev_edge_velocity: np.ndarray | None = None

        self._generate_mesh()

    def _estimate_max_area(self) -> float:
        return np.pi * self.circle_radius ** 2 / self.target_num_cells

    def initialize(self, h0: Callable, u0: Callable = None, v0: Callable = None):
        cx = self.geo.centers[:, 0]
        cy = self.geo.centers[:, 1]
        self.cell_vars = np.zeros((len(cx), 3), dtype=float)
        self.cell_vars[:, 0] = np.vectorize(h0)(cx, cy)
        if u0 is not None:
            self.cell_vars[:, 1] = np.vectorize(u0)(cx, cy)
        if v0 is not None:
            self.cell_vars[:, 2] = np.vectorize(v0)(cx, cy)
        self.time = 0.0
        self._prev_edge_velocity = None

    def evolve(self, dt: float, moving_mesh: bool = True) -> float:
        if self.cell_vars is None:
            raise ValueError("initialize() must be called before evolve()")
        if moving_mesh:
            self._step_ale(dt)
        else:
            self._step_static(dt)
        self.time += dt
        return dt

    def compute_cfl_dt(self, cfl: float = 0.5, moving_mesh: bool = False) -> float:
        if moving_mesh:
            return self._compute_ale_cfl_dt(cfl)
        h = self.cell_vars[:, 0]
        u = self.cell_vars[:, 1]
        v = self.cell_vars[:, 2]
        speed = np.sqrt(u ** 2 + v ** 2) + np.sqrt(self.g * h)
        char_size = np.sqrt(self.areas)
        return cfl * np.min(char_size / speed)

    def _step_static(self, dt: float):
        fluxes = self._solve_all_riemann_rotated(self.geo.normals, edge_velocities=None)
        lengths = self._edge_lengths()
        Q_cons = self._to_conservative()
        contribution = fluxes * lengths[:, np.newaxis] * dt
        self._scatter_contributions(Q_cons, contribution, self.areas, divide_by_area=True)
        self._from_conservative(Q_cons)
        self._prev_edge_velocity = None

    def _step_ale(self, dt: float):
        old_points = self.geo.points.copy()
        old_areas = self.areas.copy()
        old_normals = self.geo.normals.copy()
        old_lengths = self._edge_lengths()

        Q_physical = self._compute_Q_physical_vectorized(old_normals, old_areas)
        node_velocities = self._compute_node_velocities_vectorized(Q_physical, old_normals)

        new_points = old_points + node_velocities * dt
        self._apply_boundary_constraints(new_points)
        self.geo.points = new_points
        self._update_geometry()

        new_areas = self.areas
        new_normals = self.geo.normals
        new_lengths = self._edge_lengths()

        mid_lengths = 0.5 * (old_lengths + new_lengths)
        mid_normals = self._average_normals(old_normals, new_normals)
        Q_swept = self._compute_Q_swept_vectorized(old_points, new_points, dt, mid_lengths)
        fluxes = self._solve_all_riemann_rotated(mid_normals, edge_velocities=Q_swept)
        self._apply_ale_update(fluxes, mid_lengths, old_areas, new_areas, dt)
        self._prev_edge_velocity = Q_swept.copy()

    def _prepare_riemann_states(self, normals, edge_velocities=None):
        n_edges = len(self.geo.edges)
        cell_L = self.geo.cell_L
        cell_R = self.geo.cell_R
        h_L = np.empty(n_edges)
        un_L = np.empty(n_edges)
        vt_L = np.empty(n_edges)
        h_R = np.empty(n_edges)
        un_R = np.empty(n_edges)
        vt_R = np.empty(n_edges)
        nx = normals[:, 0]
        ny = normals[:, 1]

        internal = (cell_L >= 0) & (cell_R >= 0)
        idx_int = np.where(internal)[0]
        if len(idx_int) > 0:
            cL, cR = cell_L[idx_int], cell_R[idx_int]
            nxi, nyi = nx[idx_int], ny[idx_int]
            qL, qR = self.cell_vars[cL], self.cell_vars[cR]

            h_L[idx_int] = qL[:, 0]
            un_L[idx_int] = qL[:, 1] * nxi + qL[:, 2] * nyi
            vt_L[idx_int] = -qL[:, 1] * nyi + qL[:, 2] * nxi

            h_R[idx_int] = qR[:, 0]
            un_R[idx_int] = qR[:, 1] * nxi + qR[:, 2] * nyi
            vt_R[idx_int] = -qR[:, 1] * nyi + qR[:, 2] * nxi

        idx_bnd = np.where(~internal)[0]
        if len(idx_bnd) > 0:
            self._fill_boundary_states_vectorized(
                idx_bnd, cell_L, cell_R, nx, ny, h_L, un_L, vt_L, h_R, un_R, vt_R
            )

        xi = np.zeros(n_edges) if edge_velocities is None else edge_velocities.copy()
        return h_L, un_L, vt_L, h_R, un_R, vt_R, xi

    def _evaluate_exact(self, x, y, t):
        if self.exact_solution is None:
            raise ValueError("exact_solution is required for exact boundary conditions")
        if hasattr(self.exact_solution, "evaluate"):
            return self.exact_solution.evaluate(x, y, t)
        return self.exact_solution(x, y, t)

    def _fill_boundary_states_vectorized(
        self, idx_bnd, cell_L, cell_R, nx, ny, h_L, un_L, vt_L, h_R, un_R, vt_R
    ):
        edges = self.geo.edges
        cL, cR = cell_L[idx_bnd], cell_R[idx_bnd]
        nxi, nyi = nx[idx_bnd], ny[idx_bnd]
        has_left = cL >= 0
        inner_cell = np.where(has_left, cL, cR)
        q_in = self.cell_vars[inner_cell]

        if self.boundary_condition == BoundaryCondition.EXACT:
            edge_mids = 0.5 * (self.geo.points[edges[idx_bnd, 0]] + self.geo.points[edges[idx_bnd, 1]])
            h_g, u_g, v_g = self._evaluate_exact(edge_mids[:, 0], edge_mids[:, 1], self.time)
            q_ghost = np.column_stack(
                [np.asarray(h_g, dtype=float), np.asarray(u_g, dtype=float), np.asarray(v_g, dtype=float)]
            )
        elif self.boundary_condition == BoundaryCondition.REFLECTING:
            un_in = q_in[:, 1] * nxi + q_in[:, 2] * nyi
            vt_in = -q_in[:, 1] * nyi + q_in[:, 2] * nxi
            u_ghost = -un_in * nxi - vt_in * nyi
            v_ghost = -un_in * nyi + vt_in * nxi
            q_ghost = np.column_stack([q_in[:, 0], u_ghost, v_ghost])
        else:
            q_ghost = q_in.copy()

        qL = np.where(has_left[:, np.newaxis], q_in, q_ghost)
        qR = np.where(has_left[:, np.newaxis], q_ghost, q_in)

        h_L[idx_bnd] = qL[:, 0]
        un_L[idx_bnd] = qL[:, 1] * nxi + qL[:, 2] * nyi
        vt_L[idx_bnd] = -qL[:, 1] * nyi + qL[:, 2] * nxi
        h_R[idx_bnd] = qR[:, 0]
        un_R[idx_bnd] = qR[:, 1] * nxi + qR[:, 2] * nyi
        vt_R[idx_bnd] = -qR[:, 1] * nyi + qR[:, 2] * nxi

    def _prepare_global_states(self, idx):
        cL = self.geo.cell_L[idx]
        cR = self.geo.cell_R[idx]
        n = len(idx)
        qL = np.empty((n, 3))
        qR = np.empty((n, 3))

        internal = (cL >= 0) & (cR >= 0)
        i_int = np.where(internal)[0]
        if len(i_int) > 0:
            qL[i_int] = self.cell_vars[cL[i_int]]
            qR[i_int] = self.cell_vars[cR[i_int]]

        i_bnd = np.where(~internal)[0]
        if len(i_bnd) > 0:
            real_idx = idx[i_bnd]
            cL_b, cR_b = cL[i_bnd], cR[i_bnd]
            has_left = cL_b >= 0
            inner_cell = np.where(has_left, cL_b, cR_b)
            q_in = self.cell_vars[inner_cell]

            if self.boundary_condition == BoundaryCondition.EXACT:
                edge_mids = 0.5 * (self.geo.points[self.geo.edges[real_idx, 0]] + self.geo.points[self.geo.edges[real_idx, 1]])
                h_g, u_g, v_g = self._evaluate_exact(edge_mids[:, 0], edge_mids[:, 1], self.time)
                q_ghost = np.column_stack(
                    [np.asarray(h_g, dtype=float), np.asarray(u_g, dtype=float), np.asarray(v_g, dtype=float)]
                )
            elif self.boundary_condition == BoundaryCondition.REFLECTING:
                nxi = self.geo.normals[real_idx, 0]
                nyi = self.geo.normals[real_idx, 1]
                un_in = q_in[:, 1] * nxi + q_in[:, 2] * nyi
                vt_in = -q_in[:, 1] * nyi + q_in[:, 2] * nxi
                u_ghost = -un_in * nxi - vt_in * nyi
                v_ghost = -un_in * nyi + vt_in * nxi
                q_ghost = np.column_stack([q_in[:, 0], u_ghost, v_ghost])
            else:
                q_ghost = q_in.copy()

            qL[i_bnd] = np.where(has_left[:, np.newaxis], q_in, q_ghost)
            qR[i_bnd] = np.where(has_left[:, np.newaxis], q_ghost, q_in)

        return qL, qR

    def _solve_all_riemann_rotated(self, normals, edge_velocities=None):
        n_edges = len(self.geo.edges)
        fluxes = np.zeros((n_edges, 3))
        hL, unL, vtL, hR, unR, vtR, xi = self._prepare_riemann_states(normals, edge_velocities)

        cell_L = self.geo.cell_L
        cell_R = self.geo.cell_R
        internal = (cell_L >= 0) & (cell_R >= 0)
        du_global = np.zeros(n_edges)
        dv_global = np.zeros(n_edges)
        idx_int = np.where(internal)[0]
        if len(idx_int) > 0:
            du_global[idx_int] = self.cell_vars[cell_R[idx_int], 1] - self.cell_vars[cell_L[idx_int], 1]
            dv_global[idx_int] = self.cell_vars[cell_R[idx_int], 2] - self.cell_vars[cell_L[idx_int], 2]

        dv_mag = np.sqrt(du_global ** 2 + dv_global ** 2)
        standard_mask = dv_mag < self.eps_rotated
        rotated_mask = ~standard_mask

        idx_std = np.where(standard_mask)[0]
        if len(idx_std) > 0:
            h_s, un_s, vt_s = riemann_solve_batch(
                hL[idx_std],
                unL[idx_std],
                vtL[idx_std],
                hR[idx_std],
                unR[idx_std],
                vtR[idx_std],
                xi[idx_std],
                self.g,
            )
            f = self._compute_fluxes_vectorized(h_s, un_s, vt_s, normals[idx_std])
            U = self._compute_U_stars_vectorized(h_s, un_s, vt_s, normals[idx_std])
            fluxes[idx_std] = f - xi[idx_std, np.newaxis] * U

        idx_rot = np.where(rotated_mask)[0]
        if len(idx_rot) > 0:
            fluxes[idx_rot] = self._solve_rotated_batch(idx_rot, normals, xi, du_global, dv_global, dv_mag)
        return fluxes

    def _solve_rotated_batch(self, idx_rot, normals, xi, du_global, dv_global, dv_mag):
        dm = dv_mag[idx_rot]
        n1 = np.column_stack([du_global[idx_rot], dv_global[idx_rot]]) / dm[:, np.newaxis]
        n_edge = normals[idx_rot]

        alpha = np.sum(n_edge * n1, axis=1)
        flip1 = alpha < 0
        n1[flip1] *= -1
        alpha = np.abs(alpha)

        n2 = np.column_stack([-n1[:, 1], n1[:, 0]])
        beta = np.sum(n_edge * n2, axis=1)
        flip2 = beta < 0
        n2[flip2] *= -1
        beta = np.abs(beta)

        xi_1 = xi[idx_rot] * alpha
        xi_2 = xi[idx_rot] * beta
        qL_g, qR_g = self._prepare_global_states(idx_rot)

        hL1 = qL_g[:, 0]
        unL1 = qL_g[:, 1] * n1[:, 0] + qL_g[:, 2] * n1[:, 1]
        vtL1 = -qL_g[:, 1] * n1[:, 1] + qL_g[:, 2] * n1[:, 0]
        hR1 = qR_g[:, 0]
        unR1 = qR_g[:, 1] * n1[:, 0] + qR_g[:, 2] * n1[:, 1]
        vtR1 = -qR_g[:, 1] * n1[:, 1] + qR_g[:, 2] * n1[:, 0]

        hL2 = qL_g[:, 0]
        unL2 = qL_g[:, 1] * n2[:, 0] + qL_g[:, 2] * n2[:, 1]
        vtL2 = -qL_g[:, 1] * n2[:, 1] + qL_g[:, 2] * n2[:, 0]
        hR2 = qR_g[:, 0]
        unR2 = qR_g[:, 1] * n2[:, 0] + qR_g[:, 2] * n2[:, 1]
        vtR2 = -qR_g[:, 1] * n2[:, 1] + qR_g[:, 2] * n2[:, 0]

        hs1, uns1, vts1 = riemann_solve_batch(hL1, unL1, vtL1, hR1, unR1, vtR1, xi_1, self.g)
        hs2, uns2, vts2 = riemann_solve_batch(hL2, unL2, vtL2, hR2, unR2, vtR2, xi_2, self.g)

        f1 = self._compute_fluxes_vectorized(hs1, uns1, vts1, n1)
        f2 = self._compute_fluxes_vectorized(hs2, uns2, vts2, n2)
        U1 = self._compute_U_stars_vectorized(hs1, uns1, vts1, n1)
        U2 = self._compute_U_stars_vectorized(hs2, uns2, vts2, n2)
        return alpha[:, np.newaxis] * (f1 - xi_1[:, np.newaxis] * U1) + beta[:, np.newaxis] * (f2 - xi_2[:, np.newaxis] * U2)

    def _compute_fluxes_vectorized(self, h, un, ut, normals):
        nx, ny = normals[:, 0], normals[:, 1]
        F_h = h * un
        F_hun = h * un ** 2 + 0.5 * self.g * h ** 2
        F_hut = h * un * ut
        return np.column_stack([F_h, F_hun * nx - F_hut * ny, F_hun * ny + F_hut * nx])

    def _compute_U_stars_vectorized(self, h, un, ut, normals):
        nx, ny = normals[:, 0], normals[:, 1]
        u = un * nx - ut * ny
        v = un * ny + ut * nx
        return np.column_stack([h, h * u, h * v])

    def _compute_Q_physical_vectorized(self, normals, areas):
        n_edges = len(self.geo.edges)
        cell_L = self.geo.cell_L
        cell_R = self.geo.cell_R
        internal = (cell_L >= 0) & (cell_R >= 0)
        idx = np.where(internal)[0]
        Q = np.zeros(n_edges)
        if len(idx) == 0:
            return Q

        qL, qR = self.cell_vars[cell_L[idx]], self.cell_vars[cell_R[idx]]
        need_wave = np.abs(qL[:, 0] - qR[:, 0]) >= (qL[:, 0] + qR[:, 0])/2 * 0.01
        idx_wave = idx[need_wave]

        if len(idx_wave) > 0:
            nxw, nyw = normals[idx_wave, 0], normals[idx_wave, 1]
            qLw, qRw = self.cell_vars[cell_L[idx_wave]], self.cell_vars[cell_R[idx_wave]]
            h_L_loc = qLw[:, 0]
            un_L_loc = qLw[:, 1] * nxw + qLw[:, 2] * nyw
            vt_L_loc = -qLw[:, 1] * nyw + qLw[:, 2] * nxw
            h_R_loc = qRw[:, 0]
            un_R_loc = qRw[:, 1] * nxw + qRw[:, 2] * nyw
            vt_R_loc = -qRw[:, 1] * nyw + qRw[:, 2] * nxw
            Q[idx_wave] = compute_shock_wave_speeds_batch(
                h_L_loc, un_L_loc, vt_L_loc, h_R_loc, un_R_loc, vt_R_loc, self.g
            )

        cL_r, cR_r = cell_L[idx], cell_R[idx]
        d0_r = self.geo.initial_distances[idx]
        A0_L = self.initial_areas[cL_r]
        A0_R = self.initial_areas[cR_r]
        Q[idx] += self.alpha * (A0_L / areas[cL_r] - A0_R / areas[cR_r]) / d0_r
        return Q

    def _compute_Q_swept_vectorized(self, old_points, new_points, dt, mid_lengths):
        e = self.geo.edges
        A = old_points[e[:, 0]]
        B = old_points[e[:, 1]]
        Ap = new_points[e[:, 0]]
        Bp = new_points[e[:, 1]]
        swept_area = (
            A[:, 0] * B[:, 1]
            - B[:, 0] * A[:, 1]
            + B[:, 0] * Bp[:, 1]
            - Bp[:, 0] * B[:, 1]
            + Bp[:, 0] * Ap[:, 1]
            - Ap[:, 0] * Bp[:, 1]
            + Ap[:, 0] * A[:, 1]
            - A[:, 0] * Ap[:, 1]
        ) * 0.5
        return swept_area / (dt * mid_lengths)

    def _compute_ale_cfl_dt(self, cfl: float) -> float:
        normals = self.geo.normals
        n_edges = len(self.geo.edges)
        if self._prev_edge_velocity is None or len(self._prev_edge_velocity) != n_edges:
            edge_velocities = np.zeros(n_edges, dtype=float)
        else:
            edge_velocities = self._prev_edge_velocity
        face_speeds = self._compute_face_characteristic_speeds(normals, edge_velocities)
        return cfl * np.min(self._compute_cell_cfl_dt(face_speeds, self._edge_lengths()))

    def _compute_face_characteristic_speeds(self, normals, edge_velocities):
        h_L, un_L, vt_L, h_R, un_R, vt_R, xi = self._prepare_riemann_states(
            normals, edge_velocities
        )
        c_L = np.sqrt(self.g * np.maximum(h_L, 0.0))
        c_R = np.sqrt(self.g * np.maximum(h_R, 0.0))
        return np.maximum.reduce(
            [
                np.abs(un_L - xi - c_L),
                np.abs(un_L - xi + c_L),
                np.abs(un_R - xi - c_R),
                np.abs(un_R - xi + c_R),
            ]
        )

    def _compute_cell_cfl_dt(self, face_speeds, edge_lengths):
        cell_sum = np.zeros(self.n_cells, dtype=float)
        cell_L = self.geo.cell_L
        cell_R = self.geo.cell_R
        mask_L = cell_L >= 0
        mask_R = cell_R >= 0
        weighted = edge_lengths * face_speeds
        np.add.at(cell_sum, cell_L[mask_L], weighted[mask_L])
        np.add.at(cell_sum, cell_R[mask_R], weighted[mask_R])
        return self.areas / cell_sum

    def _compute_node_velocities_vectorized(self, Q_edges, normals):
        node_vel = _compute_node_velocities_pinv(
            Q_edges,
            normals,
            self._node_edge_offsets,
            self._node_edge_indices,
            len(self.geo.points),
        )
        self._apply_velocity_bc(node_vel)
        return node_vel

    def _apply_velocity_bc(self, node_vel):
        if len(self._bc_circle_idx) == 0:
            return
        idx = self._bc_circle_idx
        center = np.asarray(self.circle_center, dtype=float)
        radial = self.geo.points[idx] - center
        radial_norm = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        tangent = np.column_stack([-radial[:, 1], radial[:, 0]]) / radial_norm
        v_tang = np.sum(node_vel[idx] * tangent, axis=1)
        node_vel[idx] = v_tang[:, np.newaxis] * tangent

    def _apply_boundary_constraints(self, points):
        if len(self._bc_circle_idx) == 0:
            return
        idx = self._bc_circle_idx
        center = np.asarray(self.circle_center, dtype=float)
        radial = points[idx] - center
        radial_norm = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        points[idx] = center + radial * (self.circle_radius / radial_norm)

    def _scatter_contributions(self, Q, contribution, areas, divide_by_area=False):
        signs = self.geo.edge_cell_signs
        cells = self.geo.edge_cells
        for j in range(2):
            mask = cells[:, j] >= 0
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            cell_idx = cells[idx, j]
            sign = signs[idx, j].astype(float)
            if divide_by_area:
                scaled = -sign[:, np.newaxis] * contribution[idx] / areas[cell_idx, np.newaxis]
            else:
                scaled = -sign[:, np.newaxis] * contribution[idx]
            for comp in range(3):
                np.add.at(Q[:, comp], cell_idx, scaled[:, comp])

    def _apply_ale_update(self, fluxes, lengths, old_areas, new_areas, dt):
        Q_cons = self._to_conservative()
        AQ = old_areas[:, None] * Q_cons
        contribution = fluxes * lengths[:, np.newaxis] * dt
        self._scatter_contributions(AQ, contribution, None, divide_by_area=False)
        h_new = AQ[:, 0] / new_areas
        if np.any(h_new <= 0.0):
            raise RuntimeError(f"non-positive water depth encountered at t={self.time:.6f}")
        self.cell_vars[:, 0] = h_new
        self.cell_vars[:, 1] = AQ[:, 1] / (new_areas * h_new)
        self.cell_vars[:, 2] = AQ[:, 2] / (new_areas * h_new)

    def _average_normals(self, old_normals, new_normals):
        mid = 0.5 * (old_normals + new_normals)
        return mid / np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-12)

    def _to_conservative(self):
        h, u, v = self.cell_vars.T
        return np.column_stack([h, h * u, h * v])

    def _from_conservative(self, Q):
        h = Q[:, 0]
        self.cell_vars[:, 0] = h
        self.cell_vars[:, 1] = Q[:, 1] / h
        self.cell_vars[:, 2] = Q[:, 2] / h

    @property
    def areas(self):
        pts = self.geo.points
        tri = self.geo.triangles
        v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        return 0.5 * np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )

    def _edge_lengths(self):
        pts = self.geo.points
        e = self.geo.edges
        return np.linalg.norm(pts[e[:, 1]] - pts[e[:, 0]], axis=1)

    def _update_geometry(self):
        pts = self.geo.points
        tri = self.geo.triangles
        self.geo.centers = (pts[tri[:, 0]] + pts[tri[:, 1]] + pts[tri[:, 2]]) / 3.0
        edge_vec = pts[self.geo.edges[:, 1]] - pts[self.geo.edges[:, 0]]
        n = np.column_stack([-edge_vec[:, 1], edge_vec[:, 0]])
        self.geo.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def _compute_edge_cell_signs_and_LR(self):
        n_edges = len(self.geo.edges)
        self.geo.edge_cell_signs = np.zeros((n_edges, 2), dtype=np.int8)
        self.geo.cell_L = np.full(n_edges, -1, dtype=np.intp)
        self.geo.cell_R = np.full(n_edges, -1, dtype=np.intp)

        edge_centers = 0.5 * (self.geo.points[self.geo.edges[:, 0]] + self.geo.points[self.geo.edges[:, 1]])
        for j in range(2):
            cells_j = self.geo.edge_cells[:, j]
            valid = cells_j >= 0
            idx = np.where(valid)[0]
            if len(idx) == 0:
                continue
            c = cells_j[idx]
            vec = edge_centers[idx] - self.geo.centers[c]
            dot = np.sum(vec * self.geo.normals[idx], axis=1)
            pos = dot > 0.0
            self.geo.edge_cell_signs[idx[pos], j] = 1
            self.geo.edge_cell_signs[idx[~pos], j] = -1
            self.geo.cell_L[idx[pos]] = cells_j[idx[pos]]
            self.geo.cell_R[idx[~pos]] = cells_j[idx[~pos]]

    def _compute_initial_distances(self):
        centers = self.geo.centers
        cell_L = self.geo.cell_L
        cell_R = self.geo.cell_R
        internal = (cell_L >= 0) & (cell_R >= 0)
        idx = np.where(internal)[0]
        self.geo.initial_distances = np.zeros(len(self.geo.edges))
        if len(idx) > 0:
            self.geo.initial_distances[idx] = np.linalg.norm(centers[cell_L[idx]] - centers[cell_R[idx]], axis=1)

    @property
    def h(self):
        return None if self.cell_vars is None else self.cell_vars[:, 0]

    @property
    def u(self):
        return None if self.cell_vars is None else self.cell_vars[:, 1]

    @property
    def v(self):
        return None if self.cell_vars is None else self.cell_vars[:, 2]

    @property
    def points(self):
        return self.geo.points

    @property
    def triangles(self):
        return self.geo.triangles

    @property
    def edges(self):
        return self.geo.edges

    @property
    def cell_centers(self):
        return self.geo.centers

    @property
    def total_mass(self):
        return np.sum(self.h * self.areas)

    @property
    def n_cells(self):
        return len(self.geo.triangles)

    @property
    def n_edges(self):
        return len(self.geo.edges)

    @property
    def n_points(self):
        return len(self.geo.points)

    def _generate_mesh(self):
        self._generate_circle_mesh()
        self._build_edges()
        self._precompute_node_edge_adjacency()
        self._build_edge_cells()
        self._classify_boundary()
        self._update_geometry()
        self._compute_edge_cell_signs_and_LR()
        self._compute_initial_distances()
        self._precompute_boundary_arrays()
        self.initial_areas = self.areas.copy()

    def _triangle_flags(self):
        return f"pq{self.min_angle:.1f}a{self.max_area:.10f}e"

    def _generate_circle_mesh(self):
        if tr is None:
            raise ImportError("triangle library is required")

        cx, cy = self.circle_center
        r = self.circle_radius
        edge_len = np.sqrt(4.0 * self.max_area / np.sqrt(3.0))
        n_boundary = max(16, int(np.ceil(2.0 * np.pi * r / edge_len)))

        base_angles = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
        shock_points = self._shock_circle_intersections()
        shock_angles = [np.mod(np.arctan2(y - cy, x - cx), 2.0 * np.pi) for x, y in shock_points]

        angles = base_angles.copy()
        angular_step = 2.0 * np.pi / n_boundary
        for shock_angle in shock_angles:
            diff = np.abs(angles - shock_angle)
            diff = np.minimum(diff, 2.0 * np.pi - diff)
            nearest = int(np.argmin(diff))
            if diff[nearest] < 0.5 * angular_step:
                angles[nearest] = shock_angle
            else:
                angles = np.append(angles, shock_angle)

        angles = np.sort(np.mod(angles, 2.0 * np.pi))
        if len(angles) > 1:
            keep = np.concatenate([[True], np.diff(angles) > 1e-10])
            angles = angles[keep]

        points = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
        n_points = len(points)
        segments = [[i, (i + 1) % n_points] for i in range(n_points)]
        markers = [1] * n_points

        if len(shock_angles) == 2:
            idx1 = self._nearest_angle_index(angles, shock_angles[0])
            idx2 = self._nearest_angle_index(angles, shock_angles[1])
            if idx1 == idx2:
                raise ValueError("shock line intersections collapsed onto one boundary node")
            segments.append([idx1, idx2])
            markers.append(2)

        mesh = tr.triangulate(
            {"vertices": np.asarray(points, dtype=float), "segments": np.asarray(segments), "segment_markers": np.asarray(markers)},
            self._triangle_flags(),
        )
        self.geo = Geometry(points=mesh["vertices"], triangles=mesh["triangles"])
        self._mesh_output = mesh

    def _shock_circle_intersections(self):
        p0 = np.asarray(self.line_point, dtype=float)
        tangent = self.unit_tangent

        center = np.asarray(self.circle_center, dtype=float)
        offset = p0 - center
        b = 2.0 * np.dot(tangent, offset)
        c = np.dot(offset, offset) - self.circle_radius ** 2
        discriminant = b ** 2 - 4.0 * c
        if discriminant < 0.0:
            raise ValueError("shock line does not intersect the circular domain")

        root = np.sqrt(max(discriminant, 0.0))
        s1 = (-b - root) / 2.0
        s2 = (-b + root) / 2.0
        intersections = [p0 + s1 * tangent, p0 + s2 * tangent]
        return [point.tolist() for point in intersections]

    def _nearest_angle_index(self, angles, target):
        diff = np.abs(angles - target)
        diff = np.minimum(diff, 2.0 * np.pi - diff)
        return int(np.argmin(diff))

    def _build_edges(self):
        tri = self.geo.triangles
        e0 = np.sort(tri[:, [0, 1]], axis=1)
        e1 = np.sort(tri[:, [1, 2]], axis=1)
        e2 = np.sort(tri[:, [2, 0]], axis=1)
        self.geo.edges = np.unique(np.vstack([e0, e1, e2]), axis=0)

    def _precompute_node_edge_adjacency(self):
        self._node_edge_offsets, self._node_edge_indices = _build_node_edge_adjacency(
            self.geo.edges,
            len(self.geo.points),
        )

    def _build_edge_cells(self):
        edge_to_cells = {}
        for cell_idx, tri in enumerate(self.geo.triangles):
            for i in range(3):
                key = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edge_to_cells.setdefault(key, []).append(cell_idx)

        self.geo.edge_cells = np.full((len(self.geo.edges), 2), -1, dtype=np.intp)
        for i, (p1, p2) in enumerate(self.geo.edges):
            cells = edge_to_cells.get(tuple(sorted([p1, p2])), [])
            if len(cells) >= 1:
                self.geo.edge_cells[i, 0] = cells[0]
            if len(cells) >= 2:
                self.geo.edge_cells[i, 1] = cells[1]

    def _classify_boundary(self):
        edges = self._mesh_output.get("edges", [])
        markers = self._mesh_output.get("edge_markers", [])
        if len(markers) and getattr(markers, "ndim", 1) == 2:
            markers = markers.flatten()

        self.boundary_types.clear()
        segment_pts: dict[int, set[int]] = {}
        for edge, marker in zip(edges, markers):
            segment_pts.setdefault(int(marker), set()).update(edge)

        outer = segment_pts.get(1, set())
        diagonal = segment_pts.get(2, set())
        for idx in outer:
            self.boundary_types[idx] = BoundaryType.CIRCLE
        for idx in diagonal:
            if idx not in outer:
                self.boundary_types[idx] = BoundaryType.DIAGONAL

    def _precompute_boundary_arrays(self):
        circle = [idx for idx, boundary_type in self.boundary_types.items() if boundary_type == BoundaryType.CIRCLE]
        self._bc_circle_idx = np.array(sorted(circle), dtype=np.intp)
