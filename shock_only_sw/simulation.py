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

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    mesh: object
    times: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    h_history: list[np.ndarray] = field(default_factory=list)
    u_history: list[np.ndarray] = field(default_factory=list)
    v_history: list[np.ndarray] = field(default_factory=list)
    points_history: list[np.ndarray] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(
        default_factory=lambda: {"l1_h": [], "l1_u": [], "l1_v": [], "l1_total": []}
    )
    summary: dict[str, float] = field(default_factory=dict)


def _evaluate_exact(exact_solution, x, y, t):
    if hasattr(exact_solution, "evaluate"):
        return exact_solution.evaluate(x, y, t)
    return exact_solution(x, y, t)


def compute_l1_errors(mesh):
    if mesh.exact_solution is None:
        raise ValueError("mesh.exact_solution is required to compute L1 errors")
    x = mesh.cell_centers[:, 0]
    y = mesh.cell_centers[:, 1]
    h_ex, u_ex, v_ex = _evaluate_exact(mesh.exact_solution, x, y, mesh.time)
    areas = mesh.areas
    area_sum = np.sum(areas)

    l1_h = np.sum(areas * np.abs(mesh.h - h_ex)) / area_sum
    l1_u = np.sum(areas * np.abs(mesh.u - u_ex)) / area_sum
    l1_v = np.sum(areas * np.abs(mesh.v - v_ex)) / area_sum
    l1_total = np.sum(
        areas
        * (
            np.abs(mesh.h - h_ex)
            + np.abs(mesh.u - u_ex)
            + np.abs(mesh.v - v_ex)
        )
    ) / area_sum
    return {
        "l1_h": float(l1_h),
        "l1_u": float(l1_u),
        "l1_v": float(l1_v),
        "l1_total": float(l1_total),
    }


def _append_frame(result: SimulationResult, mesh, step_count: int):
    result.times.append(float(mesh.time))
    result.steps.append(int(step_count))
    result.h_history.append(mesh.h.copy())
    result.u_history.append(mesh.u.copy())
    result.v_history.append(mesh.v.copy())
    result.points_history.append(mesh.points.copy())
    metrics = compute_l1_errors(mesh)
    for key, value in metrics.items():
        result.metrics[key].append(value)


def _integrate_metric(times, values):
    if len(times) <= 1:
        return float(values[0]) if values else 0.0
    total_time = times[-1] - times[0]
    if total_time <= 0.0:
        return float(values[-1])
    return float(np.trapezoid(values, times) / total_time)


def run_simulation(
    mesh,
    t_end: float,
    dt_fixed: float | None = None,
    cfl: float = 0.5,
    save_interval: float = 0.01,
    moving_mesh: bool = True,
):
    result = SimulationResult(mesh=mesh)

    step_count = 0
    next_save_time = mesh.time + save_interval
    _append_frame(result, mesh, step_count)

    while mesh.time < t_end - 1e-14:
        dt = dt_fixed if dt_fixed is not None else mesh.compute_cfl_dt(cfl=cfl, moving_mesh=moving_mesh)
        dt = min(dt, t_end - mesh.time)
        mesh.evolve(dt, moving_mesh=moving_mesh)
        step_count += 1

        should_save = mesh.time >= next_save_time - 1e-14 or mesh.time >= t_end - 1e-14
        if should_save:
            _append_frame(result, mesh, step_count)
            while next_save_time <= mesh.time + 1e-14:
                next_save_time += save_interval

    times = result.times
    for name in ("h", "u", "v"):
        key = f"l1_{name}"
        values = result.metrics[key]
        result.summary[f"final_l1_{name}"] = float(values[-1])
        result.summary[f"max_l1_{name}"] = float(np.max(values))
        result.summary[f"integrated_l1_{name}"] = _integrate_metric(times, values)
    result.summary["final_l1_total"] = float(result.metrics["l1_total"][-1])
    result.summary["max_l1_total"] = float(np.max(result.metrics["l1_total"]))
    result.summary["integrated_l1_total"] = _integrate_metric(times, result.metrics["l1_total"])
    result.summary["t_final"] = float(result.times[-1])
    result.summary["n_steps"] = int(step_count)
    result.summary["n_frames"] = int(len(result.times))
    return result
