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


@njit(cache=True)
def _phi(c, c_i):
    sigma = c / c_i
    if sigma >= 1.0:
        return (c - c_i) * (sigma + 1.0) * np.sqrt(0.5 * (1.0 + sigma ** (-2)))
    return 2.0 * (c - c_i)


@njit(cache=True)
def _dphi_dc(c, c_i):
    sigma = c / c_i
    if sigma >= 1.0:
        sigma2 = sigma * sigma
        numerator = 2.0 * sigma2 + 1.0 + 1.0 / sigma2
        denominator = np.sqrt(2.0 * (1.0 + sigma2))
        return numerator / denominator
    return 2.0


@njit(cache=True)
def solve_star_state(h_L, u_L, h_R, u_R, g, max_iter=20, tol=1e-10):
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)
    c_s = 0.25 * (u_L - u_R) + 0.5 * (c_L + c_R)

    for _ in range(max_iter):
        pL = _phi(c_s, c_L)
        pR = _phi(c_s, c_R)
        F = pL + pR + (u_R - u_L)

        dpL = _dphi_dc(c_s, c_L)
        dpR = _dphi_dc(c_s, c_R)
        dF = dpL + dpR

        dc = -F / dF
        c_new = c_s + dc
        if c_new <= 0.0:
            c_new = 0.5 * c_s

        if abs(dc) < tol * (1.0 + c_new):
            c_s = c_new
            break
        c_s = c_new

    h_s = c_s * c_s / g
    pL = _phi(c_s, c_L)
    pR = _phi(c_s, c_R)
    u_s = 0.5 * (u_L + u_R) + 0.5 * (pR - pL)
    return h_s, u_s


@njit(cache=True)
def _sample_riemann_fan(h_L, u_L, h_R, u_R, h_star, u_star, xi, g):
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)
    c_star = np.sqrt(g * h_star)

    if h_star > h_L:
        s_head_L = u_L - c_star * np.sqrt(0.5 * (1.0 + h_star / h_L))
        s_tail_L = s_head_L
    else:
        s_head_L = u_L - c_L
        s_tail_L = 1.5 * u_star - c_L - 0.5 * u_L

    if h_star > h_R:
        s_head_R = u_R + c_star * np.sqrt(0.5 * (1.0 + h_star / h_R))
        s_tail_R = s_head_R
    else:
        s_head_R = u_R + c_R
        s_tail_R = 1.5 * u_star + c_R - 0.5 * u_R

    if xi <= s_head_L:
        return h_L, u_L
    if xi <= s_tail_L:
        c = (2.0 * c_L + u_L - xi) / 3.0
        return c * c / g, xi + c
    if xi <= s_tail_R:
        return h_star, u_star
    if xi <= s_head_R:
        c = (2.0 * c_R - u_R + xi) / 3.0
        return c * c / g, xi - c
    return h_R, u_R


@njit(cache=True)
def _solve_single_riemann(h_L, un_L, vt_L, h_R, un_R, vt_R, xi, g):
    h_star, u_star = solve_star_state(h_L, un_L, h_R, un_R, g)
    h_out, un_out = _sample_riemann_fan(h_L, un_L, h_R, un_R, h_star, u_star, xi, g)
    if u_star > xi:
        vt_out = vt_L
    else:
        vt_out = vt_R
    return h_out, un_out, vt_out


@njit(cache=True)
def _shock_speed_single(h_L, un_L, vt_L, h_R, un_R, vt_R, g):
    h_s, u_s = solve_star_state(h_L, un_L, h_R, un_R, g)
    c_s = np.sqrt(g * h_s)

    E_L = h_L * (un_L ** 2 + vt_L ** 2) / 2.0 + g * h_L ** 2 / 2.0
    E_R = h_R * (un_R ** 2 + vt_R ** 2) / 2.0 + g * h_R ** 2 / 2.0
    E_star_L = h_s * (u_s ** 2 + vt_L ** 2) / 2.0 + g * h_s ** 2 / 2.0
    E_star_R = h_s * (u_s ** 2 + vt_R ** 2) / 2.0 + g * h_s ** 2 / 2.0

    left_is_shock = h_s > h_L
    right_is_shock = h_s > h_R

    left_speed = 0.0
    left_amp = -1.0
    if left_is_shock:
        left_speed = un_L - c_s * np.sqrt(0.5 * (1.0 + h_s / h_L))
        left_amp = abs(E_star_L - E_L)

    right_speed = 0.0
    right_amp = -1.0
    if right_is_shock:
        right_speed = un_R + c_s * np.sqrt(0.5 * (1.0 + h_s / h_R))
        right_amp = abs(E_R - E_star_R)

    if left_amp < 0.0 and right_amp < 0.0:
        return 0.0
    if left_amp >= right_amp:
        return left_speed
    return right_speed


@njit(cache=True)
def _riemann_batch_loop(h_L, un_L, vt_L, h_R, un_R, vt_R, xi, g):
    n = len(h_L)
    h_out = np.empty(n)
    un_out = np.empty(n)
    vt_out = np.empty(n)
    for i in prange(n):
        h_out[i], un_out[i], vt_out[i] = _solve_single_riemann(
            h_L[i], un_L[i], vt_L[i], h_R[i], un_R[i], vt_R[i], xi[i], g
        )
    return h_out, un_out, vt_out


@njit(cache=True)
def _shock_speeds_batch_loop(h_L, un_L, vt_L, h_R, un_R, vt_R, g):
    n = len(h_L)
    speeds = np.empty(n)
    for i in prange(n):
        speeds[i] = _shock_speed_single(
            h_L[i], un_L[i], vt_L[i], h_R[i], un_R[i], vt_R[i], g
        )
    return speeds


def riemann_solve_batch(h_L, un_L, vt_L, h_R, un_R, vt_R, xi, g):
    return _riemann_batch_loop(h_L, un_L, vt_L, h_R, un_R, vt_R, xi, g)


def compute_shock_wave_speeds_batch(h_L, un_L, vt_L, h_R, un_R, vt_R, g):
    return _shock_speeds_batch_loop(h_L, un_L, vt_L, h_R, un_R, vt_R, g)
