#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import mujoco
import mujoco.viewer
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Math helpers
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler_xyz(quat_wxyz: np.ndarray) -> Tuple[float, float, float]:
    """quat_wxyz: [w,x,y,z] -> (roll,pitch,yaw) XYZ Tait-Bryan"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = _clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# -----------------------------
# Discrete-time LQR
# -----------------------------
def solve_DARE(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
               maxiter: int = 1000, eps: float = 1e-12) -> np.ndarray:
    P = Q.copy()
    for _ in range(maxiter):
        BtPB = B.T @ P @ B
        inv_term = np.linalg.inv(R + BtPB)
        Pn = A.T @ P @ A - A.T @ P @ B @ inv_term @ B.T @ P @ A + Q
        if np.max(np.abs(Pn - P)) < eps:
            P = Pn
            break
        P = Pn
    return P


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_DARE(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K


# -----------------------------
# Approximate linear model for wheel-balancing
# state: [x, xdot, pitch, pitch_rate, yaw, yaw_rate]
# control: [uL, uR]
# -----------------------------
@dataclass
class IPParams:
    m: float = 0.28
    M: float = 4.1
    l: float = 0.30
    r: float = 0.022
    d: float = 0.1524
    g: float = 9.8
    dt: float = 0.01  # control update dt (not sim dt)


def get_model_matrix(p: IPParams) -> Tuple[np.ndarray, np.ndarray]:
    m, M, l, r, d, g, dt = p.m, p.M, p.l, p.r, p.d, p.g, p.dt

    I = 0.5 * M * r * r
    Jp = (1.0/3.0) * M * (l**2)
    Qeq = M * Jp + (Jp + M*l*l) * (2*m + 2*I/(r**2))
    Jdelta = (1.0/12.0) * M * (d**2)

    A23 = -(M**2 * l**2 * g) / Qeq
    A43 = (M * l * g * (M + 2*m + 2*I/(r**2))) / Qeq

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, A23, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, A43, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)
    A = np.eye(6) + dt * A

    B21 = (Jp + M*l*l + M*l*r) / (Qeq * r)
    B41 = -((M*l)/r + M + 2*m + 2*I/(r**2)) / Qeq
    B42 = B41

    Bd = 1.0 / (r * (m*d + (I*d)/(r*r) + (2*Jdelta)/d))
    B61 = Bd
    B62 = -Bd

    B = np.array([
        [0,   0],
        [B21, B21],
        [0,   0],
        [B41, B42],
        [0,   0],
        [B61, B62],
    ], dtype=float)
    B = B * dt
    return A, B


# -----------------------------
# IO config
# -----------------------------
@dataclass
class IOConfig:
    # actuators
    L_thigh_act: str = "L_thigh"
    L_calf_act: str = "L_calf"
    L_wheel_act: str = "L_wheel"
    R_thigh_act: str = "R_thigh"
    R_calf_act: str = "R_calf"
    R_wheel_act: str = "R_wheel"

    # sensors
    imu_quat: str = "imu_quat"
    imu_gyro: str = "imu_gyro"
    frame_ang_vel: str = "frame_ang_vel"

    L_wheel_pos: str = "L_wheel_pos"
    R_wheel_pos: str = "R_wheel_pos"
    L_wheel_vel: str = "L_wheel_vel"
    R_wheel_vel: str = "R_wheel_vel"

    # axis selection
    pitch_rate_axis: int = 0  # imu_gyro axis for pitch rate (0/1/2)
    yaw_rate_axis_world: int = 2  # frame_ang_vel axis for yaw rate

    # signs (normally keep 1; if still反，先改 pitch_sign)
    pitch_sign: float = 1.0
    yaw_sign: float = 1.0
    yaw_rate_sign: float = 1.0


class BalanceLQR:
    def __init__(self, model: mujoco.MjModel, io: IOConfig, ip: IPParams,
                 Q6: np.ndarray, R2: np.ndarray,
                 u_limit: float = 15.0,
                 ix_lim: float = 0.10,
                 ki_freeze_pitch: float = 0.25,
                 ki_freeze_u_frac: float = 0.7,
                 theta_ref_lim: float = 0.35) -> None:
        self.model = model
        self.io = io
        self.ip = ip
        self.u_limit = float(u_limit)
        # ---- target state (default all zeros) ----
        # state order: [x, xdot, pitch, pitch_rate, yaw, yaw_rate, ix]
        self.X_ref = np.zeros((7, 1), dtype=float)


        # integral (for x)
        self.ix = 0.0
        self.ix_lim = float(ix_lim)
        self.ki_freeze_pitch = float(ki_freeze_pitch)
        self.ki_freeze_u_frac = float(ki_freeze_u_frac)

        # optional: if R wheel direction opposite, set -1
        self.wheel_sign_R = 1.0

        # wheel reference offsets
        self.qL0 = 0.0
        self.qR0 = 0.0

        # attitude zero offsets
        self.pitch0 = None
        self.yaw0 = None

        self.pitch_err_last = 0.0
        self.pitch_rate_f = 0.0

        self.theta_ref = 0.0
        self.theta_ref_sat = False


        # use real control dt for integrator
        self.t_u_last = None

        # IDs
        self.act_Lw  = model.actuator(io.L_wheel_act).id
        self.act_Rw  = model.actuator(io.R_wheel_act).id
        self.act_Lth = model.actuator(io.L_thigh_act).id
        self.act_Lca = model.actuator(io.L_calf_act).id
        self.act_Rth = model.actuator(io.R_thigh_act).id
        self.act_Rca = model.actuator(io.R_calf_act).id

        # build LQI (augment ix)
        A6, B2 = get_model_matrix(ip)
        A = np.zeros((7, 7), dtype=float)
        B = np.zeros((7, 2), dtype=float)
        A[:6, :6] = A6
        B[:6, :] = B2
        A[6, 6] = 1.0
        A[6, 0] = ip.dt  # ix_{k+1} = ix_k + dt * x

        Q7 = np.zeros((7, 7), dtype=float)
        Q7[:6, :6] = Q6
        Q7[6, 6] = 50.0

        self.K = dlqr(A, B, Q7, R2)  # (2,7)

        # ---- LQR -> PID-like gains (sum/diff decouple) ----
        self.Ksum  = 0.5 * (self.K[0, :] + self.K[1, :])  # u_sum ≈ -Ksum @ X
        self.Kdiff = 0.5 * (self.K[0, :] - self.K[1, :])  # u_diff ≈ -Kdiff @ X

        # inner-loop (attitude) gains = LQR equivalent
        self.k_theta  = float(self.Ksum[2])
        self.k_dtheta = float(self.Ksum[3])

        # outer-loop (position -> theta_ref) gains MUST be ratios
        den = self.k_theta if abs(self.k_theta) > 1e-6 else 1.0
        self.k_x  = float(self.Ksum[0] / den)
        self.k_dx = float(self.Ksum[1] / den)
        self.k_i  = float(self.Ksum[6] / den)
        self.k_psi    = float(self.Kdiff[4])
        self.k_dpsi   = float(self.Kdiff[5])

        self.theta_ref_lim = float(theta_ref_lim)

        self.last_u = np.zeros(2, dtype=float)

    def reset(self, data: mujoco.MjData) -> None:
        self.ix = float(self.X_ref[6, 0])  # default 0
        self.t_u_last = None
        self.pitch_err_last = 0.0
        self.pitch_rate_f = 0.0


        self.pitch0 = None
        self.yaw0 = None

        self.qL0 = float(data.sensor(self.io.L_wheel_pos).data[0])
        self.qR0 = float(data.sensor(self.io.R_wheel_pos).data[0])

        self.last_u[:] = 0.0

    def auto_pitch_sign(self, data: mujoco.MjData, duration_s: float = 0.6) -> None:
        """Align pitch_sign using correlation between gyro axis and finite-diff pitch (needs small motion)."""
        t0 = float(data.time)
        samples_fd = []
        samples_g = []
        pitch_prev = None
        t_prev = None
        while float(data.time) - t0 < duration_s:
            mujoco.mj_step(self.model, data)
            t = float(data.time)
            quat = data.sensor(self.io.imu_quat).data
            _, pitch, _ = quat_to_euler_xyz(quat)
            gyro = data.sensor(self.io.imu_gyro).data
            g = float(gyro[int(self.io.pitch_rate_axis)])
            if pitch_prev is not None and t_prev is not None:
                dt = max(1e-6, t - t_prev)
                pitch_fd = (pitch - pitch_prev) / dt
                samples_fd.append(pitch_fd)
                samples_g.append(g)
            pitch_prev, t_prev = pitch, t
        if len(samples_fd) < 20:
            return
        fd = np.array(samples_fd)
        gg = np.array(samples_g)
        if np.std(fd) < 1e-4 or np.std(gg) < 1e-4:
            return
        corr = float(np.corrcoef(fd, gg)[0, 1])
        if corr < 0.0:
            self.io.pitch_sign *= -1.0

    def read_state(self, data: mujoco.MjData) -> Tuple[np.ndarray, Dict[str, float]]:
        # --- attitude ---
        quat = data.sensor(self.io.imu_quat).data
        _, pitch, yaw = quat_to_euler_xyz(quat)
        pitch = float(pitch) * float(self.io.pitch_sign)
        yaw = float(yaw) * float(self.io.yaw_sign)

        # zero-offset calibration (treat current as 0)
        if self.pitch0 is None:
            self.pitch0 = pitch
        if self.yaw0 is None:
            self.yaw0 = yaw

        pitch_err = pitch - self.pitch0
        yaw_err = wrap_pi(yaw - self.yaw0)

        # --- dt_u MUST be computed before using it ---
        t = float(data.time)  # MuJoCo simulation time accumulates with mj_step :contentReference[oaicite:1]{index=1}
        if self.t_u_last is None:
            dt_u = self.ip.dt
        else:
            dt_u = max(1e-6, t - self.t_u_last)
        self.t_u_last = t

        # --- pitch_rate from finite difference (consistent with pitch_err) ---
        if getattr(self, "pitch_err_last", None) is None:
            self.pitch_err_last = pitch_err
            self.pitch_rate_f = 0.0

        pitch_rate_fd = (pitch_err - self.pitch_err_last) / dt_u
        self.pitch_err_last = pitch_err

        alpha = 0.85
        self.pitch_rate_f = alpha * self.pitch_rate_f + (1.0 - alpha) * pitch_rate_fd
        pitch_rate = self.pitch_rate_f

        # --- yaw_rate ---
        w_world = data.sensor(self.io.frame_ang_vel).data
        yaw_rate = float(w_world[int(self.io.yaw_rate_axis_world)]) * float(self.io.yaw_rate_sign)

        # --- wheels -> x, xdot ---
        qL = float(data.sensor(self.io.L_wheel_pos).data[0]) - self.qL0
        qR = float(data.sensor(self.io.R_wheel_pos).data[0]) - self.qR0
        qdL = float(data.sensor(self.io.L_wheel_vel).data[0])
        qdR = float(data.sensor(self.io.R_wheel_vel).data[0])
        x = self.ip.r * 0.5 * (qL + self.wheel_sign_R * qR)
        xdot = self.ip.r * 0.5 * (qdL + self.wheel_sign_R * qdR)

        # anti-windup integrate x_err
        u_mag = float(max(abs(self.last_u[0]), abs(self.last_u[1])))
        if (abs(pitch_err) < self.ki_freeze_pitch) and (u_mag < self.ki_freeze_u_frac * self.u_limit):
            x_err = x - float(self.X_ref[0, 0])
            self.ix += dt_u * x_err
            self.ix = float(np.clip(self.ix, -self.ix_lim, self.ix_lim))

        X = np.array([[x],[xdot],[pitch_err],[pitch_rate],[yaw_err],[yaw_rate],[self.ix]], dtype=float)
        E = X - self.X_ref

        dbg = {"x": x, 
                "xdot": xdot,
                "pitch": pitch_err,
                "pitch_rate": pitch_rate,
                "yaw": yaw_err, 
                "yaw_rate": yaw_rate, 
                "ix": self.ix, 
                "dt_u": dt_u}
        return E, dbg


    def compute_u_pidlike(self, X: np.ndarray, u_scale: float = 1.0) -> np.ndarray:
        x, xdot, theta, theta_dot, yaw, yaw_dot, ix = X.reshape(-1)

        if abs(theta) > 0.6:
            theta_ref_unsat = 0.0
            theta_ref = 0.0
        else:
            theta_ref_unsat = -(self.k_x * x + self.k_dx * xdot + self.k_i * ix)
            theta_ref = float(np.clip(theta_ref_unsat, -self.theta_ref_lim, self.theta_ref_lim))

        # --- store for logging ---
        self.theta_ref = float(theta_ref)
        self.theta_ref_sat = (theta_ref != theta_ref_unsat)

        u_sum  = -(self.k_theta * (theta - theta_ref) + self.k_dtheta * theta_dot)
        u_diff = -(self.k_psi * yaw + self.k_dpsi * yaw_dot)

        u = u_scale * np.array([u_sum + u_diff, u_sum - u_diff], dtype=float)
        u = np.clip(u, -self.u_limit, self.u_limit)
        self.last_u[:] = u
        return u


    def apply_wheel_u(self, data: mujoco.MjData, u: np.ndarray) -> None:
        data.ctrl[self.act_Lw] = float(u[0])
        data.ctrl[self.act_Rw] = float(u[1])

    def apply_leg_pd(self, data: mujoco.MjData,
                     leg_targets: Tuple[float, float, float, float],
                     kps: np.ndarray, kds: np.ndarray) -> None:
        qLth0, qLca0, qRth0, qRca0 = leg_targets
        qLth = float(data.sensor("L_thigh_pos").data[0]); dLth = float(data.sensor("L_thigh_vel").data[0])
        qLca = float(data.sensor("L_calf_pos").data[0]);  dLca = float(data.sensor("L_calf_vel").data[0])
        qRth = float(data.sensor("R_thigh_pos").data[0]); dRth = float(data.sensor("R_thigh_vel").data[0])
        qRca = float(data.sensor("R_calf_pos").data[0]);  dRca = float(data.sensor("R_calf_vel").data[0])

        data.ctrl[self.act_Lth] = float(kps[0] * (qLth0 - qLth) - kds[0] * dLth)
        data.ctrl[self.act_Lca] = float(kps[1] * (qLca0 - qLca) - kds[1] * dLca)
        data.ctrl[self.act_Rth] = float(kps[3] * (qRth0 - qRth) - kds[3] * dRth)
        data.ctrl[self.act_Rca] = float(kps[4] * (qRca0 - qRca) - kds[4] * dRca)


def parse_diag_list(s: str, n_toggle: int) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != n_toggle:
        raise ValueError(f"Expected {n_toggle} comma-separated values, got {len(parts)}: {s}")
    vals = [float(x) for x in parts]
    return np.diag(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default=None, help="scene.xml path")
    ap.add_argument("--control_dt", type=float, default=0.005)
    ap.add_argument("--u_limit", type=float, default=15.0)
    ap.add_argument("--plot", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--log_csv", type=str, default="balance_log.csv")

    # model params for A,B
    ap.add_argument("--m", type=float, default=0.28)
    ap.add_argument("--M", type=float, default=7.002)
    ap.add_argument("--l", type=float, default=0.37)
    ap.add_argument("--r", type=float, default=0.022)
    ap.add_argument("--d", type=float, default=0.329)

    # LQR weights (first 6 states)
    ap.add_argument("--Q6", type=str, default="20000,150,1500,1000,0,0",
                    help="diag Q for [x,xdot,pitch,pitch_rate,yaw,yaw_rate]")
    ap.add_argument("--R2", type=str, default="10,10",
                    help="diag R for [uL,uR]")

    # integrator
    ap.add_argument("--ix_lim", type=float, default=0.90)
    ap.add_argument("--ki_freeze_pitch", type=float, default=0.25)
    ap.add_argument("--ki_freeze_u_frac", type=float, default=0.7)
    ap.add_argument("--theta_ref_lim", type=float, default=0.35)

    # IMU mapping
    ap.add_argument("--pitch_rate_axis", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--pitch_sign", type=float, default=1.0)
    ap.add_argument("--auto_pitch_sign", action="store_true")

    # leg PD
    ap.add_argument("--legs", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--kp_leg", type=float, default=100.0)
    ap.add_argument("--kd_leg", type=float, default=5.0)

    # control shaping
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--ramp_time", type=float, default=0.3)
    ap.add_argument("--slew_rate", type=float, default=400.0, help="Nm/s")

    args = ap.parse_args()

    default_xml = "/home/elsie/mujoco_course/crazydog_urdf/urdf/scene.xml"
    if not args.xml:
        args.xml = default_xml
    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"XML not found: {args.xml}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    sim_dt = float(model.opt.timestep)
    ip = IPParams(m=args.m, M=args.M, l=args.l, r=args.r, d=args.d, dt=args.control_dt)
    io = IOConfig(pitch_rate_axis=args.pitch_rate_axis, pitch_sign=args.pitch_sign)

    Q6 = parse_diag_list(args.Q6, 6)
    R2 = parse_diag_list(args.R2, 2)

    ctrl = BalanceLQR(model, io, ip, Q6=Q6, R2=R2, u_limit=args.u_limit,
                      ix_lim=args.ix_lim,
                      ki_freeze_pitch=args.ki_freeze_pitch,
                      ki_freeze_u_frac=args.ki_freeze_u_frac,
                      theta_ref_lim=args.theta_ref_lim)

    mujoco.mj_resetData(model, data)
    ctrl.reset(data)

    # optional auto pitch_sign alignment (needs tiny motion)
    if args.auto_pitch_sign:
        ctrl.auto_pitch_sign(data, duration_s=0.6)
        mujoco.mj_resetData(model, data)
        ctrl.reset(data)

    # leg targets (your bent pose)
    q_L_thigh0 = 1.27 
    q_L_calf0  = -2.127 
    q_R_thigh0 = 1.27 
    q_R_calf0  = -2.127 
    leg_targets = (q_L_thigh0, q_L_calf0, q_R_thigh0, q_R_calf0)

    kps = np.array([args.kp_leg, args.kp_leg, 0.0, args.kp_leg, args.kp_leg, 0.0], dtype=float)
    kds = np.array([args.kd_leg, args.kd_leg, 0.0, args.kd_leg, args.kd_leg, 0.0], dtype=float)

    print("=" * 60)
    print("Balance LQI -> PID-like (sum/diff) Controller")
    print("=" * 60)
    print(f"sim_dt      = {sim_dt:.4f} s")
    print(f"control_dt  = {args.control_dt:.4f} s")
    print(f"wheel ctrlrange L: {model.actuator(io.L_wheel_act).ctrlrange}")
    print(f"wheel ctrlrange R: {model.actuator(io.R_wheel_act).ctrlrange}")
    print(f"pitch_sign  = {io.pitch_sign}")
    print(f"pitch_rate_axis = {io.pitch_rate_axis}")
    print(f"Q6 diag = {np.diag(Q6)}")
    print(f"R2 diag = {np.diag(R2)}")
    print(f"K shape = {ctrl.K.shape}")
    print(f"(from K) k_x={ctrl.k_x:.2f}, k_dx={ctrl.k_dx:.2f}, k_theta={ctrl.k_theta:.2f}, k_dtheta={ctrl.k_dtheta:.2f}, k_i={ctrl.k_i:.2f}")
    print(f"(from K) k_psi={ctrl.k_psi:.2f}, k_dpsi={ctrl.k_dpsi:.2f}")
    print(f"ix_lim={args.ix_lim}, freeze_pitch={args.ki_freeze_pitch}, freeze_u_frac={args.ki_freeze_u_frac}, theta_ref_lim={args.theta_ref_lim}")
    print(f"u_scale={args.u_scale}, ramp_time={args.ramp_time}, slew_rate={args.slew_rate} Nm/s")
    print(f"Logging to: {os.path.abspath(args.log_csv)}")
    print("=" * 60)

    last_u_time = -1e9
    u_hold = np.zeros(2, dtype=float)

    log_rows: List[Dict[str, float]] = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_wall0 = time.time()

            if args.legs == "on":
                ctrl.apply_leg_pd(data, leg_targets, kps=kps, kds=kds)

            # update at control_dt
            if float(data.time) - last_u_time >= args.control_dt - 1e-12:
                X, dbg = ctrl.read_state(data)
                u = ctrl.compute_u_pidlike(X, u_scale=args.u_scale)

                # soft ramp
                ramp = float(np.clip(float(data.time) / max(1e-6, args.ramp_time), 0.0, 1.0))
                u = ramp * u

                # slew-rate limit
                du_max = float(args.slew_rate) * args.control_dt
                u_hold[:] = np.clip(u, u_hold - du_max, u_hold + du_max)

                last_u_time = float(data.time)

                # log
                log_rows.append({
                    "time_s": float(data.time),
                    "x_m": float(dbg["x"]),
                    "xdot_m_s": float(dbg["xdot"]),
                    "pitch_rad": float(dbg["pitch"]),
                    "pitch_rate_rad_s": float(dbg["pitch_rate"]),
                    "yaw_rad": float(dbg["yaw"]),
                    "yaw_rate_rad_s": float(dbg["yaw_rate"]),
                    "ix": float(dbg["ix"]),
                    "theta_ref": float(ctrl.theta_ref),
                    "theta_ref_sat": int(bool(ctrl.theta_ref_sat)),
                    "dt_u": float(dbg["dt_u"]),
                    "torque_L_Nm": float(u_hold[0]),
                    "torque_R_Nm": float(u_hold[1]),
                })


            ctrl.apply_wheel_u(data, u_hold)
            mujoco.mj_step(model, data)
            viewer.sync()

            # real-time pacing
            dt_sleep = sim_dt - (time.time() - t_wall0)
            if dt_sleep > 0:
                time.sleep(dt_sleep)

    if len(log_rows) == 0:
        print("No log rows collected.")
        return

    df = pd.DataFrame(log_rows)
    df.to_csv(args.log_csv, index=False)
    print(f"\nSaved log: {os.path.abspath(args.log_csv)}")

    if args.plot == "on":
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        axes[0].plot(df["time_s"], df["x_m"])
        axes[0].axhline(0.0, linestyle="--")
        axes[0].set_title("Position x vs Time")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("x [m]")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df["time_s"], df["xdot_m_s"])
        axes[1].axhline(0.0, linestyle="--")
        axes[1].set_title("Velocity xdot vs Time")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("xdot [m/s]")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(df["time_s"], df["pitch_rad"])
        axes[2].axhline(0.0, linestyle="--")
        axes[2].set_title("Pitch vs Time")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("pitch [rad]")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(df["time_s"], df["torque_L_Nm"], label="L")
        axes[3].plot(df["time_s"], df["torque_R_Nm"], label="R")
        axes[3].axhline(0.0, linestyle="--")
        axes[3].set_title("Wheel torque vs Time")
        axes[3].set_xlabel("Time [s]")
        axes[3].set_ylabel("Nm")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("\n=== Stats ===")
    print(f"x range: [{df['x_m'].min():+.3f}, {df['x_m'].max():+.3f}] m")
    print(f"xdot mean: {df['xdot_m_s'].mean():+.3f} m/s")
    print(f"pitch range: [{df['pitch_rad'].min():+.3f}, {df['pitch_rad'].max():+.3f}] rad")
    print(f"mean |torque|: {(df['torque_L_Nm'].abs().mean()):.2f} Nm")
    print(f"max  |torque|: {(df['torque_L_Nm'].abs().max()):.2f} Nm")
    print(f"ix range: [{df['ix'].min():+.6f}, {df['ix'].max():+.6f}]")
    print(f"theta_ref range: [{df['theta_ref'].min():+.6f}, {df['theta_ref'].max():+.6f}] rad")
    print(f"theta_ref_sat ratio: {df['theta_ref_sat'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
