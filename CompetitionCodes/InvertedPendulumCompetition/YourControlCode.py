import mujoco
import numpy as np
from scipy.linalg import inv, eig

#mass of pendulum = 0.2
#inertia of pendulum 0.001 0.003 0.003
#for PID: need to have a target value we can calculate error from
#useful mujoco things: d.qpos, d.qvel, d.qacc to get the quaternions of the position,
#velocity, and acceleration of the joints in global frame
#things like d.body_pos are localy framed

#chat gpt's pseudocode for pendulum balancing:
'''
def CtrlUpdate(self):
    # 1) compute pendulum vector and angle
    p_hand = self.d.xpos[hand_body_id]   # 3-vector
    p_tip  = self.d.xpos[pole_tip_body_id]
    v = p_tip - p_hand                   # pole vector pointing from hand to tip

    # theta: small-angle about x axis in the x-z plane (x horizontal, z up)
    theta = math.atan2(v[0], v[2])
    # differentiate theta (filtered)
    if not hasattr(self, "theta_prev"):
        self.theta_prev = theta
        self.t_prev = 0.0
        self.theta_dot = 0.0
    dt = self.m.opt.timestep
    raw_theta_dot = (theta - self.theta_prev) / dt
    # simple filter
    alpha = 0.2
    self.theta_dot = alpha*raw_theta_dot + (1-alpha)*self.theta_dot
    self.theta_prev = theta

    # 2) desired pivot acceleration (linear controller or LQR)
    kp_pend = 120.0   # tune these
    kd_pend = 20.0
    xdd_des = -kp_pend * theta - kd_pend * self.theta_dot

    # clamp
    xdd_des = max(min(xdd_des, 50.0), -50.0)

    # 3) desired force at hand (mass of pole m_p)
    F_x = pole_mass * xdd_des
    F = np.array([F_x, 0.0, 0.0])

    # 4) map to joint torques: tau = J^T * F
    J = compute_translational_jacobian(self.m, self.d, hand_body_id)  # 3 x n
    tau_task = J.T.dot(F)   # n vector

    # 5) gravity compensation (simple)
    tau_g = compute_gravity_compensation(self.m, self.d)  # n vector, use MuJoCo function if available

    # 6) final torques + saturation
    tau = tau_task + tau_g
    tau = np.clip(tau, -max_tau, max_tau)

    self.d.ctrl[:len(tau)] = tau[:self.m.nu]
    return True

'''
class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 50.0
    self.kd = 3.0


  def CtrlUpdate(self):
    for i in range(6):
       self.d.ctrl[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]
    return True 



