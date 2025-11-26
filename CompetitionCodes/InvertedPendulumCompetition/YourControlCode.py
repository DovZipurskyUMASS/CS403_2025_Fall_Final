import mujoco
import numpy as np
from scipy.linalg import inv, eig, expm, solve_discrete_are

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

#pseudocode for LQR
"""
def CtrlUpdate(self):
    # 0) constants (set elsewhere)
    l = pole_length
    m_p = pole_mass
    g = 9.81
    dt = self.m.opt.timestep

    # 1) read pendulum state
    p_hand = self.d.xpos[hand_body_id]    # [x,y,z]
    p_tip  = self.d.xpos[pole_tip_body_id]
    v = p_tip - p_hand
    theta = math.atan2(v[0], v[2])   # x horizontal, z up; adjust if axes differ

    # theta_dot: prefer body angvel
    av = self.d.angvel[pole_body_id]    # body angular velocity (3-vector)
    theta_dot = av[1]  # choose correct component depending on axis of rotation
    # alternatively: finite diff with LPF

    # 2) state vector
    x = np.array([theta, theta_dot])

    # 3) LQR gain K (precomputed offline or compute once in init)
    u_des = -K.dot(x)   # u_des is desired hand horizontal acceleration

    # saturate acceleration
    u_des = np.clip(u_des, -amax, amax)

    # 4) desired hand force
    F_hand = np.array([m_p * u_des, 0.0, 0.0])

    # 5) Jacobian J_trans (3 x n). Use MuJoCo helper or finite diff if needed.
    J = compute_translational_jacobian(self.m, self.d, hand_body_id)

    # 6) task torques (simple map)
    tau_task = J.T.dot(F_hand)

    # 7) gravity compensation (MuJoCo can compute it; if not approximate)
    tau_g = compute_gravity_torques(self.m, self.d)  # size n

    # 8) posture (nullspace) PD to hold init_qpos, low gain
    kp_null = 5.0
    kd_null = 0.5
    q_err = self.init_qpos - self.d.qpos
    tau_null = kp_null * q_err - kd_null * self.d.qvel
    # project nullspace: in practice you can add a small tau_null scaled, or compute nullspace projector

    # 9) final torque and limits
    tau = tau_task + tau_g + 0.05 * tau_null
    tau = np.clip(tau, -tau_max, tau_max)

    self.d.ctrl[:len(tau)] = tau[:self.m.nu]
    return True
"""
#pend length = 0.42
#mass of pendulum = 0.2
class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 50.0
    self.kd = 3.0


  def CtrlUpdate(self):
    #relevant values
    ctrl0 = [0, 0, 0, 0, 0, 0]
    nu = self.m.nu #number of actuators in the system
    nv = self.m.nv #DOF of system
    dq = np.zeros(nv)

    R = np.eye(nu) #set Matrix R to the identity matrix, can tweak this later
    Q = np.eye(2*nv) #set Matrix Q to identity, tweak later

    A = np.zeros((2*nv, 2*nv)) 
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6 #delta for mujoco's computation

    #function that computes finite, discrete time transistion matrices. Technically computes four,
    #but we set the last two params to None bc we only care about A and B
    #the values are placed in the A and B we defined above, no need to explicitly 
    #capture return values.
    mujoco.mjd_transitionFD(self.m, self.d, epsilon, True, A, B, None, None)

    #compute P on way to K, K being our actual gain matrix (sick)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    mujoco.mj_differentiatePos(self.m, dq, 1, self.init_qpos, self.d.qpos)
    dx = np.hstack((dq, self.d.qvel)).T

    self.d.ctrl = ctrl0 - K @ dx
    
    return True 



