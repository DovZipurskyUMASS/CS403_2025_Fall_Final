import mujoco
import numpy as np
from scipy.linalg import inv, eig, expm, solve_discrete_are

#mass of pendulum = 0.2
#inertia of pendulum 0.001 0.003 0.003
#for PID: need to have a target value we can calculate error from
#useful mujoco things: d.qpos, d.qvel, d.qacc to get the quaternions of the position,
#velocity, and acceleration of the joints in global frame
#things like d.body_pos are localy framed

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
    #ctrl0 = [0, 0, 0, 0, 0, 0]
    pend_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
    pend_len = 0.42
    pend_mass = 0.2
    MassM = np.zeros((self.m.nv, self.m.nv))
    g = 9.81
    dt = float(self.m.opt.timestep)

    A_cont = np.array([[0.0, 1.0],
                      [g / pend_len, 0.0]])
    B_cont = np.array([[0.0],
                      [-1.0 / pend_len]])
    
    M = np.block([[A_cont, B_cont],
                      [np.zeros((1, 3))]])
    Md = expm(M * dt)
    n = A_cont.shape[0]
    Ad = Md[:n, :n]
    Bd = Md[:n, n:n+1] 

    Q = np.diag([200.0, 30.0])   
    R = np.array([[0.1]])    
        
    P = solve_discrete_are(Ad, Bd, Q, R)
    Kd = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)  # shape (1,2)


    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    tau_g = np.zeros(self.m.nv)
    #initialized_ctrl = True

    p_hand = self.d.xpos[pend_id]   # shape (3,)
    p_pole = self.d.geom_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "mass")]   # shape (3,)
    v = p_pole - p_hand #world position of pole to measure angle from global horizontal

    theta = float(np.arctan2(v[0], v[2]))

    angvel = self.d.cvel[pend_id, 3:6]  # shape (3,)
    theta_dot = float(angvel[0]) 

    x = np.array([theta, theta_dot])

    u_des = float(- (Kd @ x))
    a_hand = np.array([u_des, 0.0, 0.0], dtype=np.float64)
    _J_prev = np.zeros((3, self.m.nv), dtype=np.float64) 

    #F_hand = np.array([pend_mass * u_des, 0.0, 0.0])

    mujoco.mj_jacBody(self.m, self.d, jacp, jacr, hand_id)
    Jv = jacp
    
    Jdot = (Jv - _J_prev) / dt
    Jdot_qdot = Jdot @ self.d.qvel[:self.m.nv]
    _J_prev[:] = Jv.copy()

    A = Jv[:, :self.m.nu]          # 3 x nu (actuated columns)
    rhs = a_hand - Jdot_qdot

    AA_T = A @ A.T + np.eye(3) 
    qdd_act = A.T @ np.linalg.solve(AA_T, rhs)   # nu-vector

    qacc_des = np.zeros(self.m.nv, dtype=np.float64)
    qacc_des[:self.m.nu] = qdd_act

    #actual inverse kinematics
    mujoco.mj_forward(self.m, self.d)
    mujoco.mj_fullM(self.m, MassM, self.d.qM)

    tau_all = MassM @ qacc_des + self.d.qfrc_bias

    tau_act = tau_all[:self.m.nu].copy()

    mujoco.mj_rne(self.m, self.d, 0, tau_g)  # last arg tau array
    tau_g = tau_g[:self.m.nu]

    q_err = self.init_qpos[:self.m.nu] - self.d.qpos[:self.m.nu]
    tau_null = 0.5 * q_err - 0.5 * self.d.qvel[:self.m.nu]

    q_err = self.init_qpos[:self.m.nu] - self.d.qpos[:self.m.nu]
    tau_null = q_err * self.d.qvel[:self.m.nu]
    tau = tau_act + tau_null

    self.d.ctrl[:self.m.nu] = tau
    
    return True 


