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
    self.kp = 8.0
    self.kd = 1.0


  def CtrlUpdate(self):
    #relevant values
    #ctrl0 = [0, 0, 0, 0, 0, 0]
    pend_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
    pend_len = 0.42
    pend_mass = 0.2
    g = 9.81
    dt = float(self.m.opt.timestep)
    
    #nu = self.m.nu #number of actuators in the system
    #nv = self.m.nv #DOF of system
    #dq = np.zeros(nv)

    #R = np.eye(nu) #set Matrix R to the identity matrix, can tweak this later
    #Q = np.eye(2*nv) #set Matrix Q to identity, tweak later

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


    #A = np.zeros((2*nv, 2*nv)) 
    #B = np.zeros((2*nv, nu))
    #epsilon = 1e-6 #delta for mujoco's computation
    #mujoco.mjd_transitionFD(self.m, self.d, epsilon, True, A, B, None, None)
    #P = solve_discrete_are(A, B, Q, R)
    #K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    Q = np.diag([200.0, 2.0])   
    R = np.array([[0.01]])    
        
    P = solve_discrete_are(Ad, Bd, Q, R)
    Kd = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)  # shape (1,2)


    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    tau_g = np.zeros(self.m.nv)
    initialized_ctrl = True

    p_hand = self.d.xpos[self.m.joint("wrist_roll").id]   # shape (3,)
    p_pole = self.d.xpos[pend_id]   # shape (3,)
    v = p_pole - p_hand #world position of pole to measure angle from global horizontal

    theta = float(np.arctan2(v[0], v[2]))

    angvel = self.d.qpos[self.m.joint("wrist_roll").id]  # shape (3,)
    theta_dot = float(angvel) 

    x = np.array([theta, theta_dot])

    #u_des = float(- (Kd @ x))
    u_des = g * -theta + pend_len * (-self.kp * -theta - self.kd * theta_dot)
    u_des = -1 * u_des

    #amax = 20.0
    #u_des = np.clip(u_des, -amax, amax)

    F_hand = np.array([pend_mass * u_des, 0.0, 0.0])

    mujoco.mj_jacBody(self.m, self.d, jacp, jacr, hand_id)

    J = jacp[:, :self.m.nu]  # 3 x nu
    J_prev = getattr(self, '_J_prev', J.copy())
    Jdot = (J - J_prev) / float(self.m.opt.timestep)
    Jdot_qdot = Jdot @ self.d.qvel[:self.m.nu]
    self._J_prev = J.copy()

    xdd_des = np.array([u_des, 0.0, 0.0])

    rhs = xdd_des - Jdot_qdot
    # damped least-squares for robustness
    damp = 1e-4
    JJ = J @ J.T + np.eye(3) * damp
    qdd_des = J.T @ np.linalg.solve(JJ, rhs)   

    mujoco.mj_forward(self.m, self.d)
  
    M = np.zeros((self.m.nv, self.m.nv))
    mujoco.mj_fullM(self.m, M, self.d.qM)

    tau_all = M @ qdd_des + self.d.qfrc_bias
    tau_task = tau_all[:self.m.nu]

    mujoco.mj_rne(self.m, self.d, 0, tau_g)  # last arg tau array
    tau_g = tau_g[:self.m.nu]

    q_err = self.init_qpos[:self.m.nu] - self.d.qpos[:self.m.nu]
    tau_null = 0.5 * q_err - 0.5 * self.d.qvel[:self.m.nu]

    tau = tau_g + tau_task + 0.08 * tau_null

    self.d.ctrl[:self.m.nu] = tau
    print("theta, theta_dot, u_des:", theta, theta_dot, u_des)
    
    return True 



