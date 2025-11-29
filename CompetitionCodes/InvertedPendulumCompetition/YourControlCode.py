import mujoco
import numpy as np
from scipy.linalg import inv, eig, solve_discrete_are, expm

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

class YourCtrl:
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()

        # Control gains (using similar values to CircularMotion)
        self.kp = 50.0
        self.kd = 3.0

        self.dt = self.m.opt.timestep # might need altering later 

        self.pendulum_body_id = self.m.body("pendulum").id
        self.pendulum_joint_id = self.m.joint("pend_roll").id
        self.hand_body_id = self.m.body("EE_Frame").id

        self.pj_pos_idx = self.m.jnt_qposadr[self.pendulum_joint_id]
        self.pj_vel_idx = self.m.jnt_dofadr[self.pendulum_joint_id]

        self.K = self.LqrGain()

        #l = pole_length #TODO UPDATE THESE BC I THINK USED IN LQR?
        #m_p = pole_mass 

    def GetState(self): #get theta, theta dot of pendulum #returns a tuple
        th = self.d.qpos[self.pj_pos_idx]
        th_dot = self.d.qvel[self.pj_vel_idx]
        return np.array([th, th_dot])

    def LqrGain(self, v):
        pend_len = 0.42
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

        Q = np.diag([200.0, 2.0])   
        R = np.array([[0.01]])    
        
        P = solve_discrete_are(Ad, Bd, Q, R)
        Kd = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)  # shape (1,2) 
        return Kd 

    def CtrlUpdate(self):
        ## 1) read pendulum state vector make into np.array? ok update that
        x = self.GetState()
        ## 3) LQR gain K (precomputed offline or compute once in init)
        # u_des is desired hand horizontal acceleration
        u_des = -self.K.dot(x) # u = - K * x

        ## saturate acceleration
        #u_des = np.clip(u_des, -amax, amax)
        u_des = np.clip(u_des, -50, 50) # what do we actually set the clipping values at #TODO
        ## 4) desired hand force
        F_hand = np.array([u_des, 0, 0]) 
        # todo should this be u_des or u_Des multipleid with mass of p?
        ## 5) Jacobian J_trans (3 x n). Use MuJoCo helper or finite diff if needed.
        #J = compute_translational_jacobian(self.m, self.d, hand_body_id)
        point = np.zeros(3) # attached to body so offset from body
        jacr = None # omega focused so orientation/angular stuff 
        jacp = np.zeros((3, self.m.nv)) # linear velocity focus?
        mujoco.mj_jac(self.m, self.d, jacp, jacr, point, self.hand_body_id) #mj_jac returns void
        ## 6) task torques (simple map)
        #tau_task = J.T.dot(F_hand)
        tau_task = jacp.T.dot(F_hand)
        ## 7) gravity compensation (MuJoCo can compute it; if not approximate)
        #tau_g = compute_gravity_torques(self.m, self.d)  # size n
        # void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result)
        tau_g = np.zeros(self.m.nv)
        mujoco.mj_rne(self.m, self.d, 0, tau_g)
        ## 8) posture (nullspace) PD to hold init_qpos, low gain
        #kp_null = 5.0
        #kd_null = 0.5
        #q_err = self.init_qpos - self.d.qpos
        q_err = self.init_qpos - self.d.qpos
        kp_null = 5.0
        kd_null = 0.5
        #tau_null = kp_null * q_err - kd_null * self.d.
        tau_null = kp_null * q_err - kd_null * self.d.qvel
        ## project nullspace: in practice you can add a small tau_null scaled, or compute nullspace projector
        ## 9) final torque and limits
        tau = tau_task + tau_g + 0.05 * tau_null # why 0.05? ?? can we do something else?
        #tau = np.clip(tau, -tau_max, tau_max)
        tau = np.clip(tau, -200, 200) # why 200? # what do we actually set the clipping values at #TODO
        #self.d.ctrl[:len(tau)] = tau[:self.m.nu] # dont think this is right, d.ctrl[] is  (nu x 1) 
        # but that doesnt seem consistent with what we are using, which is nv a lot. which better fits 
        # use of d.qfrc_applied ? 
        self.d.qfrc_applied[:self.nv] = tau 
        return True

