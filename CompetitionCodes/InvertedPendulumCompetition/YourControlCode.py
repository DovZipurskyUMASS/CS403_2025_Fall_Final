import mujoco
import numpy as np
from scipy.linalg import inv, eig

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

    def GetState(self): #get theta, theta dot of pendulum #returns a tuple
        th = self.d.qpos[self.pj_pos_idx] # is this right?
        th_dot = self.d.qvel[self.pj_vel_idx]
        return np.array([th, th_dot])

    def LqrGain(self, v):
        k = 0
        return k 
    
    def gravityComp(self):
        return 0

    def CtrlUpdate(self):
        # 0) constants (set elsewhere)
        #l = pole_length
        #m_p = pole_mass
        ## 1) read pendulum state vector make into np.array? ok update that
        x = self.GetState()
        ## 3) LQR gain K (precomputed offline or compute once in init)
        # u_des is desired hand horizontal acceleration
        u_des = -self.K.dot(x) # u = - K * x

        ## saturate acceleration
        #u_des = np.clip(u_des, -amax, amax)
        u_des = np.clip(u_des, -50, 50)
        ## 4) desired hand force
        F_hand = np.array([u_des, 0, 0]) 
        # todo should this be u_des or u_Des multipleid with mass of p?
        ## 5) Jacobian J_trans (3 x n). Use MuJoCo helper or finite diff if needed.
        #J = compute_translational_jacobian(self.m, self.d, hand_body_id)
        point = np.zeros(3) # attached to body so offset from body
        jacr = None # omega focused so orientation/angular stuff 
        jacp = np.zeros((3, self.m.nv)) # linear velocity focus?
        mujoco.mj_jac(self.m, self.d, jacp, jacr, point, self.hand_body_id)
        # returns void

        ## 6) task torques (simple map)
        #tau_task = J.T.dot(F_hand)
        tau_task = jacp.T.dot(F_hand)
        ## 7) gravity compensation (MuJoCo can compute it; if not approximate)
        #tau_g = compute_gravity_torques(self.m, self.d)  # size n
        self.gravityComp() #TODO left off here
        ## 8) posture (nullspace) PD to hold init_qpos, low gain
        #kp_null = 5.0
        #kd_null = 0.5
        #q_err = self.init_qpos - self.d.qpos
        #tau_null = kp_null * q_err - kd_null * self.d.qvel
        ## project nullspace: in practice you can add a small tau_null scaled, or compute nullspace projector
        ## 9) final torque and limits
        #tau = tau_task + tau_g + 0.05 * tau_null
        #tau = np.clip(tau, -tau_max, tau_max)
        #self.d.ctrl[:len(tau)] = tau[:self.m.nu]
        return True

