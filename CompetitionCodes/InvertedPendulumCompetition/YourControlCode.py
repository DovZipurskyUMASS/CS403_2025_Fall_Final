import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

# Heavily referenced from this youtube video: https://www.youtube.com/watch?v=HMyD0IfPHfA
class YourCtrl:
    def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
    
        # Control gains (using similar values to CircularMotion)
        self.kp = 50.0
        self.kd = 5.0
        
        self.nv = self.m.nv #dof: 7
        self.nu = self.m.nu #actuators: 6
        
        self.d_temp = mujoco.MjData(self.m) #temporary system for A, B calculations
        
        #initial position and velocities
        q0  = self.d.qpos.copy() #q1, q2, q3, q4, q5, q6, q7
        qd0 = self.d.qvel.copy() #qdot1, qdot2, qdot3, qdot4, qdot5, qdot6, qdot7
        
        #set temp model to og positions, no velocities or joint inputs
        self.d_temp.qpos[:] = q0
        self.d_temp.qvel[:] = 0.0
        self.d_temp.ctrl[:] = 0.0
        
        mujoco.mj_forward(self.m, self.d_temp)
        
        # generalized forces for starting position
        # emphasize starting position (might be bad for bad starting positions)
        bias = self.d_temp.qfrc_bias.copy()
        u0 = bias[:self.nu].copy() 
        
        #write starting torques to start position (keep it in place)
        self.d.ctrl[:self.nu] = u0
        
        x0 = np.concatenate([q0, qd0])
        inputs0 = np.concatenate([x0, u0])
        f0 = self._f(inputs0)
        
        #perturb f1 and compute A and B
        pert = 1e-2 #adjust value
        A = self._compute_A(x0, u0, f0, pert)
        B = self._compute_B(x0, u0, f0, pert)
        
        #Q and R
        Q = np.eye((2*self.nv)) # maybe not touch Q, pend looks tilted in simulation when adjusted?
        # Q = np.diag([40]*self.nv + [150]*self.nv) # joint pos err penalty, joint vel err penalty
        rho = 0.005 #adjust value
        R = rho * np.eye((self.nu))
        
        #compute K
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
        
        self.x_ref = x0
        self.u_ref = u0
        
        
    def _f(self, inputs): # 14 state + 6 control inputs, 14-dim xdot output
        #state = q1, q1dot, q2, q2dot, q3, q3dot, q4, q4dot, q5, q5dot, q6, q6dot, q7, q7dot
        #inputs = state, u1, u2, u3, u4, u5, u6
        #outputs = q1dot, q1ddot, q2dot, q2ddot, q3dot, q3ddot, q4dot, q4ddot, q5dot, q5ddot, q6dot, q6ddot, q7dot, q7ddot

        q = inputs[0:self.nv]
        qdot = inputs[self.nv:2*self.nv]
        u = inputs[2*self.nv:2*self.nv+self.nu]
        
        #write info into temp data rather than affecting actual model
        self.d_temp.qpos[:] = q
        self.d_temp.qvel[:] = qdot
        self.d_temp.ctrl[:self.nu] = u
        
        mujoco.mj_forward(self.m, self.d_temp)
 
        xdot = np.concatenate([self.d_temp.qvel.copy(), self.d_temp.qacc.copy()])
        
        return xdot
    
    def _compute_A(self, x0, u0, f0, pert):
        A = np.zeros((14, 14))
        for i in range(14):
            x_pert = x0.copy()
            x_pert[i] += pert

            inputs_pert = np.concatenate([x_pert, u0])
            f_pert = self._f(inputs_pert)
            
            A[:, i] = (f_pert - f0) / pert
        return A
        
    def _compute_B(self, x0, u0, f0, pert):
        B = np.zeros((14, 6))
        for i in range(6):
            u_pert = u0.copy()
            u_pert[i] += pert
            
            inputs_pert = np.concatenate([x0, u_pert])
            f_pert = self._f(inputs_pert)
            
            B[:, i] = (f_pert - f0) / pert
        return B
        
    
    def CtrlUpdate(self):
        q  = self.d.qpos.copy() #q1, q2, q3, q4, q5, q6, q7
        qd = self.d.qvel.copy() #qdot1, qdot2, qdot3, qdot4, qdot5, qdot6, qdot7
        x  = np.concatenate([q, qd])

        # updating self.u_ref as qcrf_bias is different depending on updated pos of robot
        self.d_temp.qpos[:] = q # this might just not be useful but doesnt seem to hurt idk
        self.d_temp.qvel[:] = 0.0 # i cant tell
        self.d_temp.ctrl[:] = 0.0
        mujoco.mj_forward(self.m, self.d_temp)
        bias = self.d_temp.qfrc_bias.copy()
        self.u_ref = bias[:self.nu].copy() 
        
        u = self.u_ref - self.K @ (x - self.x_ref)
        self.d.ctrl[:self.nu] = u
        
        return True
