import numpy as np
import sympy as sym
import time
import pickle
import os
from scipy import sparse
from scipy.sparse import linalg as splinalg

class EKF:
    '''
    A simple extended Kálmán filter implementation for tracking separate MuscleGroups.
    Takes in the motor position and velocity measurements and outputs the joint
    position and velocity estimates.
    Notation (follows the Wikipedia page for EKF):
        - n_joints: number of joints in the muscle group
        - n_motors: number of motors in the muscle group
        - x: state vector, shape (2*n_joints,1) (composed of joint positions and velocities)
        - z: measurement vector, shape (2*n_motors,1) (composed of motor positions and velocities)
        - A: state transition matrix, shape (2*n_joints,2*n_joints)
        - P: state covariance matrix, shape (2*n_joints,2*n_joints)
        - y: measurement residual, shape (2*n_motors,1) (difference between the predicted and actual measurements)
        - h: predicted measurement, shape (2*n_motors,1)
        - H: measurement Jacobian, shape (2*n_motors,2*n_joints)
        - Q: process noise covariance matrix, shape (2*n_joints,2*n_joints)
        - R: measurement noise covariance matrix, shape (2*n_motors,2*n_motors)
        - S: residual covariance matrix, shape (2*n_motors,2*n_motors)
        - K: Kálmán gain, shape (2*n_joints,2*n_motors)
        
        The dynamics are assumed to be based on a simple constant velocity model.
        That is, the state x is propagated as:
        x = A * x
        where A is an identity matrix with the upper right quadrant replaced by
        the identity times the time step.
    '''
    def __init__(self, muscle_group, gc, init_p = 0.01):
        # initialize the matrices
        self.n_joints = len(muscle_group.joint_ids)
        self.n_motors = len(muscle_group.motor_ids)
        # note the muscle group name (this is needed as the thumb needs separate treatment)
        self.muscle_group_name = muscle_group.name
        self.joint_ranges = np.deg2rad(muscle_group.joint_ranges)
        # define just the matrices that should keep their value across updates
        self.x = np.zeros((2 * self.n_joints))
        self.A = np.eye(2 * self.n_joints)
        self.P = init_p * np.eye(2 * self.n_joints)
        self.Q = np.deg2rad(2) ** 2 * np.eye(2 * self.n_joints)  # 2 degrees
        self.R = 0.005 ** 2 * np.eye(2 * self.n_motors)  # 5 mm

        # initialize the symbolic measurement Jacobian
        sym_joints = []
        for joint_i in range(self.n_joints):
            sym_joints.append(sym.Symbol(str(joint_i)))

        # computing p and the Jacobian takes time, as symbolic computation is slow.
        # therefore, try to use cached values if possible
        p_pickle_file = f'/tmp/p_{muscle_group.name}_p2.pkl'
        J_pickle_file = f'/tmp/J_{muscle_group.name}_p2.pkl'
        compute_p_and_J = True
        if os.path.exists(p_pickle_file) and os.path.exists(J_pickle_file):
            # here we must check if the pickle files are up to date.
            # compare timestamps of the pickle files and the source code
            # (there may be better ways to check if the pickle files are still valid)
            sources_to_check = ["finger_kinematics.py", "hand_controller.py", "hand_defs.yaml"]
            # get the oldest timestamp from the pickle files
            pickle_file_oldest_timestamp = min(os.path.getmtime(p_pickle_file), os.path.getmtime(J_pickle_file))
            # get the newest timestamp from the source code
            this_dir = os.path.dirname(os.path.realpath(__file__))
            source_newest_timestamp = max([os.path.getmtime(os.path.join(this_dir, source)) for source in sources_to_check])
            if source_newest_timestamp < pickle_file_oldest_timestamp:
                # the pickle files are up to date
                compute_p_and_J = False
                print(f"Using cached p and J for muscle group {muscle_group.name}")
                with open(p_pickle_file, 'rb') as f:
                    p = pickle.load(f)
                with open(J_pickle_file, 'rb') as f:
                    J = pickle.load(f)
        if compute_p_and_J:
            # initialize the symbolic measurement Jacobian
            print(f"Computing p and J for muscle group {muscle_group.name}")
            p = gc.pose2motors_sym(*sym_joints, muscle_group=muscle_group)
            q = sym_joints
            J = gc.calculate_Jacobian(p, q)
            # save the computed values to pickle files
            with open(p_pickle_file, 'wb') as f:
                pickle.dump(p, f)
            with open(J_pickle_file, 'wb') as f:
                pickle.dump(J, f)
        self.J_func = sym.lambdify(sym_joints, J)
        self.pose2motors = sym.lambdify(sym_joints, p)
        self.last_t = time.time()

    def update(self, motor_pos : np.ndarray, motor_speed : np.ndarray):
        '''
        Steps the EKF forward by:
            - predicting the state x and the measurement h
            - computing the residual y and updating the state x and
                its covariance matrix P with the Kálmán gain K times
                the residual
        '''
        dt = time.time() - self.last_t
        self.A[:self.n_joints, self.n_joints:] = dt * np.eye(self.n_joints)
        self.last_t = time.time()

        z = np.concatenate((motor_pos, motor_speed))

        # predict
        self.x[:] = self.A @ self.x
        self.P[:] = self.P + self.Q

        # update the joint-muscle model based on prediction
        motor_pos_pred = self.pose2motors(*self.x[:self.n_joints]).reshape(-1)
        J_pred = self.J_func(*self.x[:self.n_joints])
        h = np.concatenate((motor_pos_pred, J_pred @ self.x[self.n_joints:]))
        H = np.block([[J_pred, np.zeros((self.n_motors, self.n_joints))],
                      [np.zeros((self.n_motors, self.n_joints)), J_pred]])

        # update
        y = z - h
        S = H @ self.P @ H.T + self.R

        # running np.linalg.inv on S is slow, so leverage the fact that S is sparse for more efficient computation
        S_sparse = sparse.csc_matrix(S)
        # print(f"{self.muscle_group_name} S: {S}")
        K = self.P @ H.T @ splinalg.inv(S_sparse)

        self.x[:] = self.x + K @ y
        self.P[:] = (np.eye(2 * self.n_joints) - K @ H) @ self.P

        # EKF can predict a little bit outside the joint limits but not beyond this value, to avoid wacky predictions
        # set it to zero for now since base joint of thumb is unstable even at margin of 10 degrees
        margin = np.deg2rad(0)

        self.x[:self.n_joints] = np.clip(self.x[:self.n_joints], self.joint_ranges[:, 0] - margin, self.joint_ranges[:, 1] + margin)

        return self.x[:self.n_joints], self.x[self.n_joints:]
