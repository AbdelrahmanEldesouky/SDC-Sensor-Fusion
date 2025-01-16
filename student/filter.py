# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params


class Filter:
    """Kalman filter class"""

    def __init__(self):
        self.dt = params.dt
        self.q = params.q  
        self.dim_state = 6  

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        dt = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        return F

        ############
        # END student code
        ############

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        dt = self.dt
        q = self.q

        # Compute common terms
        q1 = ((dt**3) / 3) * q
        q2 = ((dt**2) / 2) * q
        q3 = dt * q

        Q = np.matrix(
            [
                [q1, 0.0, 0.0, q2, 0.0, 0.0],
                [0.0, q1, 0.0, 0.0, q2, 0.0],
                [0.0, 0.0, q1, 0.0, 0.0, q2],
                [q2, 0.0, 0.0, q3, 0.0, 0.0],
                [0.0, q2, 0.0, 0.0, q3, 0.0],
                [0.0, 0.0, q2, 0.0, 0.0, q3],
            ]
        )

        return Q

    ############
    # END student code
    ############

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        # Extract the current state x and covariance P
        x = track.x
        P = track.P

        # Compute F and Q
        F_mat = self.F()
        Q_mat = self.Q()

        # Predict the state: x_pred = F * x
        x_pred = F_mat * x

        # Predict the covariance: P_pred = F * P * F^T + Q
        P_pred = F_mat * P * F_mat.transpose() + Q_mat

        # Store the predictions back to track
        track.set_x(x_pred)
        track.set_P(P_pred)

        ############
        # END student code
        ############

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated meas, save x and P in track
        ############

        # Retrieve the meas matrix H from the sensor
        H_mat = meas.sensor.get_H(track.x)

        # Compute the residual (gamma)
        gamma_val = self.gamma(track, meas)

        # Compute the residual covariance S
        S_mat = self.S(track, meas, H_mat)

        # Compute the Kalman gain: K = P * H^T * S^-1
        P = track.P
        K = P * H_mat.transpose() * np.linalg.inv(S_mat)

        # Update the state: x_new = x_pred + K * gamma
        x_new = track.x + K * gamma_val

        # Update the covariance: P_new = (I - K*H) * P
        I = np.identity(self.dim_state)
        P_new = (I - K * H_mat) * P

        # Store updated values
        track.set_x(x_new)
        track.set_P(P_new)

        ############
        # END student code
        ############
        track.update_attributes(meas)

    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        z_meas = meas.z
        z_pred = meas.sensor.get_hx(track.x)

        # residual
        gamma_val = z_meas - z_pred
        return gamma_val

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        P = track.P
        R = meas.R

        S = H * P * H.transpose() + R
        return S

        ############
        # END student code
        ############
