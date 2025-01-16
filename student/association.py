# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Association:
    """Data association class with single nearest neighbor association and gating based on Mahalanobis distance"""

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # Number of tracks and measurements
        N = len(track_list)
        M = len(meas_list)

        # Initialize unassigned track and measurement lists
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))

        # Initialize the association matrix with infinity
        self.association_matrix = np.inf * np.ones((N, M))

        # Fill the association matrix with Mahalanobis distances
        for i in range(N):
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]

                #  Compute Mahalanobis distance
                dist = self.MHD(track, meas, KF)
                # Check if it's inside gating limit
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i, j] = dist
                else:
                    self.association_matrix[i, j] = np.inf

        ############
        # END student code
        ############

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        # Shorthand for the association matrix
        A = self.association_matrix

        # If all entries are inf, we have no valid associations left
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # Retrieve indices of the minimum distance
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape)
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # Delete the chosen row and column
        A = np.delete(A, ind_track, 0)
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        # Retrieve the actual track and measurement IDs from unassigned lists
        update_track = self.unassigned_tracks[ind_track]
        update_meas = self.unassigned_meas[ind_meas]

        # Remove them from unassigned lists
        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        ############
        # END student code
        ############
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        return MHD < limit

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        # Retrieve measurement matrix H for this sensor/track
        H = meas.sensor.get_H(track.x)

        # Residual (z - h(x))
        gamma = meas.z - meas.sensor.get_hx(track.x)

        # Residual covariance S
        S = KF.S(track, meas, H)

        # MHD = gamma^T * S^-1 * gamma
        MHD = gamma.transpose() * np.linalg.inv(S) * gamma

        # Return as a scalar
        return MHD[0, 0]

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while (
            self.association_matrix.shape[0] > 0
            and self.association_matrix.shape[1] > 0
        ):

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print("---no more associations---")
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print(
                "update track",
                track.id,
                "with",
                meas_list[ind_meas].sensor.name,
                "measurement",
                ind_meas,
            )
            KF.update(track, meas_list[ind_meas])

            # update score and track state
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print("track", track.id, "score =", track.score)
