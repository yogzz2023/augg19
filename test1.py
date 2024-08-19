import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2, multivariate_normal

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9000.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.vx = (self.Z1[0] - self.Z2[0]) / dt
            self.vy = (self.Z1[1] - self.Z2[1]) / dt
            self.vz = (self.Z1[2] - self.Z2[2]) / dt

            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt*dt)/2.0
        T_3 = (dt*dt*dt)/3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt

        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

    def gating(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        d2 = np.dot(np.dot(np.transpose(Inn), np.linalg.inv(S)), Inn)
        return d2 < self.gate_threshold


class Track:
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = 'free'  # 'free' or 'occupied'
        self.measurements = []

    def assign_measurement(self, measurement):
        self.state = 'occupied'
        self.measurements.append(measurement)

    def release(self):
        self.state = 'free'
        self.measurements.clear()

class TrackManager:
    def __init__(self):
        self.tracks = []

    def add_track(self):
        track_id = len(self.tracks) + 1
        self.tracks.append(Track(track_id))

    def get_free_track(self):
        for track in self.tracks:
            if track.state == 'free':
                return track
        # If no free track, add a new one
        self.add_track()
        return self.tracks[-1]

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups


def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])
            ma = float(row[8])
            me = float(row[9])
            mt = float(row[10])
            x, y, z = sph2cart(ma, me, mr)
            measurements.append((mr, ma, me, mt))
    return measurements


def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z


def jpda(clusters, kalman_filter, track_manager):
    hypotheses = clusters

    if not hypotheses:
        return None

    hypothesis_likelihoods = [compute_hypothesis_likelihood(h, kalman_filter) for h in hypotheses]
    total_likelihood = sum(hypothesis_likelihoods)

    if total_likelihood == 0:
        marginal_probabilities = [1.0 / len(hypotheses)] * len(hypotheses)
    else:
        marginal_probabilities = [likelihood / total_likelihood for likelihood in hypothesis_likelihoods]

    best_hypothesis_index = np.argmax(marginal_probabilities)
    best_hypothesis = hypotheses[best_hypothesis_index]

    # Assign the best hypothesis to a free track
    free_track = track_manager.get_free_track()
    free_track.assign_measurement(best_hypothesis)

    print(f"Best hypothesis assigned to Track ID: {free_track.track_id}")
    return best_hypothesis


def compute_hypothesis_likelihood(hypothesis, kalman_filter):
    Z = np.array([[hypothesis[0]], [hypothesis[1]], [hypothesis[2]]])
    Inn = Z - np.dot(kalman_filter.H, kalman_filter.Sp)
    S = np.dot(kalman_filter.H, np.dot(kalman_filter.Pp, kalman_filter.H.T)) + kalman_filter.R
    likelihood = np.exp(-0.5 * np.dot(np.dot(Inn.T, np.linalg.inv(S)), Inn))
    return likelihood


def main():
    file_path = 'ttk_84_test.csv'
    measurements = read_measurements_from_csv(file_path)

    kalman_filter = CVFilter()
    track_manager = TrackManager()

    # Prepopulate some tracks
    for _ in range(5):
        track_manager.add_track()

    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}")

        # Initialize Kalman Filter with the first measurement of the group
        first_measurement = group[0]
        kalman_filter.initialize_filter_state(*sph2cart(first_measurement[1], first_measurement[2], first_measurement[0]), first_measurement[3])

        # Process subsequent measurements in the group
        for measurement in group[1:]:
            x, y, z = sph2cart(measurement[1], measurement[2], measurement[0])
            kalman_filter.predict_step(measurement[3])
            kalman_filter.update_step(np.array([[x], [y], [z]]))

            # Check if the measurement is within the gate
            if kalman_filter.gating(np.array([[x], [y], [z]])):
                print(f"Measurement {measurement} is within the gate.")
            else:
                print(f"Measurement {measurement} is out of the gate.")

            # Hypothetical clustering and hypothesis generation
            # For now, we'll use the Kalman filter state as the hypothesis
            hypothesis = (kalman_filter.Sf[0, 0], kalman_filter.Sf[1, 0], kalman_filter.Sf[2, 0])
            clusters = [hypothesis]  # In practice, this should be your clustering result
            
            # Perform JPDA and assign measurements to tracks
            best_hypothesis = jpda(clusters, kalman_filter, track_manager)
            if best_hypothesis:
                print(f"Best hypothesis for the current measurement: {best_hypothesis}")
            else:
                print("No suitable hypothesis found.")

    # Print final track information
    print("\nFinal track information:")
    for track in track_manager.tracks:
        print(f"Track ID: {track.track_id}, State: {track.state}, Measurements: {track.measurements}")

if __name__ == "__main__":
    main()

