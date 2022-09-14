# Author: J094
# Email: jun.guo.chn@outlook.com
# Date: 2022.07.22
# Description: Extract KITTI imu and gnss data from raw data for ORB_SLAM3 evaluation.
#              The imu data and gnss data are stored in EuRoC format.

from math import degrees
import numpy as np
import csv
from scipy.spatial.transform import Rotation
import datetime
import cv2

import pykitti

def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.split("\n")
    lst = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    return lst

# Change this to the directory where you store KITTI data.
basedir = './KITTI/raw'

# Specify the dataset to load.
date = '2011_10_03'
drive = '0027'

# Specify the file to write.
file_4_imu = "./imu_" + date + "_drive_" + drive + ".txt"
file_4_gnss = "./gnss_" + date + "_drive_" + drive + ".txt"
with open(file_4_imu, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# timestamp", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
with open(file_4_gnss, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# timestamp", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])

# Specify the timestamps files.
timestamps_extract = "./" + date + "_drive_" + drive + "/timestamps_extract.txt"
timestamps_sync = "./" + date + "_drive_" + drive + "/timestamps_sync.txt"
file_4_timestamps = "./timestamps_" + date + "_drive_" + drive + ".txt"
with open(file_4_timestamps, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# timestamps"])

# Specify the gt files.
gt_pose = "./" + date + "_drive_" + drive + "/pose.txt"
gt_times = "./" + date + "_drive_" + drive + "/times.txt"
file_4_gt = "./gt_" + date + "_drive_" + drive + ".txt"
with open(file_4_gt, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# timestamp", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])

# Grab pose and times.
pose = read_file_list(gt_pose)
times = read_file_list(gt_times)
for i in range(len(pose)):
    p = pose[i]
    t = times[i][0] * 1e9
    transf = np.asmatrix(np.array([[p[0], p[1], p[2], p[3]],
                                   [p[4], p[5], p[6], p[7]],
                                   [p[8], p[9], p[10],p[11]],
                                   [0.,    0.,    0.,    1.]]))
    # print(transf)
    transf_inv = np.linalg.inv(transf)
    # print(transf_inv)
    rot_mat = transf[0:3, 0:3]
    trans = np.asarray(transf[0:3, 3]).transpose()[0]
    # print(trans)
    # print(rot_mat)
    # print(trans)
    rot = Rotation.from_matrix(rot_mat)
    # Eu = rot.as_euler("xyz", degrees=True)
    # rot = Rotation.from_euler("zyx", Eu, degrees=True)
    rot_q = rot.as_quat()
    q_w = rot_q[3]
    q_x = rot_q[0]
    q_y = rot_q[1]
    q_z = rot_q[2]
    rot_q_n = [q_w, q_x, q_y, q_z]
    pos = [t] + list(trans) + list(rot_q_n)
    with open(file_4_gt, 'a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(pos)

# Grab timestamps.
with open(timestamps_sync, 'r') as f:
    line = f.readlines()[0]
    dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
    timestamps_origin = dt.microsecond / 1e6 + dt.second + dt.minute * 60 + dt.hour * 360

timestamps = []
with open(timestamps_extract, 'r') as f:
    for line in f.readlines():
        dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
        tp = dt.microsecond / 1e6 + dt.second + dt.minute * 60 + dt.hour * 360 - timestamps_origin
        with open(file_4_timestamps, 'a', newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([tp])
        timestamps.append(tp)

# Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.raw(basedir, date, drive)
dataset_extract = pykitti.raw(basedir, date, drive, dataset="extract")
dataset_sync = pykitti.raw(basedir, date, drive, dataset="sync")

# Grab imu and gnss data.
# Calculate scale and origin for extracted data.
scale = np.cos(dataset_sync.oxts[0].packet.lat * np.pi / 180.)
# scale = 1
origin_twi_xy = scale * (dataset_sync.oxts[0].T_w_imu[:2, 3])
origin_twi_z = dataset_sync.oxts[0].T_w_imu[2, 3]
origin_twi = np.concatenate([origin_twi_xy, [origin_twi_z]], axis=0)
origin_Rwi = dataset_sync.oxts[0].T_w_imu[:3, :3]
origin_Twi = np.zeros((4, 4), dtype=np.float64)
origin_Twi[:3, :3] = origin_Rwi
origin_Twi[:3, 3] = origin_twi
origin_Twi[3, :] = [0,0,0,1]
# print(origin_Twi)
# cv2.imshow("cam0", np.array(dataset_sync.get_cam0(0)))
# cv2.waitKey(0)

for i in range(len(dataset_extract.oxts)):
    oxt_data = dataset_extract.oxts[i]
    tp = timestamps[i] * 1e9
    imu_data = [oxt_data.packet.wx, oxt_data.packet.wy, oxt_data.packet.wz,
                oxt_data.packet.ax, oxt_data.packet.ay, oxt_data.packet.az]
    imu4write = [tp] + imu_data
    with open(file_4_imu, 'a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(imu4write)
    twi_xy = scale * (oxt_data.T_w_imu[:2, 3])
    twi_z = oxt_data.T_w_imu[2, 3]
    twi = np.concatenate([twi_xy, [twi_z]], axis=0)
    Rwi = oxt_data.T_w_imu[:3, :3]
    Twi = np.zeros((4, 4), dtype=np.float64)
    Twi[:3, :3] = Rwi
    Twi[:3, 3] = twi
    Twi[3, :] = [0,0,0,1]
    Twi = np.linalg.inv(origin_Twi).dot(Twi)
    rot = Rotation.from_matrix(Twi[:3, :3])
    qwi = rot.as_quat()
    # Creat guassian noise for t and q.
    # noise_t = np.random.normal(loc=0.0, scale=5e-3, size=3)
    # noise_Eu = np.random.normal(loc=0.0, scale=5e-1, size=3)
    # Add gaussian noise to qwi.
    # Eu = Rotation.from_quat(qwi).as_euler("xyz", degrees=True)
    # new_Eu = Eu + noise_Eu
    # qwi = Rotation.from_euler("xyz", new_Eu, degrees=True).as_quat()
    q_w = qwi[3]
    q_x = qwi[0]
    q_y = qwi[1]
    q_z = qwi[2]
    qwi_n = [q_w, q_x, q_y, q_z]
    # if twi[0] == origin_twi[0]:
    #     print(Twi)
    #     print(qwi)
    #     print(Twi[:3, 3])
    twi = Twi[:3, 3]
    # Add gaussian noise to twi.
    # twi = twi + noise_t
    pwi = np.concatenate([twi, qwi_n], axis=0)
    gnss4write = [tp] + list(pwi)
    with open(file_4_gnss, 'a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(gnss4write)
