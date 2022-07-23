# Author: J094
# Email: jun.guo.chn@outlook.com
# Date: 2022.07.22
# Description: Extract KITTI imu and gnss data from raw data for ORB_SLAM3 evaluation.
#              The imu data and gnss data are stored in EuRoC format.

import numpy as np
import csv
from scipy.spatial.transform import Rotation
import datetime
import cv2

import pykitti

# Change this to the directory where you store KITTI data.
basedir = './KITTI/raw'

# Specify the dataset to load.
date = '2011_10_03'
drive = '0042'

# Specify the file to write.
file_4_imu = "./imu_" + date + "_drive_" + drive + ".txt"
file_4_gnss = "./gnss_" + date + "_drive_" + drive + ".txt"
with open(file_4_imu, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
with open(file_4_gnss, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])

# Specify the timestamps files.
timestamps_extract = "./" + date + "_drive_" + drive + "/timestamps_extract.txt"
timestamps_sync = "./" + date + "_drive_" + drive + "/timestamps_sync.txt"
file_4_timestamps = "./timestamps_" + date + "_drive_" + drive + ".txt"
with open(file_4_timestamps, 'w', newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["# timestamps"])

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
for oxt_data in dataset_extract.oxts:
    imu_data = [oxt_data.packet.wx, oxt_data.packet.wy, oxt_data.packet.wz,
                oxt_data.packet.ax, oxt_data.packet.ay, oxt_data.packet.az]
    with open(file_4_imu, 'a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(imu_data)
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
    q_w = qwi[3]
    q_x = qwi[0]
    q_y = qwi[1]
    q_z = qwi[2]
    qwi = [q_w, q_x, q_y, q_z]
    # if twi[0] == origin_twi[0]:
    #     print(Twi)
    #     print(qwi)
    #     print(Twi[:3, 3])
    twi = Twi[:3, 3]
    pwi = np.concatenate([twi, qwi], axis=0)
    with open(file_4_gnss, 'a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(pwi)

# Grab timestamps.
with open(timestamps_sync, 'r') as f:
    line = f.readlines()[0]
    dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
    timestamps_origin = dt.microsecond / 1e6 + dt.second + dt.minute * 60 + dt.hour * 360
with open(timestamps_extract, 'r') as f:
    for line in f.readlines():
        dt = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
        tp = dt.microsecond / 1e6 + dt.second + dt.minute * 60 + dt.hour * 360 - timestamps_origin
        with open(file_4_timestamps, 'a', newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([tp])
