import sys
sys.path.append('.\helper_utils')

# custom helper functions and classes 
from realsense_manager import DeviceManager
import skeleton_tracking as st
from user_control import UserControl
import lung_measurement as lung_measure 

from skeletontracker import skeletontracker
import util as cm

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

import cv2
import time
import pyrealsense2 as rs
import math

import random
import pandas as pd

class DeviceLungMeasure:
	def __init__(self, save=False, playback_file=None, test_name='test1', save_dir='D:/realsense_demos/test_run_1/', model_dir='D:/realsense_demos/models/latest_model.pkl'):

		self.test_name = test_name
		self.save_dir = save_dir
		self.model_dir = model_dir
		self.playback_file = playback_file

		self.save = save
		self.count = 0 
		self.x1 = []
		self.y1 = []
		self.line1, = plt.plot(self.x1, self.y1)

		self.chair_surface_to_head = 0
		self.chair_surface_to_shoulder = 0
		self.chest_width = 0
		self.shoulder_width = 0
		self.baseline_chest_pos = 0
		self.skeletons = None
		self.image = None

		self.fvc = 0 
		self.fev1 = 0
		self.fev1_fvc = 0
		self.pef = 0

		self.skeleton_tracking_results = [] 
		self.original_depths_imgs = []
		self.depth_colorized_imgs = []
		self.baseline_chest_positions = [] 

		self.chest_depth_rois = []
		self.chest_color_rois = []
		self.colorized_depths = []
		self.chest_metrics = []

		self.neck_movement_status_list = []
		self.shoulder_movement_status_list = []
		self.side_movement_status_list = []
		self.rocking_movement_status_list = []
		self.foot_pos_status_list = []

	def compute_movement_analysis(self, state, skeletons, color_image, depth_scaled, depth_image_color, deviceManager, joint_confidence): 
		
		# save skeleton tracking information 
		if state.lung_measure and self.save: 
			np.save(os.path.join(self.save_dir, self.demo_name, 'skeletons', 'skeleton_coords_' + str(self.count) + '.npy'), skeletons[0])
		
		# render leg position on image 
		if len(skeletons) > 0: 
			try: 
				leg_position_status = st.detect_leg_pos_and_angle(depth_image_color, skeletons[0])
				if state.lung_measure: 
					self.foot_pos_status_list.append(leg_position_status)
			except:
				print('error in leg position and angle')
			try:    
				if state.set_baseline: 
					self.original_x = np.mean([skeletons[0].joints[5][0], skeletons[0].joints[2][0]])
				side_movement_status = st.computed_side_movement(depth_image_color, skeletons[0], self.original_x)
				if state.lung_measure: 
					self.side_movement_status_list.append(side_movement_status)
			except Exception as e: 
				print('error in side to side movement')
				print(e)
			try: 
				if state.set_baseline: 
					x_ls, y_ls = int(skeletons[0].joints[5][0]), int(skeletons[0].joints[5][1])
					x_rs, y_rs = int(skeletons[0].joints[2][0]), int(skeletons[0].joints[2][1])
					x_mid, y_mid = int(skeletons[0].joints[1][0]), int(skeletons[0].joints[1][1])
					z_ls, z_rs, z_mid = depth_scaled[y_ls, x_ls], depth_scaled[y_rs, x_rs], depth_scaled[y_mid, x_mid]
					self.ref_z = np.mean([z_ls, z_rs, z_mid])
				rocking_status = st.compute_rocking(depth_scaled, depth_image_color, skeletons[0], self.ref_z)
				if state.lung_measure: 
					self.rocking_movement_status_list.append(rocking_status)
			except Exception as e: 
				print('error in rocking')
				print(e)
			try:
				if state.set_baseline: 
					self.ref_shoulder_height = np.mean([y_rs, y_ls])
				shoulder_lift_status = st.compute_shoulder_lifts(depth_image_color, skeletons[0], self.ref_shoulder_height)
				if state.lung_measure: 
					self.shoulder_lift_status_list.append(shoulder_lift_status)
			except Exception as e: 
				print('error shoulder height')
				print(e)
			try:
				if state.set_baseline: 
					neck_features = [int(skeletons[0].joints[0][0]), int(skeletons[0].joints[14][0]), int(skeletons[0].joints[15][0]), int(skeletons[0].joints[16][0]), int(skeletons[0].joints[17][0])]
					self.ref_neck = np.mean(neck_features)
				neck_status = st.compute_neck_movement(depth_image_color, skeletons[0], self.ref_neck)
				if state.lung_measure: 
					self.neck_movement_status_list.append(neck_status)
			except Exception as e: 
				print('error neck movement')
				print(e)
		
		cm.render_result(skeletons, depth_image_color, joint_confidence)
		st.render_ids_3d(depth_image_color, skeletons, deviceManager.depth_frame_filtered, deviceManager.depth_intrinsics, joint_confidence)

	def compute_chest_displacements(self, state, img, skeletons, depth_image, depth_scaled, depth_image_color): 
		
		# obtain baseline region of interest for chest using skeleton points
		if state.set_baseline and len(skeletons) > 0:  
			curr_y_ls, curr_y_rs = int(skeletons[0].joints[5][1]), int(skeletons[0].joints[2][1])
			curr_x_ls = int(skeletons[0].joints[5][0])
			self.curr_x_rs = int(skeletons[0].joints[2][0])
			self.curr_x_la, self.curr_x_ra = int(skeletons[0].joints[6][0]), int(skeletons[0].joints[3][0])
			self.curr_x_lw, curr_y_lw = int(skeletons[0].joints[11][0]), int(skeletons[0].joints[11][1])
			self.curr_x_rw, curr_y_rw = int(skeletons[0].joints[8][0]), int(skeletons[0].joints[8][1])

			self.roi_y0 = max(curr_y_ls, curr_y_rs) # top h eight of roi 
			self.roi_y1 = min(curr_y_lw, curr_y_rw) # bottom height of roi 
			self.roi_y1 = int(self.roi_y0 + ((self.roi_y1 - self.roi_y0) * .7)) # bottom height of roi 
			self.roi_y0 -= 10

			self.baseline_chest_pos = np.mean(depth_scaled[self.roi_y0:self.roi_y1,self.curr_x_rw:self.curr_x_lw])
		
		cv2.rectangle(depth_image_color, (self.curr_x_rw, self.roi_y0), (self.curr_x_lw, self.roi_y1), (0,0,0), 2)
		#cv2.putText(depth_image_color, 'dist: ' + str(round(self.baseline_chest_pos, 2)), (self.curr_x_rw, self.roi_y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

		if state.lung_measure:  
			self.chest_depth_rois.append(np.mean(depth_scaled[self.roi_y0:self.roi_y1,self.curr_x_ra:self.curr_x_la]))
			self.chest_color_rois.append(depth_image_color[self.roi_y0:self.roi_y1,self.curr_x_ra:self.curr_x_la])
			self.chest_metrics.append(self.chest_depth_rois[-1])
			self.colorized_depths.append(depth_image_color)
			
	def compute_external_features(self, state, skeletons, depth_image_color, deviceManager): 

		# compute sitting heights 
		if state.set_baseline and len(skeletons) > 0:   
			self.chair_surface_pt = (int(np.mean([int(skeletons[0].joints[12][0]), int(skeletons[0].joints[9][0])])), int(np.mean([skeletons[0].joints[12][1], skeletons[0].joints[9][1]])))
			avg_head_x = np.mean([int(skeletons[0].joints[0][0]), int(skeletons[0].joints[14][0]), int(skeletons[0].joints[15][0]), int(skeletons[0].joints[16][0]), int(skeletons[0].joints[17][0])])
			avg_head_y = np.mean([int(skeletons[0].joints[0][1]), int(skeletons[0].joints[14][1]), int(skeletons[0].joints[15][1]), int(skeletons[0].joints[16][1]), int(skeletons[0].joints[17][1])])
			#self.head_pt = (int(avg_head_x), int(avg_head_y)) 
			self.head_pt = (int(skeletons[0].joints[0][0]), int(skeletons[0].joints[0][1]))
			self.curr_x_ls, self.curr_y_ls = int(skeletons[0].joints[5][0]), int(skeletons[0].joints[5][1])
			self.curr_x_rs, self.curr_y_rs = int(skeletons[0].joints[2][0]), int(skeletons[0].joints[2][1])
			self.avg_shoulder_pt = (int(np.mean([int(skeletons[0].joints[5][0]), int(skeletons[0].joints[2][0])])), int(np.mean([int(skeletons[0].joints[5][1]), int(skeletons[0].joints[2][1])])))

			try: 
				self.chair_surface_to_head = self.calculate_distance(self.chair_surface_pt, self.head_pt, deviceManager.depth_frame, deviceManager.color_intrinsics, axis='y')
				self.chair_surface_to_shoulder = self.calculate_distance(self.chair_surface_pt, self.avg_shoulder_pt, deviceManager.depth_frame, deviceManager.color_intrinsics, axis='y')
			except:  
				pass 
		
		# visualize chair-head height 
		cv2.line(depth_image_color, (self.curr_x_ls + 65, self.head_pt[1]), (self.curr_x_ls + 65, self.chair_surface_pt[1]), (0, 0, 0), thickness=2)
		cv2.putText(depth_image_color, str(round(self.chair_surface_to_head,2)) + ' m', (self.curr_x_ls + 67, self.head_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

		# visualize chair-shoulder height
		cv2.line(depth_image_color, (self.curr_x_ls + 70, self.avg_shoulder_pt[1]), (self.curr_x_ls + 70, self.chair_surface_pt[1]), (0, 0, 0), thickness=2)
		cv2.putText(depth_image_color, str(round(self.chair_surface_to_shoulder,2)) + ' m', (self.curr_x_ls + 72, self.avg_shoulder_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

		print('CHAIR TO HEAD: ', self.chair_surface_to_head)
		print('CHAIR TO SHOULDER: ', self.chair_surface_to_shoulder)
		
		# compute chest and shoulder width measurements 
		if state.set_baseline and len(skeletons) > 0:   
			curr_x_la, curr_x_ra = int(skeletons[0].joints[6][0]), int(skeletons[0].joints[3][0])
			
			# self.chest_p1_y, self.chest_p2_y = int(np.mean([int(skeletons[0].joints[5][1]), int(skeletons[0].joints[11][1])])), int(np.mean([int(skeletons[0].joints[2][0]), int(skeletons[0].joints[8][1])]))
			# self.chest_width_p1 = (curr_x_la, self.chest_p1_y)
			# self.chest_width_p2 = (curr_x_ra, self.chest_p2_y)
			# self.chest_width = self.calculate_distance(self.chest_width_p1, self.chest_width_p2, deviceManager.depth_frame, deviceManager.color_intrinsics, axis='x')

			try: 
				self.chest_width_point = [[self.curr_x_rw, self.roi_y0], [self.curr_x_lw, self.roi_y1]]  	
				mid_chest_y = (self.roi_y0 + self.roi_y1) // 2
				self.chest_width = self.calculate_distance((self.curr_x_lw, mid_chest_y), (self.curr_x_rw, mid_chest_y), deviceManager.depth_frame, deviceManager.color_intrinsics, axis='x')
				self.shoulder_width = self.calculate_distance((self.curr_x_ls, self.avg_shoulder_pt[1]), (self.curr_x_rs, self.avg_shoulder_pt[1]), deviceManager.depth_frame, deviceManager.color_intrinsics, axis='x')
			except: 
				pass

		# visualize chest width
		cv2.putText(depth_image_color, 'width: ' + str(round(self.chest_width, 2)) + ' m', (self.curr_x_rw, self.roi_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		cv2.putText(depth_image_color, 'dist: ' + str(round(self.baseline_chest_pos, 2)) + ' m', (self.curr_x_rw, self.roi_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

		# visualize shoulder width 
		cv2.line(depth_image_color, (self.curr_x_ls, self.avg_shoulder_pt[1]), (self.curr_x_rs, self.avg_shoulder_pt[1]), (0, 0, 0), thickness=2)
		cv2.putText(depth_image_color, str(round(self.shoulder_width, 2)) + ' m', (self.curr_x_ls + 5, self.avg_shoulder_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		
		# # place text for lung function values (FVC, FEV1, FEV1/FVC)
		# cv2.putText(depth_image_color, 'Chair-Head: ' + str(round(self.chair_surface_to_head, 3)), (465, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		# cv2.putText(depth_image_color, 'Chair-Should: ' + str(round(self.chair_surface_to_shoulder, 3)), (465, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		# cv2.putText(depth_image_color, 'Chest-Width: ' + str(round(self.chest_width, 3)), (465, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		# cv2.putText(depth_image_color, 'Should-Width: ' + str(round(self.shoulder_width, 3)), (465, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		
		print('CHEST WIDTH: ', self.chest_width)
		print('SHOULDER WIDTH: ', self.shoulder_width)

	def compute_lung_params(self, state, depth_image_color): 
		if not state.lung_measure and len(self.y1) > 50: 
				
			# translate from depth information to lung volume scaling 
			test_pft_results = lung_measure.translate_chest_to_lung_params(self.y1, self.model_dir)
			
			# display results
			cv2.putText(depth_image_color, 'FVC: ' + str(round(test_pft_results[0], 2)), (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'FEV1: ' + str(round(test_pft_results[1], 2)), (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'FEV1/FVC: ' + str(round(test_pft_results[2], 2)), (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'PEF: ' + str(round(test_pft_results[3], 2)), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

	def start_lung_measure(self, joint_confidence=.2): 

		# intialize user control class 
		state = UserControl()

		# initialize realsense
		deviceManager = DeviceManager(playback_file=self.playback_file)
		deviceManager.startDevice()
		pc = rs.pointcloud()

		# initialize the cubemos api (skeleton tracking)
		skeletrack = skeletontracker(cloud_tracking_api_key="")

		# create window for result
		cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)

		# capture necessary information 
		self.capture_information(skeletrack, joint_confidence, deviceManager, state)

		# end device when finished 
		deviceManager.stopDevice()
		 
	def capture_information(self, skeletrack, joint_confidence, deviceManager, state): 

		# intialize matplotlib figure 
		fig = plt.figure()

		# redraw the canvas
		fig.canvas.draw()

		# convert canvas to image
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
				sep='')
		img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		while True: 

			# set timer for fps
			script_start = time.time()

			#
			# Collecting Necessary Data From RealSense
			#

			if not self.playback_file: 
				deviceManager.toggleEmitter(state.laser)
			result = deviceManager.getFrames()
			
			if not result: 
				continue
			
			# get necessary depth information 
			depth_image = np.asanyarray(deviceManager.depth_frame_filtered.get_data())
			depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)
			depth_scaled = depth_image * deviceManager.pixel_to_meters_scaling

			# get necessary color information
			color_image = np.asanyarray(deviceManager.color_frame.get_data())
			color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_CUBIC)
			color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

			# resize image
			img = cv2.resize(img, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_CUBIC)

			# 
			# set visualization information
			# 
			box_coords = ((5, 20), (5 + 255, 20 + 130))
			cv2.rectangle(depth_image_color, box_coords[0], box_coords[1], ((255, 255, 255)), cv2.FILLED)
			#cv2.rectangle(depth_image_color, (450, 20), (645, 150), ((255, 255, 255)), cv2.FILLED)

			# place text for lung function values (FVC, FEV1, FEV1/FVC)
			cv2.rectangle(depth_image_color, (5, 200), (5 + 255, 200 + 125), (255, 255, 255), cv2.FILLED)
			cv2.putText(depth_image_color, 'FVC: ', (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'FEV1: ', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'FEV1/FVC: ', (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			cv2.putText(depth_image_color, 'PEF: ', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			
			if not state.set_baseline: 
				cv2.putText(depth_image_color, 'Compute Baseline: OFF', (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			else:
				cv2.putText(depth_image_color, 'Compute Baseline: ON', (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
			
			# save depth information 
			if state.lung_measure and self.save: 
				np.save(os.path.join(self.save_dir, self.demo_name, 'depth_imgs', 'depth_img_' + str(self.count) + '.npy'), depth_image)

			# perform inference and update the tracking id
			skeletons = skeletrack.track_skeletons(color_image)
			self.skeletons = skeletons

			#
			# Run Skeleton Tracking & Movement Detections
			#
			self.compute_movement_analysis(state, skeletons, color_image, depth_scaled, depth_image_color, deviceManager, joint_confidence)
			
			#
			# Obtain Chest Region of Interests Information 
			#
			self.compute_chest_displacements(state, img, skeletons, depth_image, depth_scaled, depth_image_color)

			#
			# Compute Physiological Measures
			#
			self.compute_external_features(state, skeletons, depth_image_color, deviceManager)
			
			#
			# Store and Generate Breathing Graph 
			#   
			if state.lung_measure: 

				self.x1.append(self.count)
				self.y1.append(self.chest_metrics[-1])
				self.line1, = plt.plot(self.x1, self.y1, 'k') 
	 
				# redraw the canvas
				fig.canvas.draw()

				# convert canvas to image
				img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
						sep='')
				img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
				img = cv2.resize(img, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_CUBIC)

			# 
			# Compute Lung Parameters
			#
			self.compute_lung_params(state, depth_image_color)
			

			# end script time for FPS
			script_end = time.time() - script_start

			#
			# Visualize information 
			#
			cv2.setWindowTitle(
				state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
				(deviceManager.w, deviceManager.h, 1.0/script_end, script_end*1000, "PAUSED" if state.paused else ""))
				
			final = cv2.hconcat([depth_image_color, img])
			cv2.imshow(state.WIN_NAME, final)

			#
			# Set User Control
			#
			key = cv2.waitKey(1)
			if key == 32:
				state.lung_measure = 1 - state.lung_measure
				
			if key == ord("b"):
				state.set_baseline = 1 - state.set_baseline
				
			if key == ord("v"):
				state.compute_volume = 1 - state.compute_volume
				
			if key == ord("l"):
				state.laser = 1 - state.laser
				
			if key == ord("e"):
				self.x1 = []
				self.y1 = []
				
				fig.clear()
				state.lung_measure = False

			if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
				break

			# increment count 
			self.count += 1

	def record_coaching_information():

		return 
 

	def export_data(self): 

		save_dic = {}
		save_dic['test_name'] = self.test_name
		save_dic['save_dir'] = self.save_dir

		save_dic['x1'] = self.x1
		save_dic['y1'] = self.y1
		save_dic['chair_surface_to_head'] = self.chair_surface_to_head
		save_dic['chair_surface_to_shoulder'] = self.chair_surface_to_shoulder
		save_dic['chest_width'] = self.chest_width
		save_dic['shoulder_width'] = self.shoulder_width

		save_dic['neck_movement_status_list'] = self.neck_movement_status_list
		save_dic['shoulder_movement_status_list'] = self.shoulder_movement_status_list
		save_dic['side_movement_status_list'] = self.side_movement_status_list
		save_dic['rocking_movement_status_list'] = self.rocking_movement_status_list
		save_dic['foot_pos_status_list'] = self.foot_pos_status_list

		save_dic['fvc'] = self.fvc
		save_dic['fev1'] = self.fev1
		save_dic['fev1_fvc'] = self.fev1_fvc
		save_dic['pef'] = self.pef

		np.save(os.path.join(self.save_dir, 'patient_info.npy'), save_dic) 

		print('SUCCESS')

	def convert_depth_frame_to_pointcloud(self, depth_image, camera_intrinsics):
		"""
		Convert the depthmap to a 3D point cloud

		Parameters:
		-----------
		depth_frame : rs.frame()
		The depth_frame containing the depth map
		camera_intrinsics : The intrinsic values of the image in whose coordinate system the depth_frame is computed

		Return:
		----------
		x : array
			The x values of the pointcloud in meters
		y : array
			The y values of the pointcloud in meters
		z : array
			The z values of the pointcloud in meters

		"""

		[height, width] = depth_image.shape

		nx = np.linspace(0, width-1, width)
		ny = np.linspace(0, height-1, height)
		u, v = np.meshgrid(nx, ny)
		x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
		y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

		z = depth_image.flatten() / 1000
		x = np.multiply(x,z)
		y = np.multiply(y,z)

		x = x[np.nonzero(z)]
		y = y[np.nonzero(z)]
		z = z[np.nonzero(z)]

		return x, y, z

	def calculate_chest_width(self, roi, depth_frame, intrinsics): 

		# obtain bounds 
		x_min, x_max = roi[0][0], roi[1][0]
		y_min, y_max = roi[0][1], roi[1][1]

		# obtain depth image
		depth_image = np.asanyarray(depth_frame.get_data())
		depth_image = depth_image[x_min:x_max,y_min:y_max].copy()

		# threshold image
		x_mean, y_mean = depth_image.shape[0] // 2, depth_image.shape[1] // 2 
		center_dist = depth_image[x_mean, y_mean]
		depth_image[depth_image > center_dist + .2] = 0
		depth_image[depth_image < center_dist - .2] = 0

		# obtain average distance to threshold chest
		x, y, z = self.convert_depth_frame_to_pointcloud(depth_image, intrinsics)

		# obtain width
		width = abs(x.max() - x.min()) 

		return width

	def calculate_distance(self, p1, p2, depth_frame, intrinsics, axis='both'):
	
		ix, iy = p1
		x, y = p2
		
		print(p1, p2)
		
		udist = depth_frame.get_distance(ix, iy)
		vdist = depth_frame.get_distance(x, y)

		point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], udist)
		point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], vdist) 

		print(point1)
		print(point2)

		if axis == 'x': 
			dist = abs(point1[0] - point2[0])
		elif axis == 'y': 
			dist = abs(point1[1] - point2[1])
		else: 
			dist = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(point1[2] - point2[2], 2))
		
		return dist