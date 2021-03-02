import sys
sys.path.append('C:\Program Files\Cubemos\SkeletonTracking\samples\python')

import numpy as np
import math 
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import closing
import cv2
import pyrealsense2 as rs

# =====================================================
# SKELETON TRACKING FUNCTIONS
# =====================================================
def render_ids_3d(render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence):
	
	thickness = 1
	text_color = (255, 255, 255)
	rows, cols, channel = render_image.shape[:3]
	distance_kernel_size = 5
	
	# calculate 3D keypoints and display them
	for skeleton_index in range(len(skeletons_2d)):
		skeleton_2D = skeletons_2d[skeleton_index]
		joints_2D = skeleton_2D.joints
		did_once = False
		for joint_index in range(len(joints_2D)):
			if did_once == False:
				cv2.putText(
					render_image,
					"id: " + str(skeleton_2D.id),
					(int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.55,
					text_color,
					thickness,
				)
			did_once = True
			# check if the joint was detected and has valid coordinate
			if skeleton_2D.confidences[joint_index] > joint_confidence:
				distance_in_kernel = []
				low_bound_x = max(
					0,
					int(
						joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
					),
				)
				upper_bound_x = min(
					cols - 1,
					int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
				)
				low_bound_y = max(
					0,
					int(
						joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
					),
				)
				upper_bound_y = min(
					rows - 1,
					int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
				)
				for x in range(low_bound_x, upper_bound_x):
					for y in range(low_bound_y, upper_bound_y):
						distance_in_kernel.append(depth_map.get_distance(x, y))
				median_distance = np.percentile(np.array(distance_in_kernel), 50)
				depth_pixel = [
					int(joints_2D[joint_index].x),
					int(joints_2D[joint_index].y),
				]
				if median_distance >= 0.3:
					point_3d = rs.rs2_deproject_pixel_to_point(
						depth_intrinsic, depth_pixel, median_distance
					)
					point_3d = np.round([float(i) for i in point_3d], 3)
					point_str = [str(x) for x in point_3d]
					
					cv2.putText(
						render_image,
						#str(joint_index), 
						str(point_3d),
						(int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
						cv2.FONT_HERSHEY_DUPLEX,
						0.4,
						text_color,
						thickness,
						)

# =====================================================
# MOVEMENT & POSITION TRACKING FUNCTIONS
# =====================================================
					
def compute_orientation(img, display=False):
	
	# apply threshold
	thresh = threshold_otsu(img)
	bw = closing(img > thresh, square(3))

	label_img = label(bw)
	props = regionprops(label_img)

	# largest img
	largest_index = np.argmax([p.area for p in props])
	prop = props[largest_index]
	
	if display: 

		plt.figure()
		plt.imshow(prop.image)

		x0 = prop['Centroid'][1]
		y0 = prop['Centroid'][0]
		x2 = x0 - math.sin(prop['Orientation']) * 0.9 * prop['MinorAxisLength']
		y2 = y0 - math.cos(prop['Orientation']) * 0.9 * prop['MinorAxisLength']

		plt.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
		plt.plot(x0, y0, '.g', markersize=15)

		minr, minc, maxr, maxc = prop['BoundingBox']
		bx = (minc, maxc, maxc, minc, minc)
		by = (minr, minr, maxr, maxr, minr)
		
		plt.plot(bx, by, '-b', linewidth=2.5)
		plt.show()
		
	return prop.orientation

def detect_legs(img, skeleton): 
	
	x_rk, y_rk = int(skeleton.joints[9][0]), int(skeleton.joints[9][1])
	x_ra, y_ra = int(skeleton.joints[10][0]), int(skeleton.joints[10][1])
	x_lk, y_lk = int(skeleton.joints[12][0]), int(skeleton.joints[12][1])
	x_la, y_la = int(skeleton.joints[13][0]), int(skeleton.joints[13][1])
	
	# zero out each leg
	img_left_leg = np.zeros(img.shape,np.uint8)
	img_left_leg[y_lk:y_la, x_la-30:x_la+30] = img[y_lk:y_la, x_la-30:x_la+30]
	img_right_leg = np.zeros(img.shape, np.uint8)
	img_right_leg[y_rk:y_ra, y_ra-30:y_ra+30] = img[y_rk:y_ra, x_ra-30:x_ra+30]

	# threshold each leg 
	gray_left = cv2.cvtColor(img_left_leg, cv2.COLOR_BGR2GRAY)
	ret_left, thresh_left = cv2.threshold(gray_left, 0, 255, cv2.THRESH_OTSU)
	gray_right = cv2.cvtColor(img_right_leg, cv2.COLOR_BGR2GRAY)
	ret_right, thresh_right = cv2.threshold(gray_right, 0, 255, cv2.THRESH_OTSU)

	# find contours for each leg 
	x_l, y_l, w_l, h_l = cv2.boundingRect(thresh_left)
	x_r, y_r, w_r, h_r = cv2.boundingRect(thresh_right)

#     final_img = cv2.rectangle(img, (x_l, y_l), (x_l + w_l, y_l + h_l), (36,255,12), 2)
#     final_img = cv2.rectangle(final_img, (x_r, y_r), (x_r + w_r, y_r + h_r), (36,255,12), 2)

	# display
	#cv2.putText(final_img, 'left leg', (x_l, y_l-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
	#cv2.putText(final_img, 'right leg', (x_r, y_r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

	cv2.imshow('image', final_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# distance formula
def distance(x1, y1, x2,y2):
	dist = math.sqrt((math.fabs(x2-x1))**2 + ((math.fabs(y2-y1)))**2)
	return dist

def compute_angle(x1, y1, x2, y2):
	
	# quantifies the hypotenuse of the triangle
	hypotenuse =  distance(x1, y1, x2, y2)
	
	#quantifies the horizontal of the triangle
	horizontal = distance(x1, y1, x2, y1)
	
	# makes the third-line of the triangle
	thirdline = distance(x2, y2, x2, y1)
	
	#calculates the angle using trigonometry
	angle = np.arcsin((thirdline / hypotenuse)) * 180 / math.pi
	
	return angle
	
def detect_leg_pos_and_angle(img, skeleton):
	
	x_rk, y_rk = int(skeleton.joints[9][0]), int(skeleton.joints[9][1])
	x_ra, y_ra = int(skeleton.joints[10][0]), int(skeleton.joints[10][1])
	
	x_lk, y_lk = int(skeleton.joints[12][0]), int(skeleton.joints[12][1])
	x_la, y_la = int(skeleton.joints[13][0]), int(skeleton.joints[13][1])
	
	left_angle = compute_angle(x_lk, y_lk, x_la, y_la)
	right_angle = compute_angle(x_rk, y_rk, x_ra, y_ra)
	
	#(255,12,36), (36,255,12)
	if (left_angle <= 80 or left_angle >= 100) or (right_angle <= 80 or right_angle >= 100): 
		cv2.putText(img, 'Leg Position: Bad', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		return 0
	else: 
		cv2.putText(img, 'Leg Position: Good', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		return 1
		
def detect_leg_pos_old(img, skeleton, display=False): 
	
	x_rk, y_rk = int(skeleton.joints[9][0]), int(skeleton.joints[9][1])
	x_ra, y_ra = int(skeleton.joints[10][0]), int(skeleton.joints[10][1])
	x_lk, y_lk = int(skeleton.joints[12][0]), int(skeleton.joints[12][1])
	x_la, y_la = int(skeleton.joints[13][0]), int(skeleton.joints[13][1])
		
	# zero out each leg
	img_left_leg = np.zeros(img.shape,np.uint8)
	img_left_leg[y_lk:y_la-15, x_la-70:x_la+70] = img[y_lk:y_la-15, x_la-70:x_la+70]
	img_right_leg = np.zeros(img.shape, np.uint8)
	img_right_leg[y_rk:y_ra-15, x_ra-70:x_ra+70] = img[y_rk:y_ra-15, x_ra-70:x_ra+70]

	# threshold each leg 
	gray_left = cv2.cvtColor(img_left_leg, cv2.COLOR_BGR2GRAY)
	ret_left, thresh_left = cv2.threshold(gray_left, 0, 255, cv2.THRESH_OTSU)
	gray_right = cv2.cvtColor(img_right_leg, cv2.COLOR_BGR2GRAY)
	ret_right, thresh_right = cv2.threshold(gray_right, 0, 255, cv2.THRESH_OTSU)

	# obtain contour
	cntrs_left = cv2.findContours(thresh_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cntrs_left = cntrs_left[0] if len(cntrs_left) == 2 else cntrs_left[1]
	cntrs_left = max(cntrs_left, key=cv2.contourArea)
	cntrs_right = cv2.findContours(thresh_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cntrs_right = cntrs_right[0] if len(cntrs_right) == 2 else cntrs_right[1]
	cntrs_right = max(cntrs_right, key=cv2.contourArea)

	# get bounding box for legs
	x_l, y_l, w_l, h_l = cv2.boundingRect(cntrs_left)
	x_r, y_r, w_r, h_r = cv2.boundingRect(cntrs_right)

	# fit ellipsoid to find orientation 
	ellipse_left, ellipse_right = cv2.fitEllipse(cntrs_left), cv2.fitEllipse(cntrs_right)

	_, angle_l = get_angle_orienation(img, ellipse_left)
	_, angle_r = get_angle_orienation(img, ellipse_right)
	
	print('ANGLES: ', angle_l, angle_r)

	# display 
	if angle_l >= 40 and angle_l <= 120: 
		color_l = (36,255,12)
		status_l = True
	else: 
		status_l = False

	if angle_r >= 50 and angle_r <= 100:
		color_r = (36,255,12)
		status_r = True
	else: 
		status_r = False
		
	cv2.putText(img, str(round(angle_l, 1)), (x_l, y_l-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,12,36), 2)
	cv2.putText(img, str(round(angle_r, 1)), (x_r, y_r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,12,36), 2)
	
	if not status_r or not status_l: 
		cv2.putText(img, 'Leg Position: Bad', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 3)
		
		#cv2.putText(img, str(round(angle_l, 1)), (x_l, y_l-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,12,36), 2)
		cv2.rectangle(img, (x_l, y_l), (x_l + w_l, y_l + h_l), (255,12,36), 2)

		#cv2.putText(img, str(round(angle_r, 1)), (x_r, y_r-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,12,36), 2)
		cv2.rectangle(img, (x_r, y_r), (x_r + w_r, y_r + h_r), (255,12,36), 2)
	else: 
		cv2.putText(img, 'Leg Position: Good', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
	
	if display: 
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	return (status_r and status_l)
		
def get_angle_orienation(img, ellipse, draw_line=False): 
	
	result = None
	(xc, yc), (d1, d2), angle = ellipse
	
	# draw orienation line
	rmajor = max(d1, d2) / 2
	if angle > 90:
		angle = angle - 90
	else:
		angle = angle + 90

	xtop = xc + math.cos(math.radians(angle))*rmajor
	ytop = yc + math.sin(math.radians(angle))*rmajor
	xbot = xc + math.cos(math.radians(angle+180))*rmajor
	ybot = yc + math.sin(math.radians(angle+180))*rmajor
	
	if draw_line:
		result = cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
	
	return result, angle

def compute_rocking(depth_img, colorized, skeleton, prev_z): 
	
	# extract key points 
	x_ls, y_ls = int(skeleton.joints[5][0]), int(skeleton.joints[5][1])
	x_rs, y_rs = int(skeleton.joints[2][0]), int(skeleton.joints[2][1])
	x_mid, y_mid = int(skeleton.joints[1][0]), int(skeleton.joints[1][1])
	z_ls, z_rs, z_mid = depth_img[y_ls, x_ls], depth_img[y_rs, x_rs], depth_img[y_mid, x_mid]
		
	# compare to last z location 
	avg_position = np.mean([z_ls, z_rs, z_mid])
	
	print('ROCKING: ', prev_z, avg_position)
	
	# display bounding box and status of rocking
	if abs(prev_z - avg_position) > .05: 
		cv2.putText(colorized, 'Rocking Movement: Bad', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		cv2.rectangle(colorized, (x_rs, y_rs-20), (x_ls + 20, y_ls + 20), (255,12,36), 2)
		return 0
	else: 
		cv2.putText(colorized, 'Rocking Movement: Good', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		return 1
		
def computed_side_movement(img, skeleton, prev_x): 
	
	# extract key points 
	x_ls, y_ls = int(skeleton.joints[5][0]), int(skeleton.joints[5][1])
	x_rs, y_rs = int(skeleton.joints[2][0]), int(skeleton.joints[2][1])
	
	# compare to previous x value and display 
	avg_position = np.mean([x_ls, x_rs])
	
	print('SIDE MOVEMENT: ', prev_x, avg_position)
	
	if abs(prev_x - avg_position) > 5: 
		cv2.putText(img, 'Side Movement: Bad', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		cv2.rectangle(img, (x_rs, y_rs-20), (x_ls + 20, y_ls + 20), (255,12,36), 2)
		return 0
	else: 
		cv2.putText(img, 'Side Movement: Good', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		return 1
	
def compute_shoulder_lifts(img, skeleton, ref_y): 
	
	# extract key points 
	y_rs, y_ls = int(skeleton.joints[2][1]), int(skeleton.joints[5][1])
	avg_pos = np.mean([y_rs, y_ls])
	
	print('SHOULDER LIFT: ', ref_y, avg_pos)
	
	if abs(ref_y - avg_pos) > 5: 
		cv2.putText(img, 'Shoulder Movement: Bad', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		cv2.rectangle(img, (x_rs, y_rs-20), (x_ls + 20, y_ls + 20), (255,12,36), 2)
		return 0
	else: 
		cv2.putText(img, 'Shoulder Movement: Good', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		return 1
	
def compute_neck_movement(img, skeleton, ref_neck_pos):
	
	avg_pos = np.mean([int(skeleton.joints[0][0]), int(skeleton.joints[14][0]), int(skeleton.joints[15][0]), int(skeleton.joints[16][0]), int(skeleton.joints[17][0])])
	print('NECK MOVEMENT: ', ref_neck_pos, avg_pos)
	if abs(ref_neck_pos - avg_pos) > 10: 
		cv2.putText(img, 'Neck Movement: Bad', (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		return 0 
	else: 
		cv2.putText(img, 'Neck Movement: Good', (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
		return 1
