import os
import cv2
import face_recognition
import dlib
import numpy as np
from PIL import Image

#_mask_img=Image.open("blue-mask.png")

#width,height=_mask_img.width,_mask_img.height
#print(width_mask)
def _mask_face(face_landmark:dict,frame):
	
	nose_bridge = face_landmark['nose_bridge']
	
	nose_point	= nose_bridge[len(nose_bridge) *1//4]
	#print(type(nose_point))
	nose_v		= np.array(nose_point)
	chin		= face_landmark['chin']
	chin_len	= len(chin)#17
	
	chin_bottom_point =chin[chin_len //2]#8
	chin_bottom_v	  = np.array(chin_bottom_point)
	chin_left_point= chin[chin_len //8]
	chin_right_point=chin[chin_len *7//8]
	width_ratio=1.2
	new_height = int(np.linalg.norm(nose_v-chin_bottom_v))
	#print(new_height)
	#left
	mask_left_img=_mask_img.crop((0,0,width//2,height))
	mask_left_img = _mask_img.crop((0, 0, width // 2, height))
	mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
	mask_left_width = int(mask_left_width * width_ratio)
	mask_left_img = mask_left_img.resize((mask_left_width, new_height))

	# right
	mask_right_img = _mask_img.crop((width // 2, 0, width, height))
	mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
	mask_right_width = int(mask_right_width * width_ratio)
	mask_right_img = mask_right_img.resize((mask_right_width, new_height))

	# merge mask
	size = (mask_left_img.width + mask_right_img.width, new_height)
	mask_img = Image.new('RGBA', size)
	mask_img.paste(mask_left_img, (0, 0), mask_left_img)
	mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

	# rotate mask
	angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
	rotated_mask_img = mask_img.rotate(angle, expand=True)
	
	# calculate mask location
	center_x = (nose_point[0] + chin_bottom_point[0]) // 2
	center_y = (nose_point[1] + chin_bottom_point[1]) // 2

	offset = mask_img.width // 2 - mask_left_img.width
	radian = angle * np.pi / 180
	box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
	
	box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

		# add mask	   
		
	frame.paste(mask_img, (box_x, box_y),mask_img)
	
	
	
def get_distance_from_point_to_line(point, line_point1, line_point2):
		distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
						  (line_point1[0] - line_point2[0]) * point[1] +
						  (line_point2[0] - line_point1[0]) * line_point1[1] +
						  (line_point1[1] - line_point2[1]) * line_point1[0]) / \
				   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
						   (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
		return int(distance)	
cap=cv2.VideoCapture(0)
_mask_img=Image.open("blue-mask.png")

width,height=_mask_img.width,_mask_img.height
while True:
	
	_,frame=cap.read()
	face_locations=face_recognition.face_locations(frame,model="hog")
	
	face_landmarks=face_recognition.face_landmarks(frame,face_locations)
	for face_landmark in face_landmarks:
		frame=Image.fromarray(frame)
		_mask_face(face_landmark,frame)
	frame= np.array(frame)
	cv2.imshow('frame',frame)		 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()