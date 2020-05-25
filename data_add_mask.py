import cv2
import numpy as np
import dlib

def load_yolo():
	net=cv2.dnn.readNet("yolov3_6000.weights","yolov3.cfg")
	classes=[]
	with open ("yolo.names","r") as f:
		classes=[line.strip() for line in f.readlines()]
	layers_names=net.getLayerNames()
	output_layers=[layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	
	return net,classes,output_layers
def detect_objects(img,net,output_layers):
	blob=cv2.dnn.blobFromImage(img,scalefactor=0.000392,size=(320,320),mean=(0,0,0),swapRB=True,crop=False)
	net.setInput(blob)
	outputs=net.forward(output_layers)
	return blob,outputs	   
def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids	  
	
def main():
	predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	img=cv2.imread("quoc.jpg")
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	mask=np.zeros_like(img_gray)
	height,width,chanels=img.shape
	model,classes,output_layers=load_yolo()
	blob,outputs=detect_objects(img,model,output_layers)
	boxes,confs,class_ids=get_box_dimensions(outputs,height,width)
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			landmarks = predictor(img, dlib.rectangle(x,y-20,x+w,y+h))
			landmarks_points = []
			for n in range(2, 15):
				x_ = landmarks.part(n).x
				y_ = landmarks.part(n).y
				landmarks_points.append((x_, y_))
				points = np.array(landmarks_points, np.int32)
				convexhull = cv2.convexHull(points)
	#cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
				cv2.fillConvexPoly(img, convexhull, (255,255,255))
				face_image_1 = cv2.bitwise_and(img, img, mask=mask)
				face=img[y:y+h, x:x+w] 
				cv2.imwrite('quoc1.jpg',face)
	cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)
	cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0), 1)
			
	cv2.imshow("Image", img) 
	while True:
		key=cv2.waitKey(1)
		if key==27:
			break
if __name__=="__main__":
	main()