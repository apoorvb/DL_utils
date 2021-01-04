import numpy as np
import cv2

def bbox(frame):

	net = cv2.dnn.readNet("/Users/apb/Desktop/Work/github/py_practice/yolov3.weights",
							"/Users/apb/Desktop/Work/github/py_practice/darknet/cfg/yolov3.cfg")

	layer_names = net.getLayerNames()
	outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	height, width, channel = frame.shape

	#Detect objects
	blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)
	net.setInput(blob)
	outs = net.forward(outputlayers)

	#Getting bboxes
	class_ids=[]
	confidences=[]
	boxes=[]
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.3:
			#object detected
				center_x= int(detection[0]*width)
				center_y= int(detection[1]*height)
				w = int(detection[2]*width)
				h = int(detection[3]*height)

				#cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
				#rectangle co-ordinaters
				x=int(center_x - w/2)
				y=int(center_y - h/2)
				#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

				boxes.append([x,y,w,h]) #put all rectangle areas
				confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
				class_ids.append(class_id) #name of the object tha was detected


	indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

	colors= np.random.uniform(0,255,size=(80,3))

	for i in range(len(boxes)):
		if i in indexes:
			x,y,w,h = boxes[i]
			confidence= confidences[i]
			color = colors[class_ids[i]]
			cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)




def main():
	cap = cv2.VideoCapture(0)

	font = cv2.FONT_HERSHEY_PLAIN


	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		frame = cv2.flip(frame, 1)
		# Our operations on the frame come here
		bbox(frame)
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":

	main()
