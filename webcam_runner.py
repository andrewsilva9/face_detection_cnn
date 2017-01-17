import cv2
import face_detector

# class CameraRunner:
# 	def __init__(self):
# 		self.image_width = 350
# 		self.image_height = 350
# 		self.capture = cv2.VideoCapture(0)


face_finder = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
face_finder2 = cv2.CascadeClassifier("cascades/haarcascade_alt2.xml")
face_finder3 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
face_finder4 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt_tree.xml")


def draw_boxes(im, boxes):
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	for i in range(x1.shape[0]):
		cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
	return im


def run():
	capture = cv2.VideoCapture(0)
	finder = face_detector.FaceDetector()
	while True:
		ret, frame = capture.read()
		# Our operations on the frame come here
		input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# find face bounding boxes
		faces, _ = finder.find_faces(input_image)

		face1 = face_finder.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
		# face2 = face_finder2.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
		# face3 = face_finder3.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
		# face4 = face_finder4.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

		# Go over detected faces, stop at first detected face, return empty if no face.
		if len(face1) == 1:
			face = face1[0]
		# elif len(face2) == 1:
		# 	face = face2[0]
		# elif len(face3) == 1:
		# 	face = face3[0]
		# elif len(face4) == 1:
		# 	face = face4[0]
		else:
			face = []

		new_img = draw_boxes(frame, faces)
		if len(face) > 0:
			cv2.rectangle(new_img, (int(face[0]), int(face[1])), (int(face[0] + face[2]), int(face[1] + face[3])), (255, 0, 0), 1)
		cv2.imshow('frame', new_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	capture.release()
	cv2.destroyAllWindows()
	# # Cut and save face
	# # scale image
	# gray = gray[face[1]:face[1] + face[3],
	# 			face[0]:face[0] + face[2]]
	# out = cv2.resize(gray, (self.image_height, self.image_width))  # Resize face so all images have same size
	# return self.emotions[self.classifier.classify_emotion(out)]

run()

