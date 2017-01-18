import cv2
import face_detector

eye_finder = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")


def run():
	capture = cv2.VideoCapture(0)
	finder = face_detector.FaceDetector()
	while True:
		ret, frame = capture.read()
		# Our operations on the frame come here
		input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# find face bounding boxes
		faces, _ = finder.find_faces(input_image)

		for face in faces:
			x1 = int(face[0])
			y1 = int(face[1])
			x2 = int(face[2])
			y2 = int(face[3])
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
			face_box = frame[y1:y2, x1:x2]
			eyes = eye_finder.detectMultiScale(face_box)
			for (ex, ey, ew, eh) in eyes:
				ex += x1
				ey += y1
				cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	capture.release()
	cv2.destroyAllWindows()

run()

