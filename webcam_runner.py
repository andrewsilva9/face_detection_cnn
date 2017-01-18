import cv2
import face_detector
import numpy as np

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
			eyes = np.array(eyes)
			# print eyes[:, 1] / float(y2-y1)
			# Ensure we have eyes to do the below operation:
			if len(eyes) > 2:
				# Remove eyes found on the lower half of the face
				eyes = eyes[(eyes[:, 1] / float(y2-y1)) < 0.5]
			for (ex, ey, ew, eh) in eyes:
				# print i, ':', ex / float(x2-x1)
				# x for forward facing: 0.1 - 0.2, 0.6 - 0.7
				# x for left facing: 0.29 - 0.34, 0.7 - 0.75
				# x for right facing: 0.0 - 0.09, 0.34-0.48
				pos_x = ex / float(x2 - x1)
				# pos_y = ey / float(y2 - y1)
				if 0.1 < pos_x < 0.2 or 0.6 < pos_x < 0.7 and len(eyes) > 1:
					print 'center'
				else:
					print 'not center'
				ex += x1
				ey += y1
				# Try to reject mouths...
				# if eh / float(y2-y1) > 0.3 or ew / float(x2-x1) > 0.3:
				# 	continue
				# elif eh / float(y2-y1) < 0.1 or ew / float(x2-x1) < 0.1:
				# 	continue
				# Debugging
				# print eh / float(y2-y1)
				# print ew / float(x2-x1)
				cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	capture.release()
	cv2.destroyAllWindows()

run()

