# USAGE
# python blink.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python blink.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]



EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
angle = 0.00
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
    return ear
 
def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = get_head_pose(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
		angle=euler_angle[2,0]

            cv2.imshow("Head", frame)
		
	    if(angle >=20):
		print "Detecting Blink..........."
		break
	    if(euler_angle[2,0] <=-20):
		print "delete character"
		break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)


if __name__ == '__main__':
    main()
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(args["video"]).start()
    fileStream = True
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
    time.sleep(1.0)

    # loop over frames from the video stream
    while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
	    break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame2 = vs.read()
	frame2 = imutils.resize(frame2, width=450)
	gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
	    # determine the facial landmarks for the face region, then
	    # convert the facial landmark (x, y)-coordinates to a NumPy
	    # array
	    shape = predictor(gray, rect)
	    shape = face_utils.shape_to_np(shape)

	    # extract the left and right eye coordinates, then use the
	    # coordinates to compute the eye aspect ratio for both eyes
	    leftEye = shape[lStart:lEnd]
	    rightEye = shape[rStart:rEnd]
	    leftEAR = eye_aspect_ratio(leftEye)
	    rightEAR = eye_aspect_ratio(rightEye)

	    # average the eye aspect ratio together for both eyes
	    ear = (leftEAR + rightEAR) / 2.0

	    # compute the convex hull for the left and right eye, then
	    # visualize each of the eyes
	    leftEyeHull = cv2.convexHull(leftEye)
	    rightEyeHull = cv2.convexHull(rightEye)
	    cv2.drawContours(frame2, [leftEyeHull], -1, (0, 255, 0), 1)
	    cv2.drawContours(frame2, [rightEyeHull], -1, (0, 255, 0), 1)

	    # check to see if the eye aspect ratio is below the blink
	    # threshold, and if so, increment the blink frame counter
	    if ear < EYE_AR_THRESH:
		COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
	    else:
		# if the eyes were closed for a sufficient number of
		# then increment the total number of blinks
		if COUNTER >= EYE_AR_CONSEC_FRAMES:
		    TOTAL += 1

		# reset the eye frame counter
		COUNTER = 0

	    # draw the total number of blinks on the frame along with
	    # the computed eye aspect ratio for the frame
	    cv2.putText(frame2, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    cv2.putText(frame2, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	 
	# show the frame
	cv2.imshow("Blink", frame2)
	key = cv2.waitKey(1) & 0xFF
	 
	# if the `q` key was pressed, break from the loop
	if(angle <= -20):
	    vs.stop()
	if key == ord("q"):
	    cv2.destroyAllWindows()
	    vs.stop()
	    break

