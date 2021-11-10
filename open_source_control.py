from imutils import face_utils
import numpy as np

from pynput.mouse import Button, Controller
import time

mouse = Controller()

import imutils
import dlib
import cv2

FPS = 19  # Average FPS measured.

# Thresholds and consecutive frame length for triggering the mouse action.
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 15
WINK_AR_DIFF_THRESH = 1000  # Tune
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 5
EYE_CLOSED_FRAMES = 5

# Initialize the frame counters for each action as well as
# booleans used to indicate if action is performed or not
MOUTH_COUNTER = 0
EYE_CLOSED_FRAME_COUNTER = 0
WINK_COUNTER = 0
INPUT_MODE = False
EYE_CLICK = False
LEFT_WINK = False
RIGHT_WINK = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)

# Colors
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)

# Initialize Dlib's face detector (HOG-based) and then create
# the facial landmark predictor
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Grab the indexes of the facial landmarks for the left and
# right eye, nose and mouth respectively
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(noseStart, noseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
# (mouseStart, mouseEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# (faceStart, faceEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# Video capture
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h


def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear


mode = 'STANDARD'

while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale channels
    t = time.time()
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.putText(frame, "FACE NOT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    # mouth = shape[mouseStart:mouseEnd]
    leftEye = shape[leftEyeStart:leftEyeEnd]
    rightEye = shape[rightEyeStart:rightEyeEnd]
    nose = shape[noseStart:noseEnd]
    # jaw = shape[faceStart:faceEnd]

    # Because I flipped the frame, left is right, right is left.
    temp = leftEye
    leftEye = rightEye
    rightEye = temp

    nose_point = (nose[3, 0], nose[3, 1])

    # Average the mouth aspect ratio together for both eyes
    # mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    cv2.putText(frame, f"FACE CENTER: {nose_point}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
    # cv2.putText(frame, str(rightEAR), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)

    # Compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    # mouthHull = cv2.convexHull(mouth)
    # jawHull = cv2.convexHull(jaw)
    leftEyeHull = cv2.convexHull(leftEye)
    # mouthHull = cv2.convexHull(nose)
    rightEyeHull = cv2.convexHull(rightEye)

    # cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    # cv2.drawContours(frame, [jawHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    # cv2.drawContours(frame, [nose], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

    # for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
    for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
        cv2.circle(frame, (x, y), 2, BLUE_COLOR, -1)

    # Check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    # https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

    # cv2.putText(frame, f'EAR DIFF: {diff_ear}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
    cv2.putText(frame, f'MODE: {mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

    if diff_ear > WINK_AR_DIFF_THRESH:

        if leftEAR < rightEAR:
            # if leftEAR < EYE_AR_THRESH:
            WINK_COUNTER += 1

            if WINK_COUNTER > 2:
                mouse.click(Button.left)

                WINK_COUNTER = 0

        elif leftEAR > rightEAR:
            # if rightEAR < EYE_AR_THRESH:
            WINK_COUNTER += 1

            if WINK_COUNTER > 2:
                mouse.click(Button.right)
                WINK_COUNTER = 0
        else:
            WINK_COUNTER = 0
    else:
        # When two eyes are closed for 1s
        if ear <= EYE_AR_THRESH:
            EYE_CLOSED_FRAME_COUNTER += 1

            if EYE_CLOSED_FRAME_COUNTER >= EYE_CLOSED_FRAMES:
                # SCROLL_MODE = not SCROLL_MODE
                pass

                # # Activates the system! TODO
                # INPUT_MODE = not INPUT_MODE
                # EYE_CLOSED_FRAME_COUNTER = 0
                # ANCHOR_POINT = nose_point

        else:
            EYE_CLOSED_FRAME_COUNTER = 0
            WINK_COUNTER = 0
    if INPUT_MODE:

        cv2.putText(frame, "SYSTEM ACTIVATED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

        x, y = ANCHOR_POINT
        nx, ny = nose_point

        if not SCROLL_MODE:
            color = YELLOW_COLOR
        else:
            color = WHITE_COLOR
        cv2.line(frame, ANCHOR_POINT, nose_point, color, 5)

        # Displacement Equation!!!!!
        dx = nx - x
        dy = ny - y

        if abs(dx) < 20:
            dx = 0
        if abs(dy) < 20:
            dy = 0

        if not SCROLL_MODE:
            mouse.move(0.5 * dx, 0.5 * dy)

        if SCROLL_MODE and abs(dy) >= 20:
            dy = - dy
            if dy > 0:
                mouse.scroll(0, 0.1 * dy)
            else:
                mouse.scroll(0, 0.1 * dy)

    else:
        cv2.putText(frame, "SYSTEM DEACTIVATED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

    cv2.putText(frame, f'FPS: {round((1 / (time.time() - t)) * 10) / 10}', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                RED_COLOR, 2)

    # cv2.putText(frame, "MAR: {:.2f}".format(mar), (500, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    # cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (460, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    # cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (460, 130),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
    # cv2.putText(frame, "Diff EAR: {:.2f}".format(np.abs(leftEAR - rightEAR)), (460, 80),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press Esc to exit
    if key == 27:
        break

    if key == 115:
        INPUT_MODE = not INPUT_MODE
        ANCHOR_POINT = nose_point

    if key == 100:
        INPUT_MODE = not INPUT_MODE
        SCROLL_MODE = not SCROLL_MODE
        ANCHOR_POINT = nose_point
        if SCROLL_MODE:
            mode = 'SCROLLING'
        else:
            mode = 'STANDARD'

cv2.destroyAllWindows()
vid.release()
