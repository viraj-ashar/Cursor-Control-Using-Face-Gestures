import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import pyautogui as pag

# Required Threshold And Frame Length to trigger the action  of mouse
mouthArThresh = 0.6
mouthArConsecutiveFrames = 15
eyeArThresh = 0.19
eyeArConsecutiveFrames = 15
winkArDiffThresh = 0.04
winkArCloseThresh = 0.19
winkConsecutiveFrames = 1

# Initializing counters to count the action and booleans to check whether to
# the action is executing or not.t
mouthCounter = eyeCounter = winkCounter = 0
inputMode = eyeClick = leftWink = rightWink = scrollMode = False
anchorPoint = (0,0)

whiteColor = (255,255,255)
yellowColor = (0,255,255)
redColor = (0,0,255)
greenColor = (0,255,0)
blueColor = (255,0,0)
blackColor = (0,0,0)

# dlib's facedetector and facial landmarks predictor
shapePredictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Assigning indexex for left and right eye, nose and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS['nose']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

# Capturing Video from camera
cap = cv2.VideoCapture(0)
resolutionWidth = 1366
resolutionHeight = 768
camWidth = 640
camHeight = 480
unitWidth = resolutionWidth/camWidth
unitHeight = resolutionHeight/camHeight

def eyeAspectRatio(eye):
    # Calculating the euclidean distance between the 2 set of vertical eye landmarks
    # and 1 set ofhorizantal landmark
    a = np.linalg.norm(eye[1]-eye[5])
    b = np.linalg.norm(eye[2] - eye[4])

    c = np.linalg.norm(eye[0]-eye[3])

    # Calculate the eye aspect ratio(ear)
    ear = (a+b)/(3*c)
    return ear

def mouthAspectRatio(mouth):
    # Calculating the euclidean distance between the 3 set of vertical eye landmarks
    # and 1 set ofhorizantal landmark
    a = np.linalg.norm(mouth[13] - mouth[19])
    b = np.linalg.norm(mouth[14] - mouth[18])
    c = np.linalg.norm(mouth[15] - mouth[17])

    d = np.linalg.norm(mouth[12] - mouth[16])

    mar = (a+b+c)/(d)
    return mar


def direction(nosePoint, anchorPoint, width, height, multiple = 1):
    noseX, noseY = nosePoint
    anchorX, anchorY = anchorPoint

    if noseX > anchorX + multiple*width:
        return 'right'
    elif noseX < anchorX - multiple*width:
        return 'left'

    if noseY > anchorY + multiple*height:
        return 'down'
    elif noseY < anchorY - multiple*height:
        return 'up'

    return '-'

while True:
    # capturing the frame from cap(i.e camera) and then resizing it and converting
    # to grayscale
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=camWidth, height=camHeight)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, "Press Enter to exit", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)

    # face detector from a grayscale Image
    # it is one kind of object of all the faces recognized
    rects = detector(gray, 0)

    # Check if any face detected or not
    if len(rects) > 0:
        # assigning the first face which was detected in the rect
        rect = rects[0]
    else:
        cv2.imshow('Mouse Tracker Using Face Gestures.', frame)
        continue

    # Detecting all the landmarks of the face region in rect, and then converting
    # it into (x,y) coordinates
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Fetching the coordinates of both the eyes, and using it to compute
    # the eye aspect ratio of both the eyes
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]

    # As we flipped the frame earlier, left is right and right is left.
    leftEye, rightEye = rightEye, leftEye

    # Calculate the mouth aspect ratio and eye aspect ratio for both eyes
    MAR = mouthAspectRatio(mouth)
    leftEAR = eyeAspectRatio(leftEye)
    rightEAR = eyeAspectRatio(rightEye)
    EAR = (leftEAR + rightEAR) / 2.0
    diffEAR = np.abs(leftEAR-rightEAR)

    # giving x,y coordinate for nose
    nosePoint = (nose[3, 0], nose[3, 1])

    # Calculate the convex hull(i.e joins all the landmark points of specified contours/area)
    # of left and right eye, and then visualize both the eyes
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, yellowColor, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, yellowColor, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, yellowColor, 1)

    # Marking the green dots
    for (x,y) in np.concatenate((mouth, leftEye, rightEye), axis = 0):
        cv2.circle(frame, (x,y), 2, greenColor, -1)

    # Check if the eye aspect ratio is below the blink threshold or not
    # If it is increment the blink frame counter by 1
    if diffEAR > winkArDiffThresh:
        if leftEAR < rightEAR and leftEAR < eyeArThresh:
                winkCounter += 1
                text = '1st' + str(winkCounter)
                cv2.putText(frame, text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)
                if winkCounter > winkConsecutiveFrames:
                    pag.click(button='left')
                    winkCounter = 0

        elif leftEAR > rightEAR and rightEAR < eyeArThresh:
                winkCounter += 1
                text = '2nd' + str(winkCounter)
                cv2.putText(frame, text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)
                if winkCounter > winkConsecutiveFrames:
                    pag.click(button='right')
                    winkCounter = 0

        else:
            winkCounter = 0

    else:
        if EAR <= eyeArThresh:
            eyeCounter += 1
            if eyeCounter > eyeArConsecutiveFrames:
                scrollMode = not scrollMode
                eyeCounter = 0

        else:
            eyeCounter = 0
            winkCounter = 0

    if MAR > mouthArThresh:
        mouthCounter += 1
        if mouthCounter >= mouthArConsecutiveFrames:
            inputMode = not inputMode
            mouthCounter = 0
            anchorPoint = nosePoint

    else:
        mouthCounter = 0

    if inputMode:
        cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)
        x,y = anchorPoint
        nx, ny = nosePoint
        width, height = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x-width, y-height), (x+width, y+height), greenColor, 2)
        cv2.line(frame, anchorPoint, nosePoint, blueColor, 2)

        dir = direction(nosePoint, anchorPoint, width, height)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor, 2)
        drag = 18
        if dir == 'right':
            pag.moveRel(drag, 0)
        elif dir == 'left':
            pag.moveRel(-drag, 0)
        elif dir == 'up':
            if scrollMode:
                pag.scroll(40)
            else:
                pag.moveRel(0, -drag)
        elif dir == 'down':
            if scrollMode:
                pag.scroll(-40)
            else:
                pag.moveRel(0, drag)

    if scrollMode:
        cv2.putText(frame, 'Scroll Mode Is On!', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, redColor,2)

    cv2.imshow('Mouse Tracker Using Face Gestures.', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()