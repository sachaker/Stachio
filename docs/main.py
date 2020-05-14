from helpers import *
import imutils
from imutils.video import WebcamVideoStream
import time
import cv2
import dlib


filename = '../data/mustache.png' # http://clipart-library.com/free/mustache-transparent-background.html
mustache = cv2.imread(filename,-1) # load mustache png
stacheW = mustache.shape[1]

detector = dlib.get_frontal_face_detector() # face detector
predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat') # specific landmark (mouth)

vs = WebcamVideoStream(src=0).start() # start stream
time.sleep(1) # let it warm up

original = vs.read() # read from video stream

shrinkSize = 0.65
newW = int(round(shrinkSize*original.shape[1]))
newH = int(round(shrinkSize*original.shape[0]))

draw = True

while True:
    frame = cv2.resize(vs.read(), (newW, newH))  # shrink VS frame

    if draw:
        try:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)  # detect speaker nose (enumerate from left to right)

            for rect in rects:
                nose = getNose(frame, rect, predictor) # identify nose and pull coords
                noseW = getRange(nose[:,:,0]) # get width of nose
                noseH = getRange(nose[:,:,1])
                noseCentroid = getCentroid(nose[:,:,0], nose[:,:,1])

                #angleRad = math.atan2(noseH, noseW)
                #angle = -1*math.degrees(angleRad) # get angle of face (via base of nose)
                angle = 0

                scale = 8*noseW/stacheW # stache needs to scale in accordance w/ nose
                overlay = imutils.rotate_bound(cv2.resize(mustache, (0, 0), fx=scale, fy=scale), angle) # scale and rotate
                xShift = int(round(overlay.shape[1]/2)) # stache img starts top left, so center it

                frame = overlay_with_transparency(frame, overlay, noseCentroid[0]-xShift, noseCentroid[1]-10) # draw stache

        except:
            pass

    cv2.imshow('', frame)
    if cv2.waitKey(1) == ord('a'):  # if user presses 'a'...
        draw = draw == False  # flip (and turn off mustache)