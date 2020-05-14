from imutils import face_utils
import cv2
import numpy as np


def getNose(gray, rects, predictor):

    (mStart, mEnd) = (32, 36)  # face landmarks for nose

    shape = predictor(gray, rects)
    shape = face_utils.shape_to_np(shape)

    nose = shape[mStart:mEnd]
    coords = cv2.convexHull(nose)

    return coords


def getRange(X):
    return (max(X)-min(X))


def getCentroid(X, Y):
    return (int(round(np.mean(X))), int(round(np.mean(Y))))


# https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
def overlay_with_transparency(background,overlay,X,Y):

    # requires BRG images

    # be sure to open images using the -1 flag (i.e. unchanged)
    # otherwise you lose the alpha channel

    # background image dimensions
    bh,bw = background.shape[:2]

    # overlay dimensions
    oh,ow = overlay.shape[:2]

    # overlay start (top-left) is greater than background
    if X >= bw or Y >= bh:
        return background

    # overlay end is less than background
    if ow + X <= 0 or oh + Y <= 0:
        return background

    # resize overlay (remove part outside background)
    if X + ow > bw:
        ow = bw - X
        overlay = overlay[:,:ow] # cut 2nd layer (cols) lengths
    if Y + oh > bh:
        oh = bh - Y
        overlay = overlay[:oh] # cut first layer (rows) length

    # no alpha channel (opaque)
    if overlay.shape[2] == 3:
        background[Y:Y+oh,X:X+ow] = overlay

    # alpha channel (transparent)
    else:

        # make a mask of overlay alpha values as decimals of max value 255
        mask = overlay[...,3:]/255.0

        # the alpha channel formula: image = alpha*foreground + (1-alpha)*background
        background[Y:Y+oh,X:X+ow] = mask*overlay[...,:3] + (1-mask)*background[Y:Y+oh,X:X+ow]

    # done
    return background