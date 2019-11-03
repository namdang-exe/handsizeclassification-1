import time

import sys
import cv2 as cv
import imageio
import numpy as np
import tensorflow as tf

loaded = tf.saved_model.load('./save2/')
model = loaded.signatures['serving_default']
full_image_size = [720, 1280]
input_image_size = [int(full_image_size[0] / 2), int(full_image_size[1] / 2), 3]


class fix(object):
    def __init__(self):
        pass

    def load_params(self):
        self.mtx = np.load('mtx.npy')
        self.dist = np.load('dist.npy')

    def undistort(self, image):
        h, w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        return cv.undistort(image, self.mtx, self.dist, None, newcameramtx)


fixer = fix()
fixer.load_params()


def tf_infer(model, image):
    im = fixer.undistort(image)
    im = tf.constant(im)
    im = tf.expand_dims(im, axis=0)
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = tf.image.resize(im, [input_image_size[0], input_image_size[1]])
    pred = model(im)['output_1'].numpy()[0]
    return pred


cv.namedWindow('preview', cv.WINDOW_NORMAL)
cv.resizeWindow('preview', full_image_size[1], full_image_size[0])
reader = imageio.get_reader('<video2>')
num_frames = 0

while True:
    num_frames += 1
    time_start = time.time()
    frame = reader.get_next_data()
    pred = tf_infer(model, frame)
    time_stop = time.time()
    pred = np.tile(pred, [1, 1, 3])
    frame = cv.resize(frame, (pred.shape[1], pred.shape[0]))[..., ::-1].copy() / 255.
    combined = np.concatenate((frame, pred), axis=1)
    cv.imshow('preview', combined)
    key = cv.waitKey(1)
    if key == ord('c'):
        cv.imwrite('hand.jpg', combined)
        break

f = cv.imread('hand.jpg')
cv.namedWindow('TEST')
cv.imshow('TEST', f)
cv.waitKey(0)

cv.destroyAllWindows()

if num_frames % 100 == 0:
    fps = 1 / (time_stop - time_start)
    sys.stdout.write(f'\r{int(fps)} FPS    ')
    sys.stdout.flush()
