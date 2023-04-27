import time

import cv2
import numpy as np
import scipy.ndimage
import pandas as pd
from pathlib import Path

import labc
labc.init_discovery()

import pylsl
outlet = pylsl.StreamOutlet(pylsl.StreamInfo("overlay", channel_count=1))

detector = cv2.QRCodeDetector()

error_buffer_size = 10
min_login_time = 3
login_cooldown = 0.5
debug = 1
use_detect_multi = True
use_decode_curved = False

# Performances:
# single + flat ~ 41
# single + curved ~ 50 and some drops to 24-30
# multi + flat ~ 34
# multi + curved ~ 37

def pick_ampel_color(value):
    if value == 1:
        color = (0, 255, 0)
    elif value == 2:
        color = (0, 255, 255)
    elif value == 3:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color


def put_text(image, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 50)
    fontScale = 1
    thickness = 2
    new_image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return new_image


def put_circle(image, color):
    center_coordinates = (image.shape[1] - 70, 70)
    radius = 30
    thickness = -1
    image = cv2.circle(image, center_coordinates, radius + 5, (0,0,0), thickness)
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image


class RingBuffer:
    def __init__(self, length):
        self._buffer = np.zeros(length, dtype=int)
        self.index = 0

    def append(self, value):
        self._buffer[self.index] = value
        if (self.index + 1)  == len(self._buffer):
            self.index = 0
        else:
            self.index += 1
  
    def mode(self):
        """Returns most frequent Value"""
        vals, counts = np.unique(self._buffer, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

    def vote(self):
        vals = set(self._buffer)
        if debug > 2:
            print("Buffervalues:", vals)
        if 0 in vals:
            vals.remove(0)
        if len(vals) == 1:
            # return vals.pop()
            return next(iter(vals)) # I suspect this to be faster
        else:
            return 0

    @property
    def current(self):
        """Get the last value appended to the buffer."""
        return self._buffer[self.index]


class NutriScoreOverlay:

    def __init__(self, nutri_table):
        nutri_map = {id_: rest for idx, (id_, *rest) in nutri_table.iterrows()}
        nutri_map = {
            id_: (name, score, pick_ampel_color(score))
            for id_, (name, score) in nutri_map.items()
        }
        codings = [
            [(key, i), (i, data)]
            for i, (key, data) in enumerate(nutri_map.items(), 1)
        ]
        #codings = [[('', 0), (0, '')], *codings, (len(codings), 'unknown')]
        encoding, decoding = zip(*codings)
        self.encoding = dict(encoding)
        self.decoding = dict(decoding)
        self.encoding[''] = 0
        self.decoding[0] = ("No QR-Code", 0, (255, 0, 0)) # blue # bgr
        self.decoding[-1] = ("Invalid Item", 0, (255, 0, 0))  # blue

        self.buffer = RingBuffer(error_buffer_size)
        self.login_time = None
        self.logout_time = time.perf_counter()
        self.login_vote = None
        self.frame_counter = 0

    def bbox_size(self, bbox):
        # Use Shoelace formula: https://stackoverflow.com/questions/24467972
        x, y = bbox.T
        size = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if debug >= 2:
            print("bbox size:", size)
        return size


    def bbox_centerness(self, bbox, img_size):
        #img_size = np.array(img_size)
        x, y = bbox.T
        ys, xs = img_size # for some reason y comes first ?
        #x = x / img_size[0]
        #y = y / img_size[1]
        #norm_bbox = np.array([x,y]).T
        #xc, yc = scipy.ndimage.measurements.center_of_mass(norm_bbox)
        #centerness = ((xc - .5)**2 + (yc - .5)**2)**(.5) # 1 - euclidian norm

        xc = abs(np.mean(x / xs) -.5)
        yc = abs(np.mean(y / ys) -.5)

        centerness = xc + yc

        if debug >= 2:
            print("bbox centerness:", centerness)

        return centerness


    def bbox_squareness(self, bbox):
        ...
        return 0
        
    def bbox_score(self, bbox, img_size):
        """
        Calculate the score for a QR-Code.
        Based on the size, orientation and position of its bounding-box.
        """
        size_w = 1
        center_w = 1
        square_w = 1
        return sum([
            size_w * self.bbox_size(bbox),
            center_w * self.bbox_centerness(bbox, img_size),
            square_w * self.bbox_squareness(bbox),
        ])

    def overlay(self, img):
        """
        Add nutri-score overlay to image.

        """
        tic = time.perf_counter()
        if debug: self.frame_counter += 1

        if use_detect_multi:
            detected, points = detector.detectMulti(img)
        else:
            detected, points = detector.detect(img)

        if not detected:
            if debug > 2: print(self.frame_counter, "No qr-code detected.")
            item_idx = 0
        else:

            if use_detect_multi:
                # select the best QR-Code
                #f = lambda bb: self.bbox_score(bb, img.shape[:2])
                i = np.argmax([self.bbox_score(bb, img.shape[:2]) for bb in points])
                bbox = points[i:i+1]
            else:
                bbox = points

            try:
                if use_decode_curved:
                    item_id, _ = detector.decodeCurved(img, bbox)
                else:
                    item_id, _ = detector.decode(img, bbox)
            except cv2.error as err:
                if debug:
                    print(err.code, err.msg)
                return img

            if item_id:
                if item_id in self.encoding:
                    if debug: print(self.frame_counter, "Detected", item_id)
                    item_idx = self.encoding[item_id]
                else:
                    if debug: print(self.frame_counter, "Invalid Item-ID:" , item_id)
                    item_idx = -1

            elif item_id == '':
                if debug: print(self.frame_counter, "Cannot decode QR-Code: empty-string")
                item_idx = self.buffer.current

            else:
                # I dont think this can happen, but just in case
                if debug: raise Exception("Cannot decode QR-Code: None")
                item_idx = self.buffer.current

        self.buffer.append(item_idx)

        if self.login_time:
            if tic - self.login_time > min_login_time:
                # login time over
                voted_item_idx = self.buffer.vote()
                if voted_item_idx != self.login_vote:
                    self.login_time = None
                    labc.network.sound_service.play << "logout"
                    self.logout_time = tic
                else:
                    voted_item_idx = self.login_vote
            else:
                voted_item_idx = self.login_vote
        else:
            voted_item_idx = self.buffer.vote()

        if voted_item_idx or debug:
            name, n_score, color = self.decoding[voted_item_idx]

        if debug:
            img = put_text(img, f"{name}, {n_score}", color)

        if voted_item_idx and (tic - self.logout_time > login_cooldown):
            img = put_circle(img, color)
            if not self.login_time:
                self.login_time = tic
                labc.network.sound_service.play << "login"
                self.login_vote = voted_item_idx

        toc = time.perf_counter()
        #print("Processing time:", 1 / (toc - tic), "Hz")
        outlet.push_sample([toc - tic])
        return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    nutri_table_path = Path(__file__).parent / "nutri_table.csv"
    nutri_table = pd.read_csv(nutri_table_path)
    print(nutri_table)
    nso = NutriScoreOverlay(nutri_table)

    window_name = "Nutri-Score Overlay"

    while True:
        _, img = cap.read()
        img = nso.overlay(img)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1)
        if key in [27, ord("q")]:
            break
    cap.release()
    cv2.destroyAllWindows()
