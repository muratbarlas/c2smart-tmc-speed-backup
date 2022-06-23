import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry import LineString
import shapely


def GetAngleOfLineBetweenTwoPoints(p1, p2):
    xDiff = p2.x - p1.x
    yDiff = p2.y - p1.y
    return math.degrees(math.atan2(yDiff, xDiff))

def displayCountCV2Box(lineArr, img,scale=1):
    xval = 3
    yval = 25
    endtuple = (160, (len(lineArr)*25)+5)
    cv2.rectangle(img,(0,0),endtuple,(255,0,0),-1)
    for i in range(len(lineArr)):
        text = str(lineArr[i].id)+":Vehicles " + str(len(lineArr[i].detected))
        cv2.putText(img, text, (xval, yval), cv2.FONT_HERSHEY_SIMPLEX, scale*.75, (0, 255, 0), 2)
        yval+=25

def draw_rotated_text(image, angle, xy, text, fill, *args, **kwargs):
    """ Draw text at an angle into an image, takes the same arguments
        as Image.text() except for:

    :param image: Image to write text into
    :param angle: Angle to write text at
    """
    # get the size of our image
    width, height = image.size
    max_dim = max(width, height)

    # build a transparency mask large enough to hold the text
    mask_size = (max_dim * 2, max_dim * 2)
    mask = Image.new('L', mask_size, 0)

    # add text to mask
    draw = ImageDraw.Draw(mask)
    draw.text((max_dim, max_dim), text, 255, *args, **kwargs)

    if angle % 90 == 0:
        # rotate by multiple of 90 deg is easier
        rotated_mask = mask.rotate(angle)
    else:
        # rotate an an enlarged mask to minimize jaggies
        bigger_mask = mask.resize((max_dim * 8, max_dim * 8),
                                  resample=Image.BICUBIC)
        rotated_mask = bigger_mask.rotate(angle).resize(
            mask_size, resample=Image.LANCZOS)

    # crop the mask to match image
    mask_xy = (max_dim - xy[0], max_dim - xy[1])
    b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
    mask = rotated_mask.crop(b_box)

    # paste the appropriate color, with the text transparency mask
    color_image = Image.new('RGBA', image.size, fill)
    image.paste(color_image, mask)


class detectShape:
    def __init__(self, id, dims):
        if not all(isinstance(i, tuple) for i in dims):
            raise ValueError("Dimensions should be a list of tuples")
        self.id = id
        self.detected = []
        self.dims = dims


class detectLine(detectShape):
    def __init__(self, id, dims):
        super().__init__(id, dims)
        self.line = LineString(dims)

    def checkObjCrossesPoly(self, id, vehiclePoly: Polygon):
        inBox = False

        if vehiclePoly.crosses(self.line):
            inBox = True
        if (id not in self.detected) and inBox:
            self.detected.append(id)
            return True
        else:
            return False

    def plot(self, img):
        cv2.line(img, (int(self.line.bounds[0]), int(self.line.bounds[1])),
                 (int(self.line.bounds[2]), int(self.line.bounds[3])), (17, 9, 230), 2)

    def displayCountPIL(self, img, orientation='parallel', scale=1.2):
        count = str(len(self.detected))
        text = "Vehicles " + str(len(self.detected))
        textsize = int(16 * scale)
        if orientation == 'parallel':
            xval = int(self.line.bounds[0])  # + ((self.line.bounds[2] - self.line.bounds
            # [0]) /2))  # xval should be in the middle of the leftLine
            yval = int(self.line.bounds[1]) + 5
            coords = list(self.line.coords)
            yDiff = coords[1][1] - coords[0][1]
            xDiff = coords[1][0] - coords[0][0]
            degree = math.degrees(math.atan2(yDiff, xDiff))

            img2 = Image.fromarray(img)
            draw_rotated_text(img2, -degree, (xval, yval), text, (255, 255, 255, 255),
                              font=ImageFont.truetype("Monaco.ttf", textsize))
            img = np.asarray(img2)
            img2.save("fuck.jpg", 'JPEG')
            # cv2.putText(img, text, (xval,yval), cv2.FONT_HERSHEY_SIMPLEX, scale,
            # (0,255,0), 2)
            # cv2.putText(img, count, (xval,yval+ 20), cv2.FONT_HERSHEY_SIMPLEX, scale,
            # (0,255,0), 2)

        return img

    def displayCountCV2(self, img, orientation="parallel", scale=1):
        count = str(len(self.detected))
        text = "Vehicles " + str(len(self.detected))
        if orientation == 'parallel':
            xval = int((self.line.bounds[0])  + ((self.line.bounds[2] - self.line.bounds[0]) /2))# xval should be in the middle of the leftLine
            yval = int(self.line.bounds[1]) + 5

        if orientation == "vertical":
            xval = int((self.line.bounds[0]) + ((self.line.bounds[2] - self.line.bounds[0]) / 2)) -60  # xval should be in the middle of the leftLine
            yval = int(self.line.bounds[1]) + 5
        cv2.putText(img, text, (xval, yval), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
        cv2.putText(img, count, (xval, yval + 20), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)

    def displayID(self,img,scale=1):
        xval=int(self.line.bounds[0])
        yval=int(self.line.bounds[1])
        cv2.putText(img, str(self.id), (xval,yval), cv2.FONT_HERSHEY_SIMPLEX, scale*1.15, (255, 0, 0),3)
        cv2.putText(img, str(self.id), (xval,yval), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0),3)


class detectPoly(detectShape):
    def __init__(self, id, dims):
        super().__init__(id, dims)
        self.poly = Polygon(dims)

    def IOU(self, polyOther: Polygon):
        poly1, poly2 = self.poly, polyOther
        return poly2.intersection(poly1).area / poly2.union(poly1).area

    def plot(self, img):
        pts = []
        for i in self.dims:
            pts.append(list(i))
        arrPts = np.array(pts)
        cv2.polylines(img, [arrPts], isClosed=True, color=(255, 0, 0, 3), thickness=1, lineType=4)


def convertToPoly(objDims):
    box = objDims
    b1_x1, b1_y1, b1_x2, b1_y2 = box[0], box[1], box[2], box[3]
    poly = Polygon([(b1_x1, b1_y1), (b1_x2, b1_y1), (b1_x2, b1_y2), (b1_x1, b1_y2), (b1_x1, b1_y1)])
    return poly
