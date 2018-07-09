#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2018 by Sven Kochmann
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   Description:
#   Script to transform an image from cartesian to polar coordinates
#   Input image: input.png; origin is assumed at x=0, y=(height/2)-1  (0-based)
#   Output image: input.polar.1.png; x-axis = radian, y-axis = angle
#
#   This script implements transforming algorithm a: for each pair of x and y (integral pixels) in the source image
#   the respective angle φ and radian r are calculated; the intensity s(x,y) is then assigned to s(φ,r).


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Filename
filename = 'input.png'

# This is the smallest angle step, in degree
anglestepdeg = 1.0

# Parameter given?
# Filename
if len(sys.argv) > 1:
    filename = sys.argv[1]

# Angle steps
if len(sys.argv) > 2:
    print(sys.argv[2], float(sys.argv[2]))
    anglestepdeg = float(sys.argv[2])

if anglestepdeg < 0.01:
    anglestepdeg = 0.01

# Read image
image = cv2.imread(filename)

cv2.imshow("Original image", image)

# Convert to gray values
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Black and white image", image)

# Get shape
height, width = image.shape

# Origin
origin = (0, int(height/2)-1)

print("Shape of image is %d×%d. The origin is at x=%d, y=%d." % (width, height, origin[0], origin[1]))

# The maximum radian can only be the distance from origin to a corner, e.g. (width-1, 0)
# Note, that this means that there might be points outside the picture
maxradian = int(math.hypot(width-1, origin[1]))

print("Maximum radian: ", maxradian)

# Steps
anglesteps = int(360.0 / anglestepdeg)

print("Angle from -180° to +180° in %.2f steps results in %d steps." % (anglestepdeg, anglesteps))

# Create empty image width x = maxradian and y = anglesteps
outputimage = np.zeros((anglesteps, maxradian), np.uint8)

# Calculate the image for each x and y
for x in range(width):
    for y in range(height):

        # Account for origin
        ox, oy = x - origin[0], origin[1] - y

        # Radian
        r = int(round(math.hypot(ox, oy), 0))

        # Zero radian? Continue
        if r == 0:
            continue

        # Angle
        angle = int(round(math.degrees(math.asin(float(oy) / float(r))), 0)) + int(180.0/anglestepdeg)

        # Read out intensity, if r and angle are in the picture; also invert intensity here, so that 0 = no signal and
        # 255 = large signal
        intensity = intensity = 255-image[y, x]

        # Write to output image; origin is bottom left corner
        if 0 <= r < maxradian and 0 <= angle < int(360/anglestepdeg):
            outputimage[angle, r] = intensity

        #print("Intensity at %d, %d = %d. Projecting to %d and %d." % (x, y, intensity, r, angle))
        #break

# Show image
cv2.imshow("Output image", outputimage)
cv2.waitKey(0)

# Save output file
cv2.imwrite(os.path.splitext(filename)[0]+'.polar.1.png', outputimage)




