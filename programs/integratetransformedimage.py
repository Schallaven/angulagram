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
#   Takes an image, integrates over its x (which is r) and plots intensity as function of y (which is the angle)

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import csv

# Latex settings for matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# Filename
filename = 'input.png'
scaling = True

# Parameter given?
if len(sys.argv) > 1:
    filename = sys.argv[1]

if len(sys.argv) > 2:
    scaling = not sys.argv[2] == "noscaling"

# Read image
image = cv2.imread(filename)

# Convert to gray values
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get shape
height, width = image.shape

# Signal as function of angle
Sx = []
Sy = []

# Step in degree
anglestepdeg = 360.0 / float(height)

# Fill Signal by integrating over r (i.e. x) of the picture
# y = 0 to 360° (from bottom to top)
for y in xrange(height):
    # Integral = y
    integral = np.sum(image[y])

    # Angle = -180 to 180 (y = 0 to 360)
    angle = y * anglestepdeg - 180.0

    # Append to data
    Sx.append(angle)
    Sy.append(integral)

# Remapping of type
if scaling:
    maxintegral = max(Sy)
    Sy = [float(float(item)/float(maxintegral)) for item in Sy]

# Show maximum
else:
    print("Maximum intensity: ", max(Sy))

# Convert to colour values
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Create empty image
imagepresent = np.zeros((height, width, 3), np.uint8)

# Plot the Figure
plt.figure()

# Add lines every 15°
for l in xrange(360/15 + 1):
    cv2.line(imagepresent, (0, int(15.0/anglestepdeg)), (width, int(15.0/anglestepdeg)),
             (0, 215, 255), [3 if l == 12 else 1][0])
    plt.axvline(x=l*15-180, color="lightgrey")

# Add the input image
imagepresent = cv2.add(imagepresent, image)

# Show the image (resize to have a height of 540, if necessary
if height > 540:
    factor = float(height) / 540.0
    imagepresent = cv2.resize(imagepresent, (540, int(width/factor)))

cv2.imshow("", imagepresent)


plt.plot(Sx, Sy)
plt.axis([-90, 90, 0, max(Sy)])
plt.xticks(np.arange(-90, 105, 15))
plt.rc('text.latex', preamble=r'\usepackage{arcs}'
                              r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}')
plt.xlabel('$\\boldsymbol{\\varphi}$ (deg)', fontsize=18)
plt.ylabel('$\\boldsymbol{\\overset{\\scriptscriptstyle\\frown}{S}(\\varphi)}$ (a.u.)', fontsize=18)
plt.savefig(os.path.splitext(filename)[0]+'.integrated.png')
plt.show()

# Write data for figure into txt file
with open(os.path.splitext(filename)[0]+'.integrated.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(Sx, Sy))


