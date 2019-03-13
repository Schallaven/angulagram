#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Copyright (C) 2019 by Sven Kochmann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the  Free Software Foundation,  either version 3  of the License, or
# (at your option) any later version.

# This program  is distributed  in the hope  that it will  be  useful,
# but  WITHOUT  ANY  WARRANTY;  without even  the implied warranty  of
# MERCHANTABILITY  or  FITNESS  FOR  A  PARTICULAR  PURPOSE.  See  the
# GNU General Public License for more details.

# You  should  have  received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# This script extracts an angulagram from a photo (reflectometric
# measurement) using the given parameters. The name 'angureflexin' was
# inspired by chem-name-gen (DOI: 10.5281/zenodo.2578428).

import cv2
import sys
import imutils
import numpy
import math
import matplotlib.pyplot as plt
import os
import csv
import argparse

# Argument setup and parsing to a dictionary
parser = argparse.ArgumentParser(description='Creates an angulagram from a reflectometric input image.')

parser.add_argument('-v', '--version', help='prints version information', action='version',
                    version='angureflexin 1.0 by Sven Kochmann')

parser.add_argument('infile', action='store', nargs='?', type=str)

parser.add_argument('-r', '--rot', action='store', type=float, default=0.0, metavar='N',
                    help='Rotates the image by N degree before extracting the separation zone.')
parser.add_argument('-s', '--sat', action='store', type=int, default=0, metavar='N',
                    help='Defines the saturation increase (can be negative) applied to the zone.')

parser.add_argument('-i', '--inlet', action='store', type=float, default=[0.0, 0.0], metavar=('X', 'Y'), nargs=2,
                    help='Defines the pixel coordinates X, Y of the inlet (after rotation) on the image.')
parser.add_argument('-z', '--zone', action='store', type=float, default=[1000.0, 1000.0], metavar=('W', 'H'), nargs=2,
                    help='Defines the width W and height H of the extracted zone (the reference point is the '
                         'inlet is at the bottom of the rectangle at half width).')
parser.add_argument('-l', '--levels', action='store', type=float, default=[0.0, 255.0], metavar=('L', 'H'), nargs=2,
                    help='Colour levels of the image are changed to the range given by L(ow) and H(igh).')
parser.add_argument('-p', '--preview', action='store', type=float, default=[25.0, 25.0], metavar=('B', 'C'), nargs=2,
                    help='Increases brightness by B and contrast by C on the cropped separation zone for a '
                         'preview image. These values are NOT used for evaluation! Contrast should be in the'
                         'range of -100 to 100.')

parser.add_argument('-m', '--mirror', help='Mirrors the separation zone horizontally before evaluation.', action='store_true')


args = vars(parser.parse_args())

# These are the parameters used to extract the separation zone. The
# modifications are applied in the following order:
# 1. rotation by rotangle
# 2. cropping to rectangle (inlet_x - width/2, inlet_y - height, inlet_x + width/2, inlet_y)
# 3. mirror rectangle content if -m is given
# 3. convert to HSL (Hue, Saturation, Luminance)
# 4. increase saturation by satinc
# 5. convert to gray picture
# 6. remap values to levels (from, to) to cut out background and emphasize streams

# Open file and begin image processing here
image = cv2.imread(args['infile'])
rotated = imutils.rotate_bound(image, args['rot'])
imagecut = rotated[int(args['inlet'][1] - args['zone'][1]):int(args['inlet'][1]),
           int(args['inlet'][0] - args['zone'][0] / 2):int(args['inlet'][0] + args['zone'][0] / 2)]

if args['mirror']:
    imagecut = cv2.flip(imagecut, 1)

hls = cv2.cvtColor(imagecut, cv2.COLOR_BGR2HLS)
zonesat = cv2.split(hls)[2]
zonesat = cv2.add(zonesat, args['sat'])
hls[:, :, 2] = zonesat
gray = cv2.cvtColor(cv2.cvtColor(hls, cv2.COLOR_HLS2BGR), cv2.COLOR_BGR2GRAY)
gray = cv2.subtract(gray, args['levels'][0])
gray = cv2.multiply(gray, 255.0 / (args['levels'][1] - args['levels'][0]))

# This small part is just for creating a preview image (used to show colour figures on Figures in
# papers for example like we did).
imagecut = cv2.convertScaleAbs(imagecut, alpha=(args['preview'][1]+100.0)/100.0, beta=args['preview'][0])
cv2.imwrite(os.path.splitext(args['infile'])[0]+'.preview.png', imagecut)


# Start creating the angulagram here by two steps:
# 1. transfer Cartesian coordinates to polar coordinates using algorithm b from
#    DOI: 10.1021/acs.analchem.8b02186, i.e. for each φ and r in a certain range
#    (e.g. 0-360° for φ) with certain steps (e.g. 1° for φ and 1 pixel for r) the
#    respective x and y coordinates (integral pixels) are calculated; the intensity
#    s(x,y) is then assigned to s(φ,r).
# 2. integrating over r

# Angle step to calculate; here we just calculate ±60° (120° in total)
anglestep = 0.1
anglerange = 120
anglesteps = int(anglerange / anglestep)

# The maximum radian can only be the distance from origin to a corner, e.g. (width-1, 0)
# Note, that this means that there might be points outside the picture
height, width = gray.shape
maxradian = int(math.hypot(width-1, args['inlet'][1]))

# Create empty image width y = maxradian and x = anglesteps
outputimage = numpy.zeros((maxradian, anglesteps), numpy.uint8)

# Calculate the signal for each radian and angle
for r in range(maxradian):
    for angle in range(anglesteps):
        # Calculate x and y
        y = int(round(r * math.cos(math.radians((angle - anglesteps/2.0) * anglestep)), 0))
        x = int(round(r * math.sin(math.radians((angle - anglesteps/2.0) * anglestep)), 0))

        # Center at origin (bottom middle of image)
        x, y = int(width/2) + x, height - y

        # Read out intensity, if x and y are in the picture; also invert intensity here, so that 0 = no signal and
        # 255 = large signal
        intensity = 0
        if 0 <= x < width and 0 <= y < height:
            intensity = 255 - gray[y, x]

        # Write to output image: in this case, we want x of the image to be the angle
        # and y of the image to be the radian; also we want it to be upside down
        outputimage[maxradian - r - 1, angle] = intensity

cv2.imwrite(os.path.splitext(args['infile'])[0] + '.evaluated.png', outputimage)

# Signal as function of angle
Sx = []
Sy = []

# Fill Signal by integrating over r (i.e. x) of the picture
# y = 0 to 360° (from bottom to top)
for x in xrange(anglesteps):
    # Integral = y
    integral = numpy.sum(numpy.transpose(outputimage)[x])
    angle = (x * anglestep - anglerange/2.0)

    # Append to data
    Sx.append(angle)
    Sy.append(integral)

maxintegral = max(Sy)
Sy = [float(float(item)/float(maxintegral)) for item in Sy]

# Plot the Figure
plt.figure()
for l in xrange(360/15 + 1):
    plt.axvline(x=l*15-anglerange, color="lightgrey")
plt.plot(Sx, Sy)
plt.axis([-anglerange/2.0, anglerange/2.0, min(Sy), max(Sy)])
plt.xticks(numpy.arange(-anglerange/2.0, anglerange/2.0, 15))
plt.xlabel('angle (deg)')
plt.ylabel('rel. signal')
plt.savefig(os.path.splitext(args['infile'])[0]+'.angulagram.png')

# Write settings and data for figure into txt file
with open(os.path.splitext(args['infile'])[0]+'.parameters.txt', 'w') as f:
    f.write(args['infile'] + '\r\n')
    f.write("Rotated by %.2f°" % (args['rot']) + '\r\n')
    f.write("Inlet at (%d, %d)" % (args['inlet'][0], args['inlet'][1]) + '\r\n')
    f.write("Zone rectangle width and height (%d, %d)" % (args['zone'][0], args['zone'][1]) + '\r\n')
    f.write("Zone was %s horizontally" % (["mirrored" if args['mirror'] else "not mirrored"][0]) + '\r\n')
    f.write("Saturation increase by %d" % (args['sat']) + '\r\n')
    f.write("Gray levels mapped to %.0f - %.0f" % (args['levels'][0], args['levels'][1]) + '\r\n')

with open(os.path.splitext(args['infile'])[0] + '.angudata.txt', 'w') as f:
    f.write('Angle (deg)\tIntensity (a.u.)\r\n')
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(Sx, Sy))

