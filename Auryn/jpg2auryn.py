#!/usr/bin/python

# Copyright 2014 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# File :
#
import matplotlib.image as mpimg
import numpy as np
import sys

filename=sys.argv[1]
img=mpimg.imread(filename)
img = img.ravel()
img = np.array(img,dtype=float)
img /= 255.0
thr = 0.5

print "# Pattern loaded from file ", filename
for c in np.argwhere(img):
    i = c[0]
    val = img[i]
    if (val>thr):
        print "%i %f"%(i,1.*val)

