<p>Image morphing can be defined as a technique which constructs the sequence of images
depicting the transition between them. The method that is used in this project involves using
Delaunay Triangulation and Affine transformation.</p>
<p>Firstly the images are divided into several parts by selecting different points on it. These points
on the image are called control points. The control points are used in order to apply the Delaunay
triangulation as well as the Affine transformation on the images on them. The details of the
methods are explained in the Algorithm section.</p>
<p>Morphing is mainly employed in the field of animations and special effects. In the present day
there exist many software like after effects, nuke etc. These software can also be used by people
who don’t know coding.</p>
# Instructions to use the code

This file contains the steps on how to execute the file.

<strong>Step-1</strong> Open the command line or terminal and enter the following -

                    python3 Morphing.py img1.jpg img2.jpg

here img1 refers to the source image and img2 refers to the destination image.

<strong>Step-2</strong> Enter the control points on img1 using mouse click and press escape after entering all points. Do the same for img2 but the order of points should remain same.

After doing so the system will display as well as save the triangulated images.

<strong>Step-3</strong> Enter the number of intermediate images you want to see (This number should exclude the source and destination image as they are already taken care of).

The code will take some time to create and save the desired number of intermediates. We have directly saved the images to save the time.

<strong>If you find any difficulty in the steps above you can refer to the video attached of the same.</strong>

<b>Note - The code is explained in the report itself.</b>
