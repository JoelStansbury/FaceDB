# FaceDB
Find and group faces in a database of images

## Project Outline
### What
 * This _hopefully_ will be a program which, given a database of images, will find all faces within, create a _unique_ identifier for each individual, allow the user to (select individuals within a picture, assign a name to them, and search for other images containing that person)
 * If, for example, you want to find the image of you and Obama at Comic-Con last year you would provide a picture of you, and a picture of Obama and query the dataset for images containing both you and Obama. The goal is to reduce the number of images you need to sift through to find the one you want
### Why
 * We have a large set of images from various events. We want the ability to search for all images containing a particular person.
### How
 * I intend to use [OpenCv](https://opencv.org/) for all face detection and labeling. 
 * The user interface will probably be browser-based, though I'm considering using PyQT.

 
 ### References
 * [[1]](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/) Tutorial for face recognition with OpenCv
