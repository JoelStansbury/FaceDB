import cv2 #https://pypi.org/project/opencv-python/
import os
from tqdm import tqdm
import loadsave as ls # simple file loading and saving
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    FNAME, BBOX, VEC = ls.load('faces.pkl')
    PROCESSED_IMAGES = ls.load('processed_images.pkl')
except:
    FNAME = []
    BBOX = []
    VEC = []
    PROCESSED_IMAGES = {}

# Face Detector
# load the required trained XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_frontalface_default.xml
# (N*M) dimensional image --> list of rectangles (bounding boxes for each face detected)
face_cascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')


# Face Embedder
# https://cmusatyalab.github.io/openface/models-and-accuracies/
# nn4.small2.v1
# Trained PyTorch Model for extracting Face Embeddings (Unique 128-dim vector)
embedder = cv2.dnn.readNetFromTorch('../data/models/nn4.small2.v1.t7')


# Given an image and a bounding box,
#    crop along the bounding edges and
#    return the image within
def crop_bbox(img, bbox):
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]

# GETEMBEDDINGS desc
    # This function opens the image, detects potential faces in the image,
    #     and computes an embedding for each face.
    # Input:
    #     img_path: Absolute or relative system path (e.g. "../funnypic.jpg")
    # Output:
    #     true_faces: list of bounding boxes for all of the detected faces
    #     embeddings: list of 128-dim face vectors
def getEmbeddings(img_path):
    true_faces = []
    embeddings = []

    img = cv2.imread(img_path)                           # OPEN IMAGE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # CONVERT TO GRAYSCALE
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # DETECT FACES --> RETURN BBOXS

    for bbox in faces:                                    # FOR EACH FACE
        f = crop_bbox(img, bbox)                          #      CROP IT OUT
        (fH, fW) = f.shape[:2]
        if fW < 20 or fH < 20:                            #      SKIP THIS FACE IF IT'S TOO SMALL
            continue

        faceBlob = cv2.dnn.blobFromImage(f, 1.0 / 255,    #      PREPROCESS FACES
            (96, 96), (0, 0, 0), swapRB=True, crop=False) #           (scaling, mean-subtraction, R-B color-swap)

        # Get face embedding
        embedder.setInput(faceBlob)                        #      GIVE FACE TO EMBEDDER
        vec = embedder.forward()                           #      CALCULATE THE EMBEDDING

        true_faces.append(bbox)                            #      ADD FACE BBOX TO true_faces
        embeddings.append(vec)                             #      ADD FACE EMBEDDING TO embeddings
    return true_faces, embeddings


# PROCESS_IMAGES desc.
    # This function runs getEmbeddings() on every image
    #      within the given directory, finding every
    #      face and corresponding embedding
    # Input:
    #      path: Path to the collection of images.
    # Output:
    #      dictionary: Collection of lists where each index describes a particular face found within the database.
    #           'fnames': list of absolute system paths to the parent image
    #           'face_vectors': list of np.arrays() of shape (128,)
    #           'bounding_boxes': list of bboxes
def process_images(path):
    img_path = os.path.abspath(path)
    imgs = []
    filetypes = ['.jpg']                   # DEFINE IMAGE FILE-EXTENSIONS
                                           # SEARCH FOR IMAGES IN path
    for root, dirs, files in os.walk(img_path, topdown=False):
        for name in files:
            if name[-4:].lower() in filetypes:
                try:
                    PROCESSED_IMAGES[os.path.join(root, name)]==0 # CHECK IF IT'S BEEN PROCESSED ALREADY
                except:
                    imgs.append(os.path.join(root, name)) # ADD IT TO THE LIST OF IMAGES TO PROCESS
    if len(imgs) > 0:
        for img in tqdm(imgs):                 # FOR EACH IMAGE
            bboxs, vecs = getEmbeddings(img)   #      FIND ALL FACES WITHIN
            for i, bb in enumerate(bboxs):     #      FOR EACH FACE
                FNAME.append(img)             #           ADD ITS INFO TO THE RESULT
                VEC.append(vecs[i][0])
                BBOX.append(bb)
            PROCESSED_IMAGES[img] = 0
        ls.save((FNAME, BBOX, VEC), 'faces.pkl')
        ls.save(PROCESSED_IMAGES, 'processed_images.pkl')
    else:
        print('No new images to process')


def show_bboxs(fname, bboxs):
    if type(bboxs[0])==np.dtype('int32'):
        bboxs = [bboxs]
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    ax = plt.gca()
    plt.title(fname)
    plt.xticks([])
    plt.yticks([])
    for bbox in bboxs:
        x,y,w,h = bbox
        rect = Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()

def find_similar(img, n=20, debug=False):
    bboxs, vecs = getEmbeddings(img)
    area = 0
    idx = 0
    if len(bboxs)>1:
        for i,bbox in enumerate(bboxs):
            x,y,w,h = bbox
            if w*h > area:
                area = w*h
                idx = i
    if debug:
        show_bboxs(img,bboxs[idx])


    vec = vecs[idx]
    X = np.array(VEC)
    dist = ((X-vec)**2).sum(1)
    idx = np.arange(len(dist))
    zipped = list(zip(dist,idx))
    zipped.sort(key=lambda x:x[0])
    dist, idx = zip(*zipped)

    fnames = []
    bboxs = []
    for i in idx[:n]:
        fnames.append(FNAME[i])
        bboxs.append(BBOX[i])



    return fnames, bboxs
