import cv2 #https://pypi.org/project/opencv-python/
import os

# Face Detector
# load the required trained XML classifiers
# https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_frontalface_default.xml
# (N*M) dimensional image --> list of rectangles (bounding boxes for each face detected)
face_cascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')

# # https://github.com/opencv/opencv/tree/master/data/haarcascades/haarcascade_eye.xml
# # Trained XML file for detecting eyes
# eye_cascade = cv2.CascadeClassifier('../data/models/haarcascade_eye.xml')

# Face Embedder
# https://cmusatyalab.github.io/openface/models-and-accuracies/
# nn4.small2.v1
# Trained PyTorch Model for extracting Face Embeddings (Unique 128-dim vector)
embedder = cv2.dnn.readNetFromTorch('../data/models/nn4.small2.v1.t7')


def crop_bbox(img, bbox):
    x,y,w,h = bbox
    return img[y:y+h, x:x+w]


def getEmbeddings(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    true_faces = []
    embeddings = []

    for bbox in faces:
        f = crop_bbox(img, bbox)
        (fH, fW) = f.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        # Pre-process images (scaling, mean-subtraction, R-B color-swap)
        faceBlob = cv2.dnn.blobFromImage(f, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)

        # Get face embedding
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        true_faces.append(bbox)
        embeddings.append(vec)

    return true_faces, embeddings


def process_images(path):
    img_path = os.path.abspath(path)
    imgs = []
    filetypes = ['.jpg']

    for root, dirs, files in os.walk(img_path, topdown=False):
        for name in files:
            if name[-4:].lower() in filetypes:
                imgs.append(os.path.join(root, name))
    source = []
    embedding = []
    bbox = []
    for img in imgs:
        bboxs, vecs = getEmbeddings(img)
        for i, bb in enumerate(bboxs):
            source.append(img)
            embedding.append(vecs[i][0])
            bbox.append(bb)

    return {'fnames':source,'face_vectors':embedding,'bounding_boxes':bbox}
