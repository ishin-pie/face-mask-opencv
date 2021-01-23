import cv2
import numpy as np


def load_model():
    # load our serialized face detector model from disk
    print('[INFO] loading face detector model...')
    face_detector_model_path = './pretrained/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    face_detector_config_path = './pretrained/face_detector/deploy.prototxt'
    face_detector_net = cv2.dnn.readNet(face_detector_model_path, face_detector_config_path)

    # load the face mask detector model from disk
    print('[INFO] loading face mask detector model...')
    face_mask_model_path = './pretrained/face_mask/mask_detector_optmized.pb'
    face_mask_config_path = './pretrained/face_mask/mask_detector_optmized.pbtxt'
    face_mask_net = cv2.dnn.readNet(face_mask_model_path, face_mask_config_path)

    return face_detector_net, face_mask_net


def detect_and_predict(frame, face_net, mask_net, min_confidence=0.5):
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces, locs, preds = [], [], []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = np.asarray(face, dtype='float32')

            # preprocess_input for tensorflow
            face /= 127.5
            face -= 1.
            # end preprocess_input

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        faces = np.swapaxes(faces, 1, 3)
        mask_net.setInput(faces)
        preds = mask_net.forward()

    # return a 2-tuple of the face locations and their corresponding locations
    return locs, preds
