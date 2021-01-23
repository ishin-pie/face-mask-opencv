import os
import cv2
import uuid
import time

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

from detector import detect_and_predict
from detector import load_model

app = Flask(__name__, static_folder="templates")

# create upload directory if not exist
UPLOAD_FOLDER = 'templates/upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename, extensions=None):
    if extensions is None:
        extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def face_mask_detection(file_name, is_save=False):
    res = []

    # load model
    face_net, mask_net = load_model()

    # read image
    img = cv2.imread(file_name)

    # predict
    (locs, preds) = detect_and_predict(img, face_net, mask_net)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        res_dict = {
            "isMask": True if mask > withoutMask else False,
            "startX": int(startX),
            "startY": int(startY),
            "endX": int(endX),
            "endY": int(endY)
        }
        res.append(res_dict)

        if is_save:
            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    # save result image
    if is_save:
        cv2.imwrite(file_name, img)
    else:
        os.remove(file_name)

    return res, file_name


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
            # save file
            file_name = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + time.strftime("%Y%m%d-%H%M%S") + '.' +
                                     uploaded_file.filename.rsplit('.', 1)[1].lower())
            uploaded_file.save(file_name)

            _, res_file = face_mask_detection(file_name, True)

            return res_file

    return render_template('index.html')


@app.route('/api', methods=['POST'])
def face_mask_api():
    res = []
    uploaded_file = request.files['file']
    if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
        # save file
        file_name = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + time.strftime("%Y%m%d-%H%M%S") + '.' +
                                 uploaded_file.filename.rsplit('.', 1)[1].lower())
        uploaded_file.save(file_name)
        res, _ = face_mask_detection(file_name)

    return jsonify(res)


if __name__ == '__main__':
    app.run()
