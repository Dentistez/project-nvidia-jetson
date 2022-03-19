import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import time

class_names = ['laying', 'sit_foor', 'toilet_siting', 'toilet_squat', 'walk']

vdo_path = 'D:/Projectz/final-project/src/clone/vdo/test/sleep/'
vdo_dir_name = 'newvdo'
vdo_dir_name = str(input("Enter VDO_DIR_NAME <= "))
vdo_path += vdo_dir_name + '/'

depth_stream = cv2.VideoCapture(vdo_path + "depth.avi")
Depth_model = tf.keras.models.load_model(
    'D:/Projectz/traindata/Model_Appications/models_InceptionV3/Depth_BG.pb')
modelsize = (150, 150)

while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')


def load_model(model_name):
    model = tf.saved_model.load(str(model_name))
    return model


PATH_TO_LABELS = 'D:/TFODCourse-main/TFODCourse-main/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

model_name = 'D:/TFODCourse-main/TFODCourse-main/Tensorflow/workspace/models/my_ssd_mobnet/export/saved_model'
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, frame):
    # take the frame from webcam feed and convert that to array
    image_np = np.array(frame)
    # Actual detection.

    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=.8,
        line_thickness=5)
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']

    im_height, im_width = image_np.shape[:2]

    box_pack = []
    score_pack = []

    for i in range(len(scores)):
        if scores[i] > 0.7:
            score = scores[i]
            [ymin, xmin, ymax, xmax] = boxes[i]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            left, right, top, bottom = (
                int(left), int(right), int(top), int(bottom))

            box_pack.append((left, right, top, bottom))
            score_pack.append(score)

    return(image_np, box_pack, score_pack)

    # Now we open the webcam and start detecting objects


while True:
    # Capture frame-by-frame
    re, frame = depth_stream.read()

    # Decode Depth
    # 8upperbits data in ch1 / 8lowerbits data in ch2 / ignore ch3
    depth_ch1, depth_ch2, _ = cv2.split(frame)
    # 8upperbits data -> convert uint16 and shift left << 8
    decoded_depth = np.left_shift(np.uint16(depth_ch1.copy()), 8)
    decoded_depth = np.bitwise_or(decoded_depth, np.uint16(
        depth_ch2.copy()))  # bitwise or with 8lowerbits

    # Adjusted for display
    decoded_depth *= 10

    decoded_depth8bit = np.uint8(decoded_depth/257)
    decoded_depth8bit_3ch = cv2.merge(
        [decoded_depth8bit, decoded_depth8bit, decoded_depth8bit])

    show_img = decoded_depth8bit_3ch.copy()

    Imagenp, boxes, scores = show_inference(
        detection_model, decoded_depth8bit_3ch)
    if len(boxes) > 0:
        max_score_index = scores.index(max(scores))
        xmin, xmax, ymin, ymax = boxes[max_score_index]
        person_crop = decoded_depth8bit_3ch[ymin:ymax, xmin:xmax]
        # cv2.imshow('person_crop', person_crop)

        # Activity prediction
        PilDepth = cv2.resize(person_crop, modelsize)
        PilDepth = np.array(PilDepth)/255.0
        PilDepth = np.expand_dims(PilDepth, axis=0)
        resultDepth = Depth_model.predict(PilDepth)

        activity_score = resultDepth[0]
        activity_name, activity_confidence = (
            class_names[np.argmax(activity_score)], float(100 * np.max(activity_score)))

# ----------------------------------------------------------------------------------------------------------------------
        cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(show_img, "{} : {:.2f}%".format(activity_name, activity_confidence),
                    org=(xmin+2, ymax-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=2)

    cv2.imshow('image', show_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
depth_stream.release()
cv2.destroyAllWindows()
