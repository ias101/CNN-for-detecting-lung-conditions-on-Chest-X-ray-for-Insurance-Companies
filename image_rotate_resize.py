import numpy as np
import tensorflow_addons as tfa
import cv2


def get_corner_coordinates(image):
    # Convert image to uint8 with range [0, 255]
    image_uint8 = (image * 255).astype(np.uint8)

    # Find contours in the image
    contours, _ = cv2.findContours(image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the object
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a rotated bounding box to the contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Get the angle of rotation
    angle = rect[2]

    # Get the corner coordinates
    return box, angle


def rotate_resize(work_image):
    """
    Rotates and resizes given numpy image (shape (w,h)) and returns it
    Params:
        work_image: given image
    Returns:
        result_image: fixed image
        result_type: what manipulations were done to the image (rotated, cropped, rotatedcroppped) or '' if nothing
    """
    corners, angle = get_corner_coordinates(work_image)
    result_image = work_image
    result_type = ''
    if not (-2 < angle < 2 or 92 >= angle > 88):
        if angle < 45:
            result_image = tfa.image.rotate(work_image, angle * 3.1415 / 180, interpolation='BILINEAR').numpy()
        else:
            result_image = tfa.image.rotate(work_image, (270+angle) * 3.1415 / 180, interpolation='BILINEAR').numpy()


        corners, angle = get_corner_coordinates(result_image)
        result_type+='rotated'

    if not (abs(sum(corners[0] - corners[1])) > 120 and abs(sum(corners[1] - corners[2])) > 120):
        result_image = result_image[max(corners[1][1],0):corners[3][1],
                        max(corners[0][0],0):corners[2][0]]
        result_image = cv2.resize(result_image, (128, 128), interpolation=cv2.INTER_LINEAR)
        result_type+='cropped'

    return result_image, result_type