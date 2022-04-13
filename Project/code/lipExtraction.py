import os
import cv2
import dlib
import numpy as np
# import faceAlignment as fa

DIR_ROOT = "../data/MIRACL_VC1/dataset/dataset/"
FACE_LANDMARK = "../data/MIRACL_VC1/shape_predictor_68_face_landmarks.dat"

personID = sorted(os.listdir(DIR_ROOT))


def crop_face_n_save_imgs():
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor(FACE_LANDMARK)  # Landmark identifier

    for pid in personID:
        phrases_path = DIR_ROOT + pid + "/phrases/"
        words_path = DIR_ROOT + pid + "/words/"

        phrases = sorted(os.listdir(phrases_path))
        words = sorted(os.listdir(words_path))

        for phrase in phrases:
            phrase_instance = phrases_path + phrase
            for instance in sorted(os.listdir(phrase_instance)):
                i_path = phrase_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    img_path = i_path + "/" + img
                    crop_img_path = i_path + "/c_" + img
                    crop_img(detector, predictor, img_path, crop_img_path)
            print("----DONE : ", phrase, "-----")

        for words in words:
            words_instance = words_path + words
            for instance in sorted(os.listdir(words_instance)):
                i_path = words_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    img_path = i_path + "/" + img
                    crop_img_path = i_path + "/c_" + img
                    crop_img(detector, predictor, img_path, crop_img_path)
            print("----DONE : ", words, "-----")
        print("****DONE : ", pid, "*****")


def crop_img(face_detector, landmark_detector, image, dest_path):
    img = cv2.imread(image)
    LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
    RESIZE = (100, 100)
    landmark_buffer = []
    face_rects = face_detector(img, 1)
    try:
        rect = face_rects[0]                    # Proper number of face
        landmark = landmark_detector(img, rect)   # Detect face landmarks
        landmark = shape_to_list(landmark)
        landmark_buffer.append(landmark)
        cropped_buffer = []
        for (i, landmark) in enumerate(landmark_buffer):
            # Landmark corresponding to lip
            lip_landmark = landmark[48:68]
            # Lip landmark sorted for determining lip region
            lip_x = sorted(lip_landmark, key=lambda pointx: pointx[0])
            lip_y = sorted(lip_landmark, key=lambda pointy: pointy[1])
            # Determine Margins for lip-only image
            x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)
            y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
            crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add,
                        lip_y[0][1]-y_add, lip_y[-1][1]+y_add)
            cropped = img[crop_pos[2]:crop_pos[3],
                          crop_pos[0]:crop_pos[1]]        # Crop image
            cropped = cv2.resize(
                cropped, (RESIZE[0], RESIZE[1]), interpolation=cv2.INTER_CUBIC)       # Resize
            cropped_buffer.append(cropped)
        cv2.imwrite(dest_path, cropped_buffer[0])
    except:
        max_x = img.shape[1] - 100
        max_y = img.shape[0] - 100
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = img[y: y + 100, x: x + 100]
        cv2.imwrite(dest_path, crop)


def remove_uncropped_img():
    """
    removes uncropped original images persent in dataset
    """
    for pid in personID:
        phrases_path = DIR_ROOT + pid + "/phrases/"
        words_path = DIR_ROOT + pid + "/words/"

        phrases = sorted(os.listdir(phrases_path))
        words = sorted(os.listdir(words_path))

        for phrase in phrases:
            phrase_instance = phrases_path + phrase
            for instance in sorted(os.listdir(phrase_instance)):
                i_path = phrase_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    if "c_c_color" not in img:
                        img_path = i_path + "/" + img
                        os.remove(img_path)

        for words in words:
            words_instance = words_path + words
            for instance in sorted(os.listdir(words_instance)):
                i_path = words_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    if "c_c_color" not in img:
                        img_path = i_path + "/" + img
                        os.remove(img_path)


def rename_img():
    """
    removes uncropped original images persent in dataset
    """
    for pid in personID:
        phrases_path = DIR_ROOT + pid + "/phrases/"
        words_path = DIR_ROOT + pid + "/words/"

        phrases = sorted(os.listdir(phrases_path))
        words = sorted(os.listdir(words_path))

        for phrase in phrases:
            phrase_instance = phrases_path + phrase
            for instance in sorted(os.listdir(phrase_instance)):
                i_path = phrase_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    os.rename(i_path + "/" + img,  i_path + "/" + img[2:])

        for words in words:
            words_instance = words_path + words
            for instance in sorted(os.listdir(words_instance)):
                i_path = words_instance + "/" + instance
                for img in sorted(os.listdir(i_path)):
                    os.rename(i_path + "/" + img,  i_path + "/" + img[2:])


def shape_to_list(shape):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


rename_img()
