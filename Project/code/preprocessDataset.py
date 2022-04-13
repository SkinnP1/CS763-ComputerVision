import os
import cv2
import dlib
from FaceAlignment import FaceAligner
# import faceAlignment as fa

DIR_ROOT = "../data/MIRACL_VC1/dataset/dataset/"
FACE_LANDMARK = "../data/MIRACL_VC1/shape_predictor_68_face_landmarks.dat"

personID = sorted(os.listdir(DIR_ROOT))

def remove_depth_img():
	"""
	removes depth map images persent in dataset
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
					if "depth" in img:
						depth_img_path = i_path + "/" + img
						os.remove(depth_img_path)


		for words in words:
			words_instance = words_path + words
			for instance in sorted(os.listdir(words_instance)):
				i_path = words_instance + "/" + instance
				for img in sorted(os.listdir(i_path)):
					if "depth" in img:
						depth_img_path = i_path + "/" + img
						os.remove(depth_img_path)

def crop_face_n_save_imgs():
	detector = dlib.get_frontal_face_detector() #Face detector
	predictor = dlib.shape_predictor(FACE_LANDMARK) #Landmark identifier

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
					crop_img(detector,predictor,img_path,crop_img_path)
			print("----DONE : ",phrase,"-----")

		for words in words:
			words_instance = words_path + words
			for instance in sorted(os.listdir(words_instance)):
				i_path = words_instance + "/" + instance
				for img in sorted(os.listdir(i_path)):
					img_path = i_path + "/" + img
					crop_img_path = i_path + "/c_" + img
					crop_img(detector,predictor,img_path,crop_img_path)
			print("----DONE : ",words,"-----")
		print("****DONE : ",pid,"*****")


def crop_img(detector,predictor,img,dest_path):
	fa = FaceAligner(predictor, desiredFaceWidth=256)
	image = cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	# loop over the face detections
	for rect in rects:
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y
		faceAligned = fa.align(image, gray, rect)
		cv2.imwrite(dest_path,faceAligned)


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
					if "c_color" not in img:
						img_path = i_path + "/" + img
						os.remove(img_path)


		for words in words:
			words_instance = words_path + words
			for instance in sorted(os.listdir(words_instance)):
				i_path = words_instance + "/" + instance
				for img in sorted(os.listdir(i_path)):
					if "c_color" not in img:
						img_path = i_path + "/" + img
						os.remove(img_path)

def get_avg_img_count():
	"""
	this function gives average no. of images per utterance intance per person for each word
	"""
	img_count_dir = {}
	img_count_dir["01"] = 0
	img_count_dir["02"] = 0
	img_count_dir["03"] = 0
	img_count_dir["04"] = 0
	img_count_dir["05"] = 0
	img_count_dir["06"] = 0
	img_count_dir["07"] = 0
	img_count_dir["08"] = 0
	img_count_dir["09"] = 0
	img_count_dir["10"] = 0

	for pid in personID:
		words_path = DIR_ROOT + pid + "/words/"
		words = sorted(os.listdir(words_path))

		for word in words:
			n_img = 0
			n_ins = 0
			words_instance = words_path + word
			for instance in sorted(os.listdir(words_instance)):
				i_path = words_instance + "/" + instance
				n_img += len(os.listdir(i_path))
				n_ins+=1
			img_count_dir[word] += n_img/n_ins

	avg = 0
	print("word","->","AVG")
	for k in img_count_dir:
		img_count_dir[k] /= len(personID)
		avg += img_count_dir[k]
		print(k," -> ","{:.2f}".format(img_count_dir[k]))
	print("OVERALL AVERAGE ","{:.2f}".format(avg/10))

def get_min_max_img_count():
	"""
	this function gives average no. of images for any utterance for each word
	"""
	img_count_dir = {}
	img_count_dir["01"] = (50,0)
	img_count_dir["02"] = (50,0)
	img_count_dir["03"] = (50,0)
	img_count_dir["04"] = (50,0)
	img_count_dir["05"] = (50,0)
	img_count_dir["06"] = (50,0)
	img_count_dir["07"] = (50,0)
	img_count_dir["08"] = (50,0)
	img_count_dir["09"] = (50,0)
	img_count_dir["10"] = (50,0)

	for pid in personID:
		words_path = DIR_ROOT + pid + "/words/"
		words = sorted(os.listdir(words_path))

		for word in words:
			n_img_max = 0
			n_img_min = 50
			words_instance = words_path + word
			for instance in sorted(os.listdir(words_instance)):
				i_path = words_instance + "/" + instance
				n_img_max = max(n_img_max,len(os.listdir(i_path)))
				n_img_min = min(n_img_min,len(os.listdir(i_path)))
			
			img_count_dir[word] = (min(img_count_dir[word][0],n_img_min), max(img_count_dir[word][1],n_img_max))

	avg = 0
	print("word"," -> ","MIN"," -> ","MAX")
	for k in img_count_dir:
		print(k,"  ->  ",img_count_dir[k][0],"  ->  ",img_count_dir[k][1])

get_avg_img_count()