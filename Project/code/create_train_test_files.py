from keras.preprocessing import image as image_utils
import numpy as np
import json
import pickle as pkl
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("type",help="choose experiment type",choices=["SEEN","UNSEEN"],default="SEEN")
args = parser.parse_args()

"""
This file creates 2 directories TRAIN_DATA_DIR, TRAIN_DATA_DIR
Each dir contains train and test data split, each file having BATCH_SIZE no. images data
These split is created to reduce the file size to be loaded at a time.
"""

DATA_DIR_ROOT = "../data/MIRACL_VC1/"

if args.type == "SEEN":
	TRAIN_DATA_DIR = "train_anno_SEEN/"
	TEST_DATA_DIR = "test_anno_SEEN/"
else:
	TRAIN_DATA_DIR = "train_anno_UNSEEN/"
	TEST_DATA_DIR = "test_anno_UNSEEN/"


def load_and_process_image(path):
	image = image_utils.load_img(DATA_DIR_ROOT+path, target_size=(175, 175))
	image = image_utils.img_to_array(image)/255
	return image

def split_data_UNSEEN(path):
	"""
	path = path of annotations.json file
	n_test = no. of test speakers
	test_id = to specify id of test speakers
	"""
	with open(path) as f:
		data = json.load(f)

	test_id = ["M08","F10"]

	print(test_id)

	n_test = len(test_id)

	blob_speaker_train = []
	blob_speaker_test = []

	for speaker in data:
		if speaker["speaker_id"] in test_id:
			blob_speaker_test.append(speaker)
			print("test : ",speaker["speaker_id"])
		else:
			blob_speaker_train.append(speaker)
			# print("train : ",speaker["speaker_id"])

	print("train count : ",len(blob_speaker_train))
	print("test count : ",len(blob_speaker_test))

	BATCH_SIZE = 1024 # max no. of entries in each train/test file

	print("generating train files ....")
	TRAIN_LABELS = []
	TRAIN_IMG_PATH = []
	for blob in blob_speaker_train:
		words = blob["words"]
		for word in words:
			label = int(word["word_id"])-1
			img_path = word["image_path"]
			img_count = len(img_path)
			TRAIN_LABELS = TRAIN_LABELS + [label]*img_count
			TRAIN_IMG_PATH = TRAIN_IMG_PATH + img_path

	# stores labels and images data in files if size BATCH_SIZE
	for i in range(0,len(TRAIN_IMG_PATH),BATCH_SIZE):
		blob_train_data = {}
		blob_train_data['labels'] = TRAIN_LABELS[i:i+BATCH_SIZE]
		blob_img_path = TRAIN_IMG_PATH[i:i+BATCH_SIZE]
		blob_img = []
		for img_path in blob_img_path:
			blob_img.append(load_and_process_image(img_path))
		blob_train_data['images'] = blob_img

		with open(DATA_DIR_ROOT+TRAIN_DATA_DIR+"{:02d}".format(i//BATCH_SIZE)+".pkl",'wb') as f:
			pkl.dump(blob_train_data,f)
		print("i = ",i)
		print(len(blob_train_data["labels"]))
		print(len(blob_train_data["images"]))


	print("generating test files ....")
	TEST_LABELS = []
	TEST_IMG_PATH = []
	for blob in blob_speaker_test:
		words = blob["words"]
		for word in words:
			label = int(word["word_id"])-1
			img_path = word["image_path"]
			img_count = len(img_path)
			TEST_LABELS = TEST_LABELS + [label]*img_count
			TEST_IMG_PATH = TEST_IMG_PATH + img_path

	# stores labels and images data in files if size BATCH_SIZE
	for i in range(0,len(TEST_IMG_PATH),BATCH_SIZE):
		blob_test_data = {}
		blob_test_data['labels'] = TEST_LABELS[i:i+BATCH_SIZE]
		blob_img_path = TEST_IMG_PATH[i:i+BATCH_SIZE]
		blob_img = []
		for img_path in blob_img_path:
			blob_img.append(load_and_process_image(img_path))
		blob_test_data['images'] = blob_img

		with open(DATA_DIR_ROOT+TEST_DATA_DIR+"{:02d}".format(i//BATCH_SIZE)+".pkl",'wb') as f:
			pkl.dump(blob_test_data,f)
		print("i = ",i)
		print(len(blob_test_data["labels"]))
		print(len(blob_test_data["images"]))
	

def split_data_SEEN(path):
	"""
	path = path of annotations.json file
	n_test = no. of test speakers
	test_id = to specify id of test speakers
	"""
	with open(path) as f:
		data = json.load(f)

	BATCH_SIZE = 1024 # max no. of entries in each train/test file

	print("spliting data ....")
	
	TRAIN_LABELS = []
	TRAIN_IMG_PATH = []

	TEST_LABELS = []
	TEST_IMG_PATH = []

	TEST_UTT_ID = ["09","10"]

	for blob in data:
		words = blob["words"]
		for word in words:
			label = int(word["word_id"])-1
			utts = word["utterances"]
			for utt in utts:
				img_path = utt["image_path"]
				img_count = len(img_path)
				if utt["utt_id"] in TEST_UTT_ID:
					TEST_LABELS = TEST_LABELS + [label]*img_count
					TEST_IMG_PATH = TEST_IMG_PATH + img_path
				else:
					TRAIN_LABELS = TRAIN_LABELS + [label]*img_count
					TRAIN_IMG_PATH = TRAIN_IMG_PATH + img_path

	print(len(TRAIN_LABELS))
	print(len(TRAIN_IMG_PATH))
	print(len(TEST_LABELS))
	print(len(TEST_IMG_PATH))

	print("generating train files ....")
	# stores labels and images data in files if size BATCH_SIZE
	for i in range(0,len(TRAIN_IMG_PATH),BATCH_SIZE):
		blob_train_data = {}
		blob_train_data['labels'] = TRAIN_LABELS[i:i+BATCH_SIZE]
		blob_img_path = TRAIN_IMG_PATH[i:i+BATCH_SIZE]
		blob_img = []
		for img_path in blob_img_path:
			blob_img.append(load_and_process_image(img_path))
		blob_train_data['images'] = blob_img

		with open(DATA_DIR_ROOT+TRAIN_DATA_DIR+"{:02d}".format(i//BATCH_SIZE)+".pkl",'wb') as f:
			pkl.dump(blob_train_data,f)
		print("i = ",i)
		print(len(blob_train_data["labels"]))
		print(len(blob_train_data["images"]))


	print("generating test files ....")
	# stores labels and images data in files if size BATCH_SIZE
	for i in range(0,len(TEST_IMG_PATH),BATCH_SIZE):
		blob_test_data = {}
		blob_test_data['labels'] = TEST_LABELS[i:i+BATCH_SIZE]
		blob_img_path = TEST_IMG_PATH[i:i+BATCH_SIZE]
		blob_img = []
		for img_path in blob_img_path:
			blob_img.append(load_and_process_image(img_path))
		blob_test_data['images'] = blob_img

		with open(DATA_DIR_ROOT+TEST_DATA_DIR+"{:02d}".format(i//BATCH_SIZE)+".pkl",'wb') as f:
			pkl.dump(blob_test_data,f)
		print("i = ",i)
		print(len(blob_test_data["labels"]))
		print(len(blob_test_data["images"]))

if not os.path.exists(DATA_DIR_ROOT+TRAIN_DATA_DIR):
	os.makedirs(DATA_DIR_ROOT+TRAIN_DATA_DIR)
if not os.path.exists(DATA_DIR_ROOT+TEST_DATA_DIR):
	os.makedirs(DATA_DIR_ROOT+TEST_DATA_DIR)

if args.type == "UNSEEN":
	split_data_UNSEEN(DATA_DIR_ROOT+"annotations.json")
if args.type == "SEEN":
	split_data_SEEN(DATA_DIR_ROOT+"annotation_utt.json")

with open(DATA_DIR_ROOT+TRAIN_DATA_DIR+"00.pkl",'rb') as f:
	train_data = pkl.load(f)

# with open(DATA_DIR_ROOT+TEST_DATA_FILE,'rb') as f:
# 	test_data = pkl.load(f)

TRAIN_LABELS = train_data["labels"]
TRAIN_IMG = train_data["images"]
print(TRAIN_LABELS[0])
print(TRAIN_IMG[0].shape)