import os
import json 

DATA_DIR_ROOT = "../data/MIRACL_VC1/"

DIR_ROOT = "../data/MIRACL_VC1/dataset/dataset/"
personID = sorted(os.listdir(DIR_ROOT))


wordDict = {"01":"Begin","02":"Choose","03":"Connection","04":"Navigation","05":"Next","06":"Previous","07":"Start","08":"Stop","09":"Hello","10":"Web"}
phraseDict = {"01":"Stop navigation.","02":"Excuse me.","03":"I am sorry.","04":"Thank you.","05":"Good bye.","06":"I love this game.","07":"Nice to meet you.","08":"You are welcome.","09":"How are you?","10":"Have a good time."}
"""
creates an annotation file structure
// list of dicts of speakers
[{	speaker_id = "F01",
	words = [
		{
			word_id = "01",
			utterances = [
					{
						utt_id = "01",
						image_path = [images_path_list],
					},
					{
						utt_id = "02",
						image_path = [images_path_list],
					},
					...
				]
			
		},
		{
			word_id = "02",
			utterances = [
					{
						utt_id = "01",
						image_path = [images_path_list],
					},
					{
						utt_id = "02",
						image_path = [images_path_list],
					},
					...
				]
		},
		....
	]
},
{	speaker_id = "F02",
	words = [
		{
			word_id = "01",
			utterances = [
				{
					utt_id = "01",
					image_path = [images_path_list],
				},
				{
					utt_id = "02",
					image_path = [images_path_list],
				},
				...
			]
		},
		{
			word_id = "02",
			utterances = [
				{
					utt_id = "01",
					image_path = [images_path_list],
				},
				{
					utt_id = "02",
					image_path = [images_path_list],
				},
				...
			]
		},
		....
	]
},
....
]
Refer this link for dataset clearity
https://sites.google.com/site/achrafbenhamadou/-datasets/miracl-vc1
"""

DATA = []
for pid in personID:
	PERSON = {}
	PERSON['speaker_id'] = pid
	WORDS = []

	words_path = DIR_ROOT + pid + "/words/"
	words = sorted(os.listdir(words_path))

	for word in words:
		CURR_WORD = {}
		CURR_WORD['word_id'] = word
		UTTERANCES = []

		words_instance = words_path + word
		for instance in sorted(os.listdir(words_instance)):
			CURR_UTT = {}
			CURR_UTT["utt_id"] = instance
			IMG_PATH_LIST = []

			i_path = words_instance + "/" + instance
			
			for img in sorted(os.listdir(i_path)):
				img_path = i_path + "/" + img
				if "c_color" in img_path:
					if "../data/MIRACL_VC1/" in img_path:
						img_path = img_path.replace("../data/MIRACL_VC1/","")
					elif "./" in img_path:
						img_path = img_path.replace("./","")

					IMG_PATH_LIST.append(img_path)

			CURR_UTT['image_path'] = IMG_PATH_LIST
			UTTERANCES.append(CURR_UTT)

		CURR_WORD["utterances"] = UTTERANCES
		WORDS.append(CURR_WORD)

	PERSON["words"] = WORDS
	DATA.append(PERSON)

with open(DATA_DIR_ROOT+'annotation_utt.json', 'w') as fp:
    json.dump(DATA, fp)

import pdb
pdb.set_trace()
with open(DATA_DIR_ROOT+'annotation_utt.json') as f:
	data = json.load(f)
print(len(data))
print(data[0].keys())