import numpy as np
import glob
import csv
import librosa
import os
import subprocess


def process_new_data(wave_file):
	target_filename =  wave_file + '.npy'
	parent_path = 'data processing/asset/data/FINAL_DATA/train/audio'
	wave_file_path = parent_path + wave_file
	for fn in os.listdir(parent_path):
		print(fn)
		if fn == wave_file:
			wave, sr = librosa.load(wave_file_path, mono=True, sr=None)
			mfcc = librosa.feature.mfcc(wave, sr=16000)
			np.save(target_filename, mfcc, allow_pickle=False)
			print('DONE process new_data train')
			break
		else:
			print('khong co file ' + wave_file)
			break
	
   
process_new_data('train_23926.wav')