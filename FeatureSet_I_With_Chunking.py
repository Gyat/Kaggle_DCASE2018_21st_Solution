'''
========================================================================================
This module is responsible for the following:
1. As a first step, chunking and padding of variable audio length files are performed.
2. The chunked files are of 4 secs.
3. This module generates the Feature Set - I, as described in the paper.

This module uses Multiprocessing to speed-up the feature engineering part.
========================================================================================

Input: Train and Test .wav files - variable length
Output: 
    1. Train and Test .wav files - fixed length of 4 secs, with 10% overlap in each chunks.
    2. Mel-scaled spectograms for Feature Set - I, in the form of pickle files.
       The generated pickles for the spectograms have been split into two parts since the single file crosses the valid
       file size.
'''


import os
import numpy as np
import IPython
import matplotlib
from matplotlib import pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import librosa
from librosa import display
import pandas as pd
import pickle
import math
from time import time
from multiprocessing import Pool
from joblib import Parallel, delayed
import socket, errno, sys


np.random.seed(1)

rootDir = "/home/ubuntu/kaggle/audioTagger/"

class Config(object):
    def __init__(self,
                 sampling_rate=22050, n_classes=41,
                 #data_dir=rootDir+'input/audio_train_trimmed',
                 n_mels=64, frame_weigth=80, frame_shift=10):
        self.sampling_rate = sampling_rate
        self.n_classes = n_classes
        #self.data_dir = data_dir
        self.n_fft = int(frame_weigth / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weigth
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)

        
       
def extractFeatures(row, duration=3, cut_duration = 4):
    try:
        fn = row[0]
        data_dir = row[1]
        fname = os.path.join(data_dir, fn)    
        #data, _ = librosa.load(fname, sr=config.sampling_rate, duration=duration)
        audio, _ = librosa.load(fname, sr=config.sampling_rate)
        tot_duration = librosa.core.get_duration(audio)
        numSamples = math.ceil(tot_duration/cut_duration)
        logMelCollection = []
        fnameCollection = []
        end_marker = 0
        part_marker = 0
        while end_marker < tot_duration:         
            if end_marker == 0:
                offset = 0
            else:
                ## Cut samples with 10% overlap
                offset = (offset + cut_duration) - (cut_duration*0.1)
            end_marker = offset + cut_duration
            data, _ = librosa.load(fname, sr=config.sampling_rate, offset = offset, duration = cut_duration)
            librosa.output.write_wav(data_dir+'4secs/'+'part'+str(part_marker)+'_'+fn, data, config.sampling_rate)
            melspec = librosa.feature.melspectrogram(data, sr=config.sampling_rate,
                                                     n_fft=config.n_fft, hop_length=config.hop_length,
                                                     n_mels=config.n_mels)
            logmel = librosa.core.power_to_db(melspec) 
            logMelCollection.append(logmel)
            fnameCollection.append(fn)
            part_marker = part_marker+1
        return logMelCollection, fnameCollection    
    except Exception as e:
        print("Error processing file: ",fn)    
        print("Numsamples: ",numSamples)
        print("Total duration: ",tot_duration)


config = Config(frame_weigth=80, frame_shift=10)

train = pd.read_csv(rootDir + "input/train.csv")
test = pd.read_csv(rootDir + "input/sample_submission.csv")
print("Removing bad files from test")
exclusionFiles = ['0b0427e2.wav','6ea0099f.wav','b39975f5.wav']
test = test[~test['fname'].isin(exclusionFiles)]
test.reset_index(drop=True, inplace = True)

file_counter = 1        
filesToBeProcessed = []
for filename in train.fname:
    if file_counter % 1000 == 0:
        print(file_counter)
        #print("Processing :",filename)
    filesToBeProcessed.append((filename, rootDir+'input/audio_train_trimmed/'))
    file_counter = file_counter+1 

numToProcess = len(filesToBeProcessed)
print("Number of files to process: Train set", numToProcess)  

t0 = time()
numSubprocess = 50
print("Starting multiprocessing with", numSubprocess, "subprocesses..." )
pool = Pool(numSubprocess)
try:
    resultSet = pool.map(extractFeatures, filesToBeProcessed)
    pool.close()
    pool.join()
except socket.error as e:
    if e.errno == errno.EPIPE:
        #remote peer disconnected
        print ({'Status':'Remote Disconnected Error'})
    else:
        print ({'Status':'Other Socket Error'})
except IOError as e:  
        print ({'Status':'IOError'})


filenames = []
X = []
for result in resultSet:
	for file_names in result[1]:
		filenames.append(file_names)
	for sound_sequence in result[0]:
		X.append(sound_sequence)

max_length = np.max([x.shape[1] for x in X])
# Pad zero to make them all the same length
X2 = [np.pad(x, ((0, 0), (0, max_length - x.shape[1])), 'constant') for x in X]

melSpectogramTrain = np.array(X2)

print("Shape of melSpectogramTrain: ", melSpectogramTrain.shape)
rows = melSpectogramTrain.shape[0]
rowsToSave = int(rows/2)
with open(rootDir + "input/audio_train_trimmed/4secs/melSpectogramTrain_p1.pkl", 'wb') as handle:
    pickle.dump(melSpectogramTrain[:rowsToSave], handle) 

with open(rootDir + "input/audio_train_trimmed/4secs/melSpectogramTrain_p2.pkl", 'wb') as handle:
    pickle.dump(melSpectogramTrain[rowsToSave:], handle) 
    
with open(rootDir + "input/audio_train_trimmed/4secs/filenamesTrain.pkl", 'wb') as handle:
    pickle.dump(filenames, handle)     
    
################ FOR TEST FILES #########################


file_counter = 1        
filesToBeProcessed = []
for filename in test.fname:
    if file_counter % 1000 == 0:
        print(file_counter)
        #print("Processing :",filename)
    filesToBeProcessed.append((filename, rootDir+'input/audio_test_trimmed/'))
    file_counter = file_counter+1 

numToProcess = len(filesToBeProcessed)
print("Number of files to process: Test set", numToProcess)  

t0 = time()
numSubprocess = 50
print("Starting multiprocessing with", numSubprocess, "subprocesses..." )
pool = Pool(numSubprocess)
try:
    resultSet = pool.map(extractFeatures, filesToBeProcessed)
    pool.close()
    pool.join()
except socket.error as e:
    if e.errno == errno.EPIPE:
        #remote peer disconnected
        print ({'Status':'Remote Disconnected Error'})
    else:
        print ({'Status':'Other Socket Error'})
except IOError as e:  
        print ({'Status':'IOError'})
        
filenames = []
X = []
for result in resultSet:
	for file_names in result[1]:
		filenames.append(file_names)
	for sound_sequence in result[0]:
		X.append(sound_sequence)

max_length = np.max([x.shape[1] for x in X])
# Pad zero to make them all the same length
X2 = [np.pad(x, ((0, 0), (0, max_length - x.shape[1])), 'constant') for x in X]

melSpectogramTest = np.array(X2)

print("Shape of melSpectogramTest: ", melSpectogramTest.shape)
with open(rootDir + "input/audio_test_trimmed/4secs/melSpectogramTest.pkl", 'wb') as handle:
    pickle.dump(melSpectogramTest, handle) 

with open(rootDir + "input/audio_test_trimmed/4secs/filenamesTest.pkl", 'wb') as handle:
    pickle.dump(filenames, handle)   
    

