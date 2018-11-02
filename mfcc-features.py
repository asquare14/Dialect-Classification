import scipy.io.wavfile
import numpy as np 
import sys
import os
import glob
from utils1 import DIALECT_DIR, DIALECT_LIST
from python_speech_features import mfcc

#creates ceps file and carries out the mfcc feature extraction
def create_ceps(wavfile):

	sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
	print(sampling_rate)
	ceps = mfcc(song_array)
	print(ceps.shape)
	bad_indices = np.where(np.isnan(ceps))
	b = np.where(np.isinf(ceps))
	ceps[bad_indices] = 0
	ceps[b] = 0
	write_ceps(ceps, wavfile)

def write_ceps(ceps, wavfile):

	base_wav, ext = os.path.splitext(wavfile)
	data_wav = base_wav + ".ceps"
	np.save(data_wav, ceps)


def main():

	for label, dialect in enumerate(DIALECT_LIST):
		for fn in glob.glob(os.path.join(DIALECT_DIR, dialect)):
			for wavfile in os.listdir(fn):
				if wavfile.endswith("wav"):
					create_ceps(os.path.join(DIALECT_DIR, dialect, wavfile))

if __name__ == "__main__":
	
	main()