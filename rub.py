import numpy as np

seq1 = np.load(r'E:\lyh\data\beijingdata\alldata\video_preprocessed_256\lip\normalize\1-2-video.npy')
seq2 = np.load(r'E:\lyh\data\beijingdata\alldata\video_preprocessed_256\lip\normalize2\1-2-video.npy')
print(seq1.shape)
print(seq2.shape)
print(seq1.max())
print(seq2.max())
print(seq1.min())
print(seq2.min())