#!/opt/local/bin/python

#####################################################################
#####################################################################
#
# Implement continuous wavelet transform (CWT) on a time series matrix
#
#    Kareem S. Aggour <aggour@ge.com>
#
# NOTE: tensor dimensions are of the form [z,x,y] NOT [x,y,z]!!!!!
#
#####################################################################
#####################################################################

import numpy as np
import sys
from numpy import genfromtxt
from scipy import signal
#import pandas as pd
import sklearn
from sklearn import preprocessing
import subprocess

sliceWidth = 5
#filename = '/home/aggour/rpi/dissertation/spark/data/L0525_DELIVERY.csv'
filename = '/home/aggour/rpi/dissertation/spark/data/L0534.csv'

def main(argv):

    # load data from csv
    dataArray = genfromtxt(filename, delimiter=',')

    (K,I) = dataArray.shape
    for i in range(0,I):
	sensor = dataArray[:,i]
	dataArray[:,i] = dataArray[:,i] - sensor.mean()
	dataArray[:,i] = dataArray[:,i] / sensor.max()

#    scaledArray = sklearn.preprocessing.normalize(dataArray, axis=0)
    scaledArray = dataArray

    # scale data between [-1,1]
    # http://scikit-learn.org/stable/modules/preprocessing.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    #scaledArray = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1), copy=True).fit_transform(dataArray)

    # CWT
    # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.cwt.html
    (K,I) = scaledArray.shape
    J = I
    widths=np.arange(1,J)
    cwtmatr = signal.cwt(scaledArray[:,1], signal.ricker, widths)
#    print cwtmatr
#    print 'array shape =',scaledArray[:,1].shape
#    print 'cwt shape =',cwtmatr.shape

    print I,J,K
    # there must be a smarter way to do this...
    Kmax = K
    while Kmax%sliceWidth != 0:
	Kmax = Kmax+1

    print 'Scaled Array signals...'
    for j in range(0,J):
	tmp = scaledArray[:,j]
	print tmp.min(), tmp.mean(), tmp.max(), tmp.shape

    # create tensor and set first slice to original time series matrix
    tensor = np.zeros((Kmax, I, J))
    tensor[0:K,:,0] = scaledArray

    # populate third mode of tensor with wavelet transform entries
    print 'Ricker CWT transform...'
    for j in range(0,J):
	cwtmatr = signal.cwt(scaledArray[:,j], signal.ricker, widths)
	print cwtmatr.min(), cwtmatr.mean(), cwtmatr.max(), cwtmatr.shape
        tensor[0:K,1:I,j] = cwtmatr.transpose()

    '''
    sz = I*J*K
    print 'siz=',sz
    #nnz = np.count_nonzero(tensor)
    nnz1 = 0.0
    nnz2 = 0.0
    nnz5 = 0.0
    nnz10 = 0.0
    nnz12 = 0.0
    for i in range(0,I):
	for j in range(0,J):
	    for k in range(0,K):
		if abs(tensor[k,i,j]) < 1e-1:
		    nnz1 = nnz1 + 1
		if abs(tensor[k,i,j]) < 1e-2:
		    nnz2 = nnz2 + 1
		if abs(tensor[k,i,j]) < 1e-5:
		    nnz5 = nnz5 + 1
		if abs(tensor[k,i,j]) < 1e-10:
		    nnz10 = nnz10 + 1
		if abs(tensor[k,i,j]) < 1e-12:
		    nnz12 = nnz12 + 1
    print 'nnz < e-1=',nnz1
    print 'nnz < e-2=',nnz2
    print 'nnz < e-5=',nnz5
    print 'nnz < e-10=',nnz10
    print 'nnz < e-12=',nnz12
    print 'ratio e-1=',nnz1*1.0/sz
    print 'ratio e-2=',nnz2*1.0/sz
    print 'ratio e-5=',nnz5*1.0/sz
    print 'ratio e-10=',nnz10*1.0/sz
    print 'ratio e-12=',nnz12*1.0/sz
    '''

#    print tensor
    print tensor.shape
    (K,I,J) = tensor.shape
    print I,J,K

    # slice tensor and save slices
    dashIdx=filename.rindex('/')
    dotIdx=filename.rindex('.')
    dirname=filename[dashIdx+1:dotIdx] + '-' + str(sliceWidth)
    outputDir='/mnt/isilon/aggour/rpi/spark/data-' + dirname + '/'
    hdfsDir='/user/aggour/rpi/spark/tensor-' + dirname + '/'

    subprocess.call(['hadoop fs -mkdir ' + hdfsDir], shell=True)
    subprocess.call(['hadoop fs -chmod 777 ' + hdfsDir], shell=True)
    subprocess.call(['mkdir ' + outputDir], shell=True)
    subprocess.call(['chmod 777 ' + outputDir], shell=True)

    print outputDir
    print hdfsDir
    for k in xrange(0,K-sliceWidth+1,sliceWidth):
	kmax = k + sliceWidth - 1
	print k,kmax
	slice = tensor[k:kmax+1,:,:]
	#print slice.shape
	pyfile = outputDir + 'X-' + str(k)
	np.save(pyfile, slice)

	if k%10000 == 0:
	    print 'Moving...',k
	    subprocess.call(['hadoop fs -moveFromLocal ' + outputDir + '*.npy ' + hdfsDir], shell=True)

    print 'Moving last...',k
    subprocess.call(['hadoop fs -moveFromLocal ' + outputDir + '*.npy ' + hdfsDir], shell=True)

if __name__ == "__main__":
    main(sys.argv)

