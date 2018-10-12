#####################################################################
#####################################################################
#
# Create synthetic tensors with K slices, each of dimension I x J x Ki
# and save to HDFS.  Use R as the number of components in each tensor.
# Tensors can have different levels of homo- and heteroskedastic error
# and different levels of collinearity in the factor matrices.
#
#    Kareem S. Aggour <aggour@ge.com>
#
# NOTE: X dimensions are of the form [z,x,y] NOT [x,y,z]!!!!!
#
#####################################################################
#####################################################################

import numpy as np
import subprocess, sys
import math
import argparse
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import norm
from pyspark import SparkContext

sc = SparkContext(appName='Create synthetic tensors')

#####################################################################
# input variables
#####################################################################
# tensor slice dimensions: I x J x Ki
I = 0
J = 0
Ki = 0

# number of slices -- note that final tensor will be I x J x (Ki*K)!!
K = 0

# rank of tensors
R = 0

# number of tensors to create
N = 0

# levels of factor matrix column collinearity
cRange = [0, 0.5, 0.9]

# levels of homoskedastic error
l1Range = [0, 1, 5, 10]
#l1Range = [0, 10]

# levels of heteroskedastic error
l2Range = [0, 1, 5]
#l2Range = [0, 5]

#####################################################################
# global variables
#####################################################################
A = 0
B = 0
c = 0

outputDir=''
hdfsDir=''


def strReplace(filename):
    with open(filename) as f:
        newText = f.read().replace(', # W, ', '')
    with open(filename, 'w') as f:
        f.write(newText)
        f.close()

def outerProduct(A, B, C):
    X = np.zeros((Ki, I, J))
    for i in range(0,I):
        for j in range(0,J):
            for k in range(0,Ki):
                sum = 0.0
                for r in range(0,R):
                    sum = sum + A.item(i,r) * B.item(j,r) * C.item(k,r)
                    X.itemset((k,i,j), sum)
#    print X
    return X

def createCollinearMatrix(rows,R,congruence):
    F = np.ones((R,R)) * congruence
    for i in range(0,R):
	F[i,i] = 1

    L = np.linalg.cholesky(F)
    L = L.T
    mat = np.random.rand(rows,R)
    Q,R = np.linalg.qr(mat)
    ret = np.dot(Q, L)
    return ret

def createTensorSlice(partition):
    ret = []
    rows = list(partition)

    rowCount = len(rows)
    stepSize = rowCount

    for row in rows:
	if c > 0:
	    Ci = createCollinearMatrix(Ki,R,c)
	else:
	    Ci = np.random.rand(Ki,R)
	#Xi = outerProduct (A, B, Ci)
	Xi = kruskal_to_tensor([Ci, A, B])
	N1 = np.random.randn(Ki,I,J)
	N2 = np.random.randn(Ki,I,J)
	normXi = norm(Xi, 2)
	normN1 = norm(N1, 2)
	normN2 = norm(N2, 2)

	filename = 'X-'+str(row*Ki)

	for l1 in l1Range:
	    for l2 in l2Range:
		add = '-C'+str(c)+'-L1_'+str(l1)+'-L2_'+str(l2)+'-'+str(globalN)+'/'
		newOutputDir = outputDir + add
		newHDFSDir = hdfsDir + add
		if l1 > 0:
		    Xi1 = Xi + math.pow(((100/l1) - 1), -0.5)*(normXi/normN1)*N1
		else:
		    Xi1 = Xi
		if l2 > 0:
		    N2Xi1 = N2 * Xi1
		    Xi2 = Xi1 + math.pow(((100/l2) - 1), -0.5)*(norm(Xi1, 2)/norm(N2Xi1, 2))*N2Xi1
		else:
		    Xi2 = Xi1

		np.save(newOutputDir + filename, Xi2)
		subprocess.call(['hadoop fs -moveFromLocal ' + newOutputDir + filename + '.npy ' + newHDFSDir], shell=True)

#        print Xi.shape
	ret.append(row)
    return ret

if __name__ == "__main__":
    global globalN

    parser = argparse.ArgumentParser(description='Create a tensor to test Spark-based implementation of PARAFAC-ALS.')
    parser.add_argument('-I', '--i', help='I dimension', type=int, required=False, default=366)
    parser.add_argument('-J', '--j', help='J dimension', type=int, required=False, default=366)
    parser.add_argument('-Ki', '--ki', help='Ki dimension', type=int, required=False, default=5)
    parser.add_argument('-K', '--k', help='K dimension', type=int, required=False, default=20000)
    parser.add_argument('-R', '--rank', help='Tensor rank, i.e., number of components in decomposition', type=int, required=False, default=5)
    parser.add_argument('-C', '--c', help='Collinearity (0=N, 1=Y)', type=int, required=False, default=0)
    parser.add_argument('-H', '--h', help='Homo- and heteroskedastic noise (0=N, 1=Y)', type=int, required=False, default=0)
    parser.add_argument('-N', '--n', help='Number of tensors', type=int, required=False, default=0)

    args = parser.parse_args()

    I = args.i
    J = args.j
    Ki = args.ki
    K = args.k
    R = args.rank
    C = args.c
    H = args.h
    N = args.n

    #outputDir='/mnt/isilon/aggour/rpi/spark/data-100x1000x5x30/'
    label = str(I)+'x'+str(J)+'x'+str(Ki)+'x'+str(K)+'-R'+str(R)
    if C == 0:
	cRange = [0]
    if H == 0:
	l1Range = [0]
	l2Range = [0]

    outputDir='/mnt/isilon/aggour/rpi/spark/data-' + label
    hdfsDir='/user/aggour/rpi/spark/tensor-' + label
    for globalN in range(0,N):
	for c in cRange:
	    for l1 in l1Range:
	        for l2 in l2Range:
		    add = '-C'+str(c)+'-L1_'+str(l1)+'-L2_'+str(l2)+'-'+str(globalN)+'/'
		    newOutputDir = outputDir + add
		    newHDFSDir = hdfsDir + add
		    print newHDFSDir
		    print newOutputDir
		    subprocess.call(['hadoop fs -rm -r -skipTrash ' + newHDFSDir], shell=True)
		    subprocess.call(['hadoop fs -mkdir ' + newHDFSDir], shell=True)
		    subprocess.call(['hadoop fs -chmod 777 ' + newHDFSDir], shell=True)
		    subprocess.call(['mkdir ' + newOutputDir], shell=True)
		    subprocess.call(['chmod 777 ' + newOutputDir], shell=True)

    for globalN in range(0,N):
	for c in cRange:
	    print 'c =',c
	    # if congruence is not 0 then need to make the factor matrices collinear!
	    if c > 0:
	        A = createCollinearMatrix(I,R,c)
	        a0=A[:,0]
	        a1=A[:,1]
	        a2=A[:,2]
	        a3=A[:,3]
	        print '  a01:',np.dot(a0, a1) / (np.linalg.norm(a0) * np.linalg.norm(a1))
	        print '  a23:',np.dot(a2, a3) / (np.linalg.norm(a2) * np.linalg.norm(a3))
	        B = createCollinearMatrix(J,R,c)
	        a0=B[:,0]
	        a1=B[:,1]
	        a2=B[:,2]
	        a3=B[:,3]
	        print '  b01:',np.dot(a0, a1) / (np.linalg.norm(a0) * np.linalg.norm(a1))
	        print '  b23:',np.dot(a2, a3) / (np.linalg.norm(a2) * np.linalg.norm(a3))
	    else:
	        A = np.random.rand(I,R) 
	        B = np.random.rand(J,R) 
	    rdd = sc.parallelize(range(0,K), 1000)
	    rdd.mapPartitions (createTensorSlice).collect()
	    print 'Number of files created:',rdd.count()

#    subprocess.call(['hadoop fs -moveFromLocal ' + outputDir + '* ' + hdfsDir], shell=True)
    print 'Tensor saved to HDFS: ',hdfsDir

