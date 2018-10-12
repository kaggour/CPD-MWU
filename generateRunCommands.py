#####################################################################
#####################################################################
#
# Generate run commands to test PARAslice-Spark
#
#    Kareem S. Aggour <aggour@ge.com>
#
#####################################################################
#####################################################################

import sys

def main(argv):
    # What tensor ranks to test
#    R=[5,10,25,50]
    R=[40]
#    R=[5]

    # What types of regularization (0 = none, 1 = Tikhonov, 2 = Proximal)
    G=[0,1,2]
#    G=[0,1]

    # Lambda values for regularization
    L=['0.001','0.01','0.1']
#    L=[0,'0.00001','0.0001','0.001']
#    L=['0.000001','0.00001','0.0001']

    # What types of sketching (0 = none, 1 = CPRAND, 2 = random slice sketching, 3 = random entry sketching)
    S=[0,1,2,3]
#    S=[2,3]
#    S=[0]
#    S=[3]

    # What levels of sketching (e.g., 10^-6)
    K=[0,'0.0001','0.000001','0.00000001']
#    K=[0]
#    K=['0.001','0.0001','0.00001','0.000001']
#    K=['0.1','0.01']
#    K=[0,'0.000001']
#    K=['0.000001']

    # How many runs to test
#    runs=range(0,10)
    runs=range(0,3)
#    runs=range(0,1)

    # Input directories in HDFS
    HDFS=['tensor-L0534-5']
#    HDFS=['tensor-366x366x5x100000-R5-C0-L1_0-L2_0-0', 'tensor-366x366x5x100000-R5-C0-L1_10-L2_5-0', 'tensor-366x366x5x100000-R5-C0-L1_0-L2_0-1', 'tensor-366x366x5x100000-R5-C0-L1_10-L2_5-1', 'tensor-366x366x5x100000-R5-C0-L1_0-L2_0-2', 'tensor-366x366x5x100000-R5-C0-L1_10-L2_5-2', 'tensor-366x366x5x100000-R5-C0-L1_0-L2_0-3', 'tensor-366x366x5x100000-R5-C0-L1_10-L2_5-3', 'tensor-366x366x5x100000-R5-C0-L1_0-L2_0-4', 'tensor-366x366x5x100000-R5-C0-L1_10-L2_5-4']

    # Print out full content that can be used to create a shell script
    print '#!/bin/bash'
    print ''

    for ru in runs:
	if ru == 0:
	    seed = 12345
	elif ru == 1:
	    seed = 6789
	elif ru == 2:
	    seed = 4567
	else:
	    seed = 0 # let code do random initial conditions
	for r in R:
	    for g in G:
		for l in L:
		    # invalid combinations (no regularization, but reg parameter != 0)
		    if g == 0 and l > 0:
			continue;
		    if g != 0 and l == 0:
			continue;
		    for s in S:
			for k in K:
			    # invalid combinations (no slice sketching, but sketching rate != 0)
			    if (s == 2 or s == 3) and k == 0:
				continue;
			    if s != 2 and s != 3 and k > 0:
				continue;
			    for hdfsDir in HDFS:
				filename = 'run-R' + str(r) + '-G' + str(g) + '-S' + str(s) + '-K' + str(k) + '-L' + str(l) + '-HDFS_' + hdfsDir + '-run_' + str(ru) + '.txt'
				filename = filename.replace(" ", "")
				realHDFS = '/user/aggour/rpi/spark/' + hdfsDir + '/'
				print './submit.sh PARAslice.py -R',r,'-G',g,'-S',s,'-K',k,'-L',l,'-Sd',seed,'-I',realHDFS,'>',filename

if __name__ == "__main__":
    main(sys.argv)

