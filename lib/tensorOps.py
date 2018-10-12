#####################################################################
#####################################################################
#
# Little library of tensor operations
#
#    Kareem S. Aggour <aggour@ge.com>
#
# NOTE: tensor dimensions are of the form [z,x,y] NOT [x,y,z]!!!!!
#
#####################################################################
#####################################################################

import numpy as np
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import norm
from tensorly.tenalg import khatri_rao
from tensorly.base import unfold, fold

# get tensor dimensions
def getDim(rows):
    '''
    i=0
    j=0
    k=0
    normX=0.0
    for row in rows:
        k = k + row[0]
	normX = normX + row[3]
        if row[1] > i:
            i = row[1]
        if row[2] > j:
            j = row[2]
    '''
    (k,x,y,normX)=np.sum(rows, axis=0)
    (x,i,j,y) = np.max(rows, axis=0)
    return (k, i, j, normX)

def calculateFNormX(tensor):
    """
    Calculate Frobenius Norm of tensor
    """
    (K,I,J) = tensor.shape
    normX = 0.0
    for i in range(0,I):
        for j in range(0,J):
            for k in range(0,K):
                normX = normX + np.square(tensor.item((k,i,j)))
    return normX

def calculateFNormXTensorly(tensor):
    return norm(tensor, 2)

def calculateError(tensor, A, B, C):
    """
    Calculate Frobenius Norm of difference between tensor and decomposed tensor
    """
    (K,I,J) = tensor.shape
    (I,R) = A.shape
    error = 0.0
    normX = 0.0
    for i in range(0,I):
        for j in range(0,J):
            for k in range(0,K):
                sum = 0.0
                for r in range(0,R):
                    sum = sum + A.item(i,r) * B.item(j,r) * C.item(k,r)
                x = tensor.item((k,i,j))
                error = error + np.square(sum) - (2.0*sum*x)
                normX = normX + np.square(x)
    return np.sqrt((normX + error) / normX)

def calculateErrorTensorly(tensor, A, B, C):
    return norm(tensor - kruskal_to_tensor([C, A, B]), 2) / calculateFNormXTensorly(tensor)

# N^T * N . M^T * M
def ZTZ(M,N):
    #print N.transpose()
    #print N
    #print np.matmul(N.transpose(), N)
    #print ''
    #print M.transpose()
    #print M
    #print ''
    #print np.matmul(M.transpose(), M)
#    return np.multiply (np.matmul(N.transpose(), N), np.matmul(M.transpose(), M))
    return np.multiply (np.dot(N.T, N), np.dot(M.T, M))

def khatriRaoProdCell(M1, M2, m2rows, k, j):
    m2idx = int(np.mod (k, m2rows))
    m1idx = int((k - m2idx) / m2rows)
#    print '  M1=',M1.item((m1idx, j))
#    print '  M2=',M2.item((m2idx, j))
    return M1.item((m1idx, j)) * M2.item((m2idx, j))

# retSize = number of rows in return matrix (I, J, or Ki)
# krRows = number of Khatri-Rao product rows
def unfolded_3D_matrix_multiply(fold, X, M1, M2, I, J, Ki, R):
#    (n,R) = M1.shape
    (m2rows,R) = M2.shape
    if (fold == 1): # retSize = IxR, krRows = J*Ki
        MM = np.zeros((I, R))
        for i in range(0,I):
#            print 'i=',i
            for j in range(0,R):
#                print 'j=',j
                for k in range(0,J*Ki):
#                    print 'k=',k
                    c = k
                    t = np.floor (c / J)
                    c = np.remainder (c, J)
#                    print 'index(t,i,c) = (',int(t),',',i,',',int(c),')'
#                    print 'X=',X.item((int(t),i,int(c)))
#                    KR=khatriRaoProdCell(M1, M2, m2rows, k, j)
#                    print 'KR=',KR
#                    print 'current val at ',i,',',j,' val=',MM.item((i,j))
#                    print 'putting at ',i,',',j,' val=',KR*X.item((t,i,c))
                    MM.itemset((i,j), MM.item((i,j)) + X.item((int(t),i,int(c))) * khatriRaoProdCell(M1, M2, m2rows, k, j))

    elif (fold == 2): # retSize = JxR, krRows = I*Ki
        MM = np.zeros((J, R))
#        print X
        for i in range(0,J):
            for j in range(0,R):
                for k in range(0,I*Ki):
                    c = k
                    t = np.floor (c / I)
                    c = np.remainder (c, I)
#                    print 't=',t
#                    print 'c=',c
#                    print 'i=',i
#                    print 'X=',X.item((t,c,i))
                    MM.itemset((i,j), MM.item((i,j)) + X.item((int(t),int(c),i)) * khatriRaoProdCell(M1, M2, m2rows, k, j))

    elif (fold == 3): # retSize = KixR krRows = I*J
        MM = np.zeros((Ki, R))
        for i in range(0,Ki):
            for j in range(0,R):
                for k in range(0,I*J):
                    c = k
                    t = np.floor (c / I)
                    c = np.remainder (c, I)
                    MM.itemset((i,j), MM.item((i,j)) + X.item(i,int(c),int(t)) * khatriRaoProdCell(M1, M2, m2rows, k, j))

    return MM

def parseString(str):
    n = np.fromstring(str, dtype=float, sep=', ')
    return n

if __name__ == "__main__":
    print ''
    print '*************************'
    print '*** Testing tensorOps ***'
    print '*************************'
    print ''

    # test ZTZ
    A = np.array([[1,2],[3,4]])
    print 'A:\n--\n', A
    B = np.array([[5,6],[7,8]])
    print 'B:\n--\n', B
    #ans =
    #     740        1204
    #    1204        2000
    Z = ZTZ(A,B)

    print ''

    print 'B^T*B . A^T*A:\n--------------\n', Z

    print ''

    # inverse
    Ainv = np.linalg.inv(A)
    #ans =
    #   -2.0000    1.0000
    #    1.5000   -0.5000
    print 'A inverse:\n----------\n', Ainv

    print ''

    # check Khatri-Rao product
    v = khatriRaoProdCell(A, B, 2, 3, 1)
    # ans = 8*4 = 32
    print 'Khatri-Rao product:\n-------------------\n',v

    print ''

    # unfolded matrix multiply
    X=np.array([[[2,1],[4,3]],[[6,5],[8,7]]])
    (K,I,J)=X.shape
    R=2
    print 'X:\n--\n',X

    C = np.random.rand(2,2)
    print ''
    Xm = unfolded_3D_matrix_multiply (1, X, B, A, I, J, K, R)
    #ans =
    #   172   304
    #   268   472
    print 'UMM 1:\n------\n',Xm
#    print 'X=',X
#    print 'unfolded(1)=',unfold(X, 1)
#    print 'KR=',khatri_rao([C, A, B], skip_matrix=0)
    print 'Tly 1:\n-----\n',np.dot(unfold(X, 1), khatri_rao([C, B, A], skip_matrix=0))

    print ''
    Xm = unfolded_3D_matrix_multiply (2, X, B, A, I, J, K, R)
    #ans =
    #   280   472
    #   232   388
    print 'UMM 2:\n------\n',Xm
#    print 'X=',X
#    print 'unfolded(2)=',unfold(X, 2)
#    print 'KR=',khatri_rao([A, C, B], skip_matrix=1)
    print 'Tly 2:\n-----\n',np.dot(unfold(X, 2), khatri_rao([B, C, A], skip_matrix=1))

    print ''
    Xm = unfolded_3D_matrix_multiply (3, X, B, A, I, J, K, R)
    #ans =
    #   116   208
    #   308   544
    # or.....
    #   140   232
    #   332   568
    print 'UMM 3:\n------\n',Xm
#    print 'X=',X
#    print 'unfolded(0)=',unfold(X, 0)
#    print 'KR=',khatri_rao([A, B, C], skip_matrix=2)
    print 'Tly 3:\n-----\n',np.dot(unfold(X, 0), khatri_rao([A, B, C], skip_matrix=2))

