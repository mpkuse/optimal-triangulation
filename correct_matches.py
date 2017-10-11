import yaml
import re
import cv2
import numpy as np
import code

import time

from helper_utils import *



def check_epipolar_constraint( F, ptA, ptB ):
    assert ptA.shape[0] == 2
    assert ptB.shape[0] == 2
    assert F.shape[0] == 3 and F.shape[1] == 3
    # To homogeneous cords
    ptA_homo = np.ones( (3,ptA.shape[1]) )
    ptB_homo = np.ones( (3,ptB.shape[1]) )
    ptA_homo[0:2,:] = ptA
    ptB_homo[0:2,:] = ptB

    sum_ = 0
    for i in range( ptA_homo.shape[1] ):
        xfx = np.matmul( np.matmul( np.transpose(ptA_homo[:,i]), F ), ptB_homo[:,i] )
        sum_ = sum_ + xfx*xfx
        # print i, 'x^T F xd = ', xfx
    print 'RMS Residue : ', np.sqrt(sum_ / ptA_homo.shape[1] )




#grad_at_S = make_gradient( S, _F, xl, xr )
# S : 5-vec
# F : 3x3
# xl, xr : 2-vector
def make_gradient( S, _F, xl, xr ):
    u = S[0]
    v = S[1]
    ud = S[2]
    vd = S[3]
    lam = S[4]

    f1=_F[0,0]
    f2=_F[0,1]
    f3=_F[0,2]
    f4=_F[1,0]
    f5=_F[1,1]
    f6=_F[1,2]
    f7=_F[2,0]
    f8=_F[2,1]
    f9=_F[2,2]

    x = xl[0]
    y = xl[1]
    xd = xr[0]
    yd = xr[1]

    L = np.zeros( 5 )
    L[0] = 2*(u-x) + lam*(f1*ud + f2*vd + f3)
    L[1] = 2*(v-y) + lam*(f4*ud + f5*vd + f6)
    L[2] = 2*(ud-xd) + lam*(f1*u + f4*v + f7)
    L[3] = 2*(ud-xd) + lam*(f2*u + f5*v + f8)
    L[4] = u*(f1*ud+f2*vd+f3) + v*(f4*ud+f5*vd+f6) + (f7*ud+f8*vd+f9)

    H = np.zeros( (5,5) )
    H[0,0] = 2
    H[0,1] = 0
    H[0,2] = lam*f1
    H[0,3] = lam*f2
    H[0,4] = f1*ud+f2*vd+f3

    H[1,0] = H[0,1]
    H[1,1] = 2
    H[1,2] = lam*f4
    H[1,3] = lam*f5
    H[1,4] = f4*ud+f5*vd+f6

    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    H[2,2] = 2
    H[2,3] = 0
    H[2,4] = f1*u+f4*v+f7

    H[3,0] = H[0,3]
    H[3,1] = H[1,3]
    H[3,2] = H[2,3]
    H[3,3] = 2
    H[3,4] = f2*u+f5*v+f8

    H[4,0] = H[0,4]
    H[4,1] = H[1,4]
    H[4,2] = H[2,4]
    H[4,3] = H[3,4]
    H[4,4] = 0

    return L, H



#curr, prev, curr_m
node_nu = (532, 23, 530)


ix_curr   = node_nu[0]
ix_prev   = node_nu[1]
ix_curr_m = node_nu[2]

FILE_BASE = 'drag/pg_%d_%d___%d' %(ix_curr, ix_prev, ix_curr_m )



FILE_NAME = FILE_BASE+'.opencv'
print 'Loading : ', FILE_NAME
Q = readYAMLFile(FILE_NAME)
Q['curr_im'] = cv2.imread( 'drag/%d.png' %(ix_curr))
Q['prev_im'] = cv2.imread( 'drag/%d.png' %(ix_prev))
Q['curr_m_im'] = cv2.imread( 'drag/%d.png' %(ix_curr_m))

for name in Q.keys():
    print name , Q[name].shape
np.set_printoptions(precision=4)


_K = Q['K']
_D = Q['D']
_F = Q['F_c_cm']

ptA = Q['undist_normed_curr']   #2xN
ptB = Q['undist_normed_curr_m'] #2xN






# Optimize ptA and ptB
startTime = time.time()
ptA_opt = np.zeros( ptA.shape )
ptB_opt = np.zeros( ptB.shape )
print 'ptA.shape : ', ptA.shape
init_lambda = .1
for i in range( ptA.shape[1] ): # do this optimization for each point independently
    xl = ptA[:,i] #2-vector
    xr = ptB[:,i] #2-vector

    S = [ xl[0], xl[1], xr[0], xr[1], init_lambda ]
    grad_at_S, hessian_at_S = make_gradient( S, _F, xl, xr )

    # Newton's Method
    S_new = S - np.matmul( np.linalg.inv( hessian_at_S ) , grad_at_S )

    ptA_opt[0,i] = S_new[0]
    ptA_opt[1,i] = S_new[1]
    ptB_opt[0,i] = S_new[2]
    ptB_opt[1,i] = S_new[3]



print 'Elapsed: ', time.time() - startTime


# Epipolar Condition
check_epipolar_constraint( _F, ptA, ptB )
check_epipolar_constraint( _F, ptA_opt, ptB_opt )



# Try plotting these points on image - Checked OK!
ptA_dist = np.matmul( _K , distort_points( ptA, _D ) )
ptB_dist = np.matmul( _K , distort_points( ptB, _D ) )
ptA_opt_dist = np.matmul( _K , distort_points( ptA_opt, _D ) )
ptB_opt_dist = np.matmul( _K , distort_points( ptB_opt, _D ) )

print 'BLUE ARE ORIGINAL POINTS AND GREEN ARE CORRECTED MATCHES'
cv2.imshow( 'plot1', points_x2_overlay( Q['curr_im'],   ptA_dist, ptA_opt_dist ) )
cv2.imshow( 'plot2', points_x2_overlay( Q['curr_m_im'], ptB_dist, ptB_opt_dist ) )
cv2.waitKey(0)
