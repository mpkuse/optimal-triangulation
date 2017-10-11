import yaml
import re
import cv2
import numpy as np
import code

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#### YAML READING ####
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

def readYAMLFile(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

######################





###### perspective projection / Geometry ################
# x must be in normalized image co-ordinates for this to work
def distort_points( x, D ):
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]

    xd = np.zeros( (3,x.shape[1]) )
    for j in range(x.shape[1]):
        _x = x[0,j]
        _y = x[1,j]

        r2 = _x*_x + _y*_y
        c = 1.0 + k1*r2 + k2*r2*r2

        xd[0,j] = _x *c + 2.*p1*_x*_y + p2*(r2+2.0*_x*_x)
        xd[1,j] = _y *c + 2.*p2*_x*_y + p1*(r2+2.0*_y*_y)
        xd[2,j] = 1.0

    return xd


# _3dpts : 4xN
# K : Camera Matrix (intrinsic) 3x3
# D : Distortion coef 4x1
# T : Optional 3x4 [R|t]. This is matrix multiplied with X
# [Returns]
# reproj_my_ideal : Reprojections (under ideal camera)
# reproj_my : Reprojections (under camera with specified distortions)
def perspective_project_3dpts( _3dpts, K, D, T=None   ):
    if T is None:
        T = np.identity(4)[0:3,:]
    X_normalized_cords = np.dot( T , _3dpts) #3xN
    X_normalized_cords = X_normalized_cords / X_normalized_cords[2,:]
    X_dist_normalized_cords = distort_points( X_normalized_cords, D )

    reproj_my_ideal = np.dot( K, X_normalized_cords )
    reproj_my       = np.dot( K, X_dist_normalized_cords )

    return reproj_my_ideal, reproj_my


# rvec : 3x1, tvec: 3x1. Returns a 4x4 pose matrix
def vec2pose( rvec, tvec ):
    R_rvec, jacbian = cv2.Rodrigues( rvec )
    Tr = np.eye(4)
    Tr[0:3,0:3] = R_rvec
    Tr[0:3,3:4] = tvec
    return Tr


# Given a 4x4 pose returns the fundamental matrix.
def make_fundamental_matrix_from_pose( T ):
    tx = T[0,3]
    ty = T[1,3]
    tz = T[2,3]
    ex = np.array( [ [0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0] ]  )
    R = T[0:3,0:3]
    F = np.dot( ex, R )

    # print "T:\n", T
    # print "ex:\n", ex
    # print "R:\n", R
    # print "F:\n", F
    return F

######################








#### Plotting #################################


# Image, pts 2xN
def points_overlay(im, pts, color=(255,0,0) ):
    if len(im.shape) == 2:
        im = cv2.cvtColor( im, cv2.COLOR_GRAY2BGR )

    if pts.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
        pts = pts / pts[2,:]

    for i in range( pts.shape[1] ):
        cv2.circle( im, tuple(np.int0(pts[0:2,i])), 3, color )
    return im



# Image, pts 2xN, pts 2xN. Overlays 2 point sets. Typically use to display projected pts and observed pts.
def points_x2_overlay(im, pts1, pts2, color1=(255,0,0), color2=(0,255,0) ):
    if len(im.shape) == 2:
        im = cv2.cvtColor( im, cv2.COLOR_GRAY2BGR )

    if pts1.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
        pts1 = pts1 / pts1[2,:]

    if pts2.shape[0] == 3: #if input is 3-row mat than  it is in homogeneous so perspective divide
        pts2 = pts2 / pts2[2,:]

    for i in range( pts1.shape[1] ):
        cv2.circle( im, tuple(np.int0(pts1[0:2,i])), 3, color1 )
        cv2.circle( im, tuple(np.int0(pts2[0:2,i])), 3, color2 )
        cv2.line( im,  tuple(np.int0(pts1[0:2,i])), tuple(np.int0(pts2[0:2,i])), color1 )
    return im

#################################################################



#### 3 way plotting ###########################################

# [Input]
# curr, prev, curr_m : Images
# pts_curr, pts_prev, pts_curr_m : 2xN each
def overlay_points_3way( curr, pts_curr, prev, pts_prev, curr_m, pts_curr_m, DEBUG=False):

    if len(curr.shape) == 2:
        curr = cv2.cvtColor( curr, cv2.COLOR_GRAY2BGR )
        prev = cv2.cvtColor( prev, cv2.COLOR_GRAY2BGR )
        curr_m = cv2.cvtColor( curr_m, cv2.COLOR_GRAY2BGR )

    zero_image = np.zeros( curr.shape, dtype='uint8' )
    cv2.putText( zero_image, str(pts_curr.shape[1]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


    r1 = np.concatenate( ( curr, prev ), axis=1 )
    r2 = np.concatenate( ( curr_m, zero_image ), axis=1 )
    gridd = np.concatenate( (r1,r2), axis=0 )

    for xi in range( pts_curr.shape[1] ):

        if DEBUG==True:
            zero_image = np.zeros( curr.shape, dtype='uint8' )
            cv2.putText( zero_image, str(xi)+' of '+str(pts_curr.shape[1]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


            r1 = np.concatenate( ( curr, prev ), axis=1 )
            r2 = np.concatenate( ( curr_m, zero_image ), axis=1 )
            gridd = np.concatenate( (r1,r2), axis=0 )

        pta = tuple( np.int0(pts_curr[:,xi]) )
        cv2.circle( gridd, pta, 4, (0,255,0) )

        ptb = tuple(   np.int0(pts_prev[:,xi]) + np.array([curr.shape[1],0])   )
        cv2.circle( gridd, ptb, 4, (0,255,0) )
        cv2.line( gridd, pta, ptb, (255,0,0) )


        ptc = tuple(    np.int0(pts_curr_m[:, xi]) + [0,curr.shape[0]]   )
        cv2.circle( gridd, ptc, 4, (0,255,0) )
        cv2.line( gridd, pta, ptc, (255,30,255) )

        if DEBUG==True:
            cv2.imshow( '_DEBUG_gridd', gridd )
            cv2.waitKey(0)

    return gridd


# [Input]
# curr, prev, curr_m : Images
# pts_curr, pts_prev, pts_curr_m : 2xN each
def overlay_points_3way_upscale( curr, pts_curr, prev, pts_prev, curr_m, pts_curr_m, DEBUG=False):
    f = 2

    curr_f = cv2.resize(curr, (0,0), fx=f, fy=f)
    prev_f = cv2.resize(prev, (0,0), fx=f, fy=f)
    curr_m_f = cv2.resize(curr_m, (0,0), fx=f, fy=f)


    if len(curr.shape) == 2:
        curr_f   = cv2.cvtColor( curr_f, cv2.COLOR_GRAY2BGR   )
        prev_f   = cv2.cvtColor( prev_f, cv2.COLOR_GRAY2BGR   )
        curr_m_f = cv2.cvtColor( curr_m_f, cv2.COLOR_GRAY2BGR )

    zero_image = np.zeros( curr_f.shape, dtype='uint8' )
    cv2.putText( zero_image,  str(pts_curr.shape[1]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


    r1 = np.concatenate( ( curr_f, prev_f ), axis=1 )
    r2 = np.concatenate( ( curr_m_f, zero_image ), axis=1 )
    gridd = np.concatenate( (r1,r2), axis=0 )


    font = cv2.FONT_HERSHEY_SIMPLEX
    for xi in range( pts_curr.shape[1] ):
        if DEBUG == True:
            zero_image = np.zeros( curr_f.shape, dtype='uint8' )
            cv2.putText( zero_image,  str(xi)+' of '+str(pts_curr.shape[1]), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2, 255 )


            r1 = np.concatenate( ( curr_f, prev_f ), axis=1 )
            r2 = np.concatenate( ( curr_m_f, zero_image ), axis=1 )
            gridd = np.concatenate( (r1,r2), axis=0 )



        pta = tuple( f*np.int0(pts_curr[:,xi]) )
        cv2.circle( gridd, pta, 4, (0,255,0) )
        cv2.putText( gridd, str(xi), pta, font, 0.4, (0,0,255) )

        ptb = tuple(  f* np.int0(pts_prev[:,xi]) + np.array([curr_f.shape[1],0])   )
        cv2.circle( gridd, ptb, 4, (0,255,0) )
        cv2.putText( gridd, str(xi), ptb, font, 0.4, (0,0,255) )
        # cv2.line( gridd, pta, ptb, (255,0,0) )


        ptc = tuple(   f* np.int0(pts_curr_m[:, xi]) + [0,curr_f.shape[0]]   )
        cv2.circle( gridd, ptc, 4, (0,255,0) )
        cv2.putText( gridd, str(xi), ptc, font, 0.4, (0,0,255) )
        # cv2.line( gridd, pta, ptc, (255,30,255) )

        if DEBUG==True:
            cv2.imshow( '_DEBUG_gridd', gridd )
            cv2.waitKey(0)

    return gridd




################################################
