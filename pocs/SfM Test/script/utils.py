import numpy as np 
import cv2
import pickle
# In Python 3, zip is a built-in function, no need to import from itertools
# Removed: from itertools import izip

def SerializeKeypoints(kp):     
    serialized_kp = []
    
    for curr_kp in kp:
        temp = (curr_kp.pt, curr_kp.size, curr_kp.angle, curr_kp.response, curr_kp.octave, 
                curr_kp.class_id)
        serialized_kp.append(temp)
    
    return serialized_kp

def DeserializeKeypoints(serialized_kp): 
    kp = []
    
    for temp in serialized_kp:
        curr_kp = cv2.KeyPoint(x=temp[0][0],y=temp[0][1],size=temp[1], angle=temp[2],
                                response=temp[3], octave=temp[4], class_id=temp[5]) 
        kp.append(curr_kp)
        
    return kp

def SerializeMatches(matches): 
    serialized_matches = []
    
    for curr_match in matches:
        temp = (curr_match.queryIdx, curr_match.trainIdx, curr_match.imgIdx, 
                curr_match.distance) 
        serialized_matches.append(temp)
    
    return serialized_matches

def DeserializeMatches(serialized_matches): 
    matches = []
    
    for temp in serialized_matches:
        # Use positional arguments instead of keyword arguments
        # OpenCV 4.11.0 has a different DMatch constructor
        curr_match = cv2.DMatch(temp[0], temp[1], temp[2], temp[3])
        matches.append(curr_match)
        
    return matches

def pts2ply(pts, colors, filename):     
    with open(filename, 'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        #pdb.set_trace()
        colors = colors.astype(int)
        for pt, cl in zip(pts, colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2], 
                                                cl[0], cl[1], cl[2]))
                                                
def DrawCorrespondences(img, ptsA, ptsB, ax, color1=(255,255,0), color2=(0,255,255)): 
    assert len(ptsA) == len(ptsB), 'Correspondence points must be 1-to-1'
    
    ax.imshow(img)
    ax.axis('off')
    ax.scatter(ptsA[:,0], ptsA[:,1], c=np.array([color1]), s=20)
    ax.scatter(ptsB[:,0], ptsB[:,1], c=np.array([color2]), s=20)
    
    for i in range(len(ptsA)): 
        ax.plot([ptsA[i,0], ptsB[i,0]], [ptsA[i,1], ptsB[i,1]], color=(0,1,0),
                 linestyle='-', linewidth=.5)
        
    return ax