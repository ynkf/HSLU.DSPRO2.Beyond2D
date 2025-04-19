import numpy as np 
import cv2
import pickle
# In Python 3, zip is a built-in function, no need to import from itertools
# Removed: from itertools import izip

def SerializeKeypoints(kp): 
    """Serialize list of keypoint objects so they can be saved using pickle
    
    Args: 
        kp: List of keypoint objects 
        
    Returns: 
        serialized_kp: Serialized list of keypoint objects"""
    
    serialized_kp = []
    
    for curr_kp in kp:
        temp = (curr_kp.pt, curr_kp.size, curr_kp.angle, curr_kp.response, curr_kp.octave, 
                curr_kp.class_id)
        serialized_kp.append(temp)
    
    return serialized_kp

def DeserializeKeypoints(serialized_kp): 
    """Deserialize list of keypoint objects so they can be converted back to
    native opencv format
    
    Args: 
        serialized_kp: Serialized list of keypoint objects 
        
    Returns: 
        kp: List of keypoint objects"""
    
    kp = []
    
    for temp in serialized_kp:
        curr_kp = cv2.KeyPoint(x=temp[0][0],y=temp[0][1],size=temp[1], angle=temp[2],
                                response=temp[3], octave=temp[4], class_id=temp[5]) 
        kp.append(curr_kp)
        
    return kp

def SerializeMatches(matches): 
    """Serialize list of match objects so they can be saved using pickle
    
    Args: 
        matches: List of match objects 
        
    Returns: 
        serialized_matches: Serialized list of match objects"""
    
    serialized_matches = []
    
    for curr_match in matches:
        temp = (curr_match.queryIdx, curr_match.trainIdx, curr_match.imgIdx, 
                curr_match.distance) 
        serialized_matches.append(temp)
    
    return serialized_matches

def DeserializeMatches(serialized_matches): 
    """Deserialize list of match objects so they can be converted back to
    native opencv format
    
    Args: 
        serialized_matches: Serialized list of match objects 
        
    Returns: 
        matches: List of match objects"""
    
    matches = []
    
    for temp in serialized_matches:
        # Use positional arguments instead of keyword arguments
        # OpenCV 4.11.0 has a different DMatch constructor
        curr_match = cv2.DMatch(temp[0], temp[1], temp[2], temp[3])
        matches.append(curr_match)
        
    return matches

def pts2ply(pts, colors, filename): 
    """Saves an Nx3 points array to a .ply file
    
    Args: 
        pts: Nx3 float array of 3D points
        colors: Nx3 uint8 array of RGB colors
        filename: filename to save to (including .ply extension)
    """
    
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
    """Draw correspondence points on image
    
    Args: 
        img: image to draw on 
        ptsA: image A points
        ptsB: image B points 
        ax: matplotlib axis to draw on
        color1: color for image A points
        color2: color for image B points
        
    Returns: 
        ax: matplotlib axis with correspondences"""
    
    assert len(ptsA) == len(ptsB), 'Correspondence points must be 1-to-1'
    
    ax.imshow(img)
    ax.axis('off')
    ax.scatter(ptsA[:,0], ptsA[:,1], c=np.array([color1]), s=20)
    ax.scatter(ptsB[:,0], ptsB[:,1], c=np.array([color2]), s=20)
    
    for i in range(len(ptsA)): 
        ax.plot([ptsA[i,0], ptsB[i,0]], [ptsA[i,1], ptsB[i,1]], color=(0,1,0),
                 linestyle='-', linewidth=.5)
        
    return ax