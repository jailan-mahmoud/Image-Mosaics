import numpy as np
import cv2
import matplotlib.pyplot as plt

class Match():
    
    def __init__(self):
        # Initiate ORB detector
        # self.orb = cv2.ORB_create()
        self.orb = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE)
        
        #SIFT descriptor
        #self.sift = cv2.xfeatures2d.SIFT_create()
  
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def getMatches(self,img1,img2,max_points=150):
 
        # find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        #Plot the features of the images
        f1 = cv2.drawKeypoints(img1,kp1,outImage = None,color=(0,255,0), flags=0)
        plt.imshow(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
        plt.title('Features of image1') 
        plt.show()

        f2 = cv2.drawKeypoints(img2,kp2,outImage = None,color=(255,0,0), flags=0)
        plt.imshow(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
        plt.title('Features of image2') 
        plt.show()

        #For matching, one can use either FLANN or BFMatcher,that is provided by opencv
        #matches = self.flann.knnMatch(des1,des2,k=2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)    
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:150] ,np.array([]), flags=2)
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)) 
        plt.title('Feature Matching')      
        plt.show()

        maxMatches = min(max_points,len(matches))
        matchedPointsCurrent=  np.array([kp1[matches[i].queryIdx].pt for i in range(maxMatches)])
        matchedPointsPrev = np.array([kp2[matches[i].trainIdx].pt for i in range(maxMatches)])

        return matchedPointsCurrent, matchedPointsPrev 
