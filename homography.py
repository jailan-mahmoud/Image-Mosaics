import numpy as np
import cv2

def convertToHomogeneous(points):
    heterogeneous = np.array(points)
    n=heterogeneous.shape[0]
    homogeneous = np.append(heterogeneous, np.ones((n,1)), axis = 1)  
    return homogeneous

def convertToHeterogeneous(points):
    for i in range(len(points)):
        points[i]/=points[i][2]
    return points[0:2]

class Homography():
    def __init__(self,src_points,dst_points):
        self.points_1 = convertToHomogeneous(src_points)
        self.points_2 = convertToHomogeneous(dst_points)
        n = self.points_1[0]
        self.A = np.array([],dtype=float)
    
    def createAi(self, point_1,point_2):

        x = point_1[0]
        y = point_1[1]
        x_= point_2[0]
        y_= point_2[1]

        ai = np.array([
            [-x,-y,-1,0,0,0,x*x_,y*x_,x_],
            [0,0,0,-x,-y,-1,x*y_,y*y_,y_]
        ])
        return ai
    
    def createA(self):
        for i in range(len(self.points_1)):
            ai  = self.createAi(self.points_1[i],self.points_2[i])
            self.A= np.concatenate((self.A,ai),axis=0) if self.A.size else ai
    
    def solve(self):
        #1. Create matrix A
        self.createA()

        #2. Compute SVD (Singular Value Decomposition)
        U, D, Vt = np.linalg.svd(self.A, full_matrices=True)

        #3. Store singular vector of the smallest singular value
        min_i = np.argmin(D) 
        singular_vector = Vt[min_i] # column of V corresponding to the smallest singular value (row of Vt)
        h = singular_vector

        #4. Reshape to get H
        self.H = h.reshape((3, 3))
        return self.H
        

    def verifyH(self,imgs):
        i=0
        print("From Image 1 to Image 2: x'=Hx")
        for x in self.points_1:
            x_ = np.dot(self.H,x)
            x_/=x_[2]
            x_ = x_[0:2]
            print(f'{x_} vs {self.points_2[i][0:2]}')
            i+=1
            # displaying a point 
            x_ = np.around(x_).astype(int)
            cv2.circle(imgs[1], (x_[0],x_[1]), radius=3, color=(0, 255, 0), thickness=1)
        i=0
        print("From Image 2 to Image 1: x=H-1x'")
        invH = np.linalg.inv(self.H) 
        for x in self.points_2:
            x_ = np.dot(invH,x)
            x_/=x_[2]
            x_ = x_[0:2]
            print(f'{x_} vs {self.points_1[i][0:2]}')
            i+=1
            # displaying a point
            x_ = np.around(x_).astype(int) 
            cv2.circle(imgs[0], (x_[0],x_[1]), radius=3, color=(0, 255, 0), thickness=1)
        
        cv2.imshow('From Image 1 to Image 2', imgs[1])  
        cv2.imshow('From Image 2 to Image 1', imgs[0])  
        cv2.waitKey(0)

