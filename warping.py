import numpy as np
from math import floor,ceil
import cv2
import matplotlib.pyplot as plt
from numpy.core.numeric import indices

# convert from image to matrix
def to_mtx(img):
    if(len(img.shape)==3):
        H,V,C = img.shape
        mtr = np.zeros((V,H,C), dtype='int')
    elif (len(img.shape)==2):
        H,V = img.shape
        mtr = np.zeros((V,H), dtype='int')
    for i in range(img.shape[0]):
        mtr[:,i] = img[i]
    
    return mtr

# convert from matrix to image
def to_img(mtr):
    V,H,C = mtr.shape
    img = np.zeros((H,V,C), dtype='int')
    for i in range(mtr.shape[0]):
        img[:,i] = mtr[i]
        
    return img

class Warping():
    def __init__(self,src_img,M):
        self.M = M
        self.src_img =src_img
        self.src_mtx = to_mtx(src_img)
    
    def within_bounds(self,i,j):
        if i<0 or i>=self.warped_mtx.shape[0]:
            return False
        if j<0 or j>=self.warped_mtx.shape[1]:
            return False    
        return True
    
    def splatting(self, i,j, p_):
        x = p_[0]
        y = p_[1]

        points = [
            [floor(x),floor(y)],
            [floor(x),ceil(y)],
            [ceil(x),floor(y)],
            [ceil(x),ceil(y)]
        ]
        points = np.array(points).astype(int)

        for point in points:
            x = point[0]
            y = point[1]
            if(self.within_bounds(x,y)): #if valid point
                for d in range(3):
                    if(self.warped_mtx[x][y][d]==0):
                        self.warped_mtx[x][y][d]= self.src_mtx[i][j][d]
                    else:
                        self.warped_mtx[x][y][d] += self.src_mtx[i][j][d]
                        self.warped_mtx[x][y][d] /= 2
    
    def get_offset_and_dimensions(self):

        r = self.src_mtx.shape[0]
        c = self.src_mtx.shape[1]

        upper_left_corner = np.array([0,0,1])
        upper_right_corner = np.array([0,c,1])
        lower_left_corner = np.array([r,0,1])
        lower_right_corner = np.array([r,c,1])

        H = self.M

        # corresponding upper_left_corner
        c1 = np.dot(H , upper_left_corner)
        c1 = c1/c1[-1]
        c1 = c1[0:2]
        
        # corresponding upper_right_corner
        c2 = np.dot(H , upper_right_corner)
        c2 = c2/c2[-1]
        c2 = c2[0:2]
        
        # corresponding lower_left_corner
        c3 = np.dot(H , lower_left_corner)
        c3 = c3/c3[-1]
        c3 = c3[0:2]

        # corresponding lower_right_corner
        c4 = np.dot(H , lower_right_corner)
        c4 = c4/c4[-1]
        c4 = c4[0:2]

        # from c1 and c2: offset x
        offsetx = min(c1[0],c2[0])

        # from c1 and c3: offset y
        offsety = min(c1[1],c3[1])

        offset = [offsetx,offsety]

        # from c3 and c4 -> get x dimension
        dx = max(c3[0],c4[0])

        # from c2 and c4 -> get y dimension
        dy = max(c2[1],c4[1])

        # corners 
        # print(c1,'',c2,'\n',c3,'',c4)
        self.corners = np.array([
            [(c1[0],c1[1]),(c2[0],c2[1]),(c4[0],c4[1]),(c3[0],c3[1])]
        ])
        self.corners -= offset
        self.corners = self.corners.astype('int32')

        return offset, dx, dy
        
    
    def forward_warp(self):

        offset,dx,dy = self.get_offset_and_dimensions()

        # offsetx = abs(int(offset[0]))
        # offsety = abs(int(offset[1]))
        offsetx = int(offset[0])*-1
        offsety = int(offset[1])*-1
        
        # Create empty warped image
        self.warped_mtx= np.zeros((int(dx)+offsetx, int(dy)+offsety,3), dtype=np.uint16)

        rows = self.src_mtx.shape[0]
        cols = self.src_mtx.shape[1]

        for i in range(rows):
            for j in range(cols):
                # 1. get new pixel location
                p_ = np.dot(self.M,np.array([i,j,1])) 

                # 2. convert to heterogeneous 
                p_/=p_[2]
                p_ = p_[0:2]
                p_ -= offset

                # 3. splatting
                self.splatting(i,j, p_)

        offset = [int(offset[0])*-1,int(offset[1])*-1]

        self.warped_img = to_img(self.warped_mtx).astype(np.uint8)
        return self.warped_img, offset


    def backward_warp(self):

        offset,dx,dy = self.get_offset_and_dimensions()
        
        offsetx = int(offset[0])*-1
        offsety = int(offset[1])*-1
        
        # Create an empty warped matrix
        self.warped_mtx= np.zeros((int(dx)+offsetx,int(dy)+offsety,3), dtype=np.uint16)

        rows = self.warped_mtx.shape[0]
        cols = self.warped_mtx.shape[1]

        # Inverse Homography Matrix
        invH = np.linalg.inv(self.M) 

        # Bounding Box as image
        mask_img = np.zeros((int(dy)+offsety, int(dx)+offsetx),dtype=np.uint16)
        cv2.fillPoly(mask_img, self.corners , (255,255,255))

        # Show Image Mask
        plt.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)) 
        plt.title('Bounding Box')      
        plt.show()

        # Convert to matrix
        mask_mtx = to_mtx(mask_img)
        mask_mtx = mask_mtx.astype(np.uint8)

        indices = np.argwhere(mask_mtx==255)

        for index in indices:
            i,j = index 

            # convert image coordinates to match that of image 2
            index  = np.around(index + offset).astype(int)
            x,y = index 
            # 1. get pixel location in source image
            p_ = np.dot(invH ,np.array([x,y,1])) 

            # 2. convert to heterogeneous 
            p_/=p_[2]
            p_ = p_[0:2]

            # 3. Linear Interpolation
            map_x = np.array([[p_[0]]], dtype=np.float32)
            map_y = np.array([[p_[1]]], dtype=np.float32)
            value = cv2.remap(self.src_img, map_x, map_y, cv2.INTER_LINEAR)
            self.warped_mtx[i][j] = value

        offset = [int(offset[0])*-1,int(offset[1])*-1]
        self.warped_img = to_img(self.warped_mtx).astype(np.uint8)

        return self.warped_img, offset
    
    def show_result(self):
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.src_img, cv2.COLOR_BGR2RGB))  
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2RGB))       
        plt.title('Warped Image')
        plt.show()

