import cv2
import matplotlib.pyplot as plt
import numpy as np

class Combine:
    
    def combine(self,left_image,right_image,offset):

        offsetx,offsety = offset

        # Create empty image that can hold both images
        # output = np.zeros((max(left_image.shape[0],right_image.shape[0]), right_image.shape[1]+offsetx,3)) 
        output = np.zeros((left_image.shape[0]+right_image.shape[0], left_image.shape[0]+right_image.shape[1],3))      

        ## Overlay right image on top of left image
        # output[0:left_image.shape[0],0:left_image.shape[1]] = left_image
        # output[offsety:right_image.shape[0]+offsety, offsetx:right_image.shape[1]+offsetx] = right_image
        
        ## Overlay left image on top of right image
        output[0:left_image.shape[0],0:left_image.shape[1]] = left_image
        i = 0
        for x in range(offsety,right_image.shape[0]+offsety):
            j=0
            for y in range(offsetx,right_image.shape[1]+offsetx):
                if(output[x][y].all()==0):
                    output[x][y]=right_image[i][j]
                j+=1
            i+=1

        output = output.astype(np.uint8)

        return output
    

    def stitchImages(self,image, warped_image, shiftX, shiftY, nChannels = 3):

        # Create empty image that can hold both images
        output = np.zeros(((image.shape[0] + warped_image.shape[0]), (image.shape[1] + warped_image.shape[1]), nChannels))
        
        # Overlay Images (warped image on top of other image)
        output[0:warped_image.shape[0],0:warped_image.shape[1]] = warped_image
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                newX, newY = col + shiftX , row + shiftY
                if(output[newY][newX].all()==0):
                    output[newY][newX] = image[row][col]

        # Delete black rows & columns (Cropping)
        idx = np.argwhere(np.all(output[..., :] == 0, axis=0))
        output = np.delete(output, idx, axis=1)
        idx = np.argwhere(np.all(output[..., :] == 0, axis=1))
        output = np.delete(output, idx, axis=0)

        # rotate image columns if shift was -ve 
        if shiftX < 0:
            output = np.roll(output, -1* shiftX, axis=1)
        if shiftY < 0:
            output = np.roll(output, -1* shiftY, axis=0)
        
        output = output.astype(np.uint8)
        return output

