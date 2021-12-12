import cv2
import numpy as np
import matplotlib.pyplot as plt
from images import Images
from homography import Homography
from warping import Warping
from combine import Combine
from match import Match

class ImageStitcher:

    def __init__(self,paths):
        self.paths = paths
        self.count = len(self.paths)
        self.left_list, self.right_list = [], []
        self.prepare_lists()
        
    def prepare_lists(self):
        self.center_idx = int(self.count/2)
        self.center_path = self.paths[int(self.center_idx)]
        for i in range(self.count):
            if(i<=self.center_idx):
                self.left_list.append(self.paths[i])
            else:
                self.right_list.append(self.paths[i])
    
    def getCorrespondences(self):
        return self.imgs.show()
    
    def computeH(self,src_points,dst_points):
        homography = Homography(src_points,dst_points)
        H = homography.solve()
        homography.verifyH(self.imgs.getimgs())
        return H
    
    def warp(self,img,H):
        warping = Warping(img,H)
        # warped_image,offset = warping.forward_warp()
        warped_image,offset = warping.backward_warp()
        warping.show_result()
        return warped_image, offset
    
    def combine(self,warped_image,image,offset):
        combine = Combine()
        offsetx,offsety = offset
        output_img  = combine.stitchImages(image,warped_image,offsetx,offsety)
        return output_img
    
    def show_output(self,output):
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  
        plt.title('Stitched Image')
        plt.show()
    
    def solve(self):

        # 1. Get correspondences
        self.imgs = Images(self.paths) 
        src_points,dst_points = self.getCorrespondences()
        src_points = [[783, 334], [816, 319], [891, 318], [982, 324], [779, 537], [757, 328], [890, 347], [999, 540]]
        dst_points = [[344, 302], [373, 290], [446, 296], [530, 309], [324, 505], [316, 295], [442, 321], [528, 509]]
        
        # 2. Compute Homography Matrix
        H = self.computeH(src_points,dst_points)

        # 3. Warping 
        warped_img, offset = self.warp(cv2.imread(self.paths[0]),H)

        # 4. Combine
        result = self.combine(warped_img,cv2.imread(self.paths[1]),offset)

        # 5. Show Result
        self.show_output(result)
    
    # Bonus
    def solve_multiple(self):

        #img to warp (I) and the output at the same time
        stitched_img = cv2.imread(self.paths[0])
        for i in range(1,self.center_idx+1): #warp stitched image into image

            # Read new destinatiom image I'
            dest_img = cv2.imread(self.paths[i])
            imgs = [stitched_img,dest_img]
            self.imgs = Images(None,imgs=imgs)

            # Get correspondences
            match=Match()
            matchedpoints1,matchedpoints2 = match.getMatches(stitched_img,dest_img)

            # Compute Homography Matrix
            H = self.computeH(matchedpoints1,matchedpoints2)

            # Warp Image
            warped_img, offset = self.warp(stitched_img,H)

            # Stitch images together
            result = self.combine(warped_img,dest_img,offset)
            self.show_output(result)
            stitched_img = result

        for i in range(self.center_idx+1,self.count): #warp image into stitched image
            
            # Read new image to warp 
            img_towarp = cv2.imread(self.paths[i])
            imgs = [img_towarp,stitched_img]
            self.imgs = Images(None,imgs=imgs)

            # Get correspondences
            match=Match()
            matchedpoints1,matchedpoints2 = match.getMatches(img_towarp,stitched_img)

            # Compute Homography Matrix
            H = self.computeH(matchedpoints1,matchedpoints2)

            # Warp Image
            warped_img, offset = self.warp(img_towarp,H)

            # Stitch images together
            result = self.combine(warped_img,stitched_img,offset)
            self.show_output(result)
            stitched_img = result

        
    