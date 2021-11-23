from images import Images
from homography import Homography

class ImageStitcher:

    def __init__(self,paths):
        self.imgs = Images(paths)
    
    def getCorrespondences(self):
        return self.imgs.show()
    
    def computeH(self,points):
        homography = Homography(points)
        H = homography.solve()
        homography.verifyH(self.imgs.getimgs())
        return H
    
    def solve(self):
        points = self.getCorrespondences()
        H = self.computeH(points)
        
    