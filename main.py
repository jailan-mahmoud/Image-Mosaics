from ImageStitcher import ImageStitcher
from images import Images
from ImageStitcher import ImageStitcher

if __name__ == '__main__':
    paths = ['./dataset/image1.jpg','./dataset/image2.jpg']
    stitcher = ImageStitcher(paths)
    stitcher.solve()

