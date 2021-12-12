from ImageStitcher import ImageStitcher

if __name__ == '__main__':

    # Stitch 2 images
    paths = ['./dataset/image2.jpg','./dataset/image1.jpg'] # paths are in left_to_right order of orientation.
    # paths = ['./dataset/left.jpeg','./dataset/right.jpeg'] # paths are in left_to_right order of orientation.
    stitcher1 = ImageStitcher(paths)
    stitcher1.solve()

    ## Bonus (Stitch 3 images)
    paths = ['./dataset/shanghai-22.png','./dataset/shanghai-21.png','./dataset/shanghai-23.png'] # paths
    stitcher2 = ImageStitcher(paths)
    stitcher2.solve_multiple()

