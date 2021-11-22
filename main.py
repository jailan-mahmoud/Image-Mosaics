from images import Images

if __name__ == '__main__':
    paths = ['./dataset/image1.jpg','./dataset/image2.jpg']
    imgs = Images(paths)
    points = imgs.show()
