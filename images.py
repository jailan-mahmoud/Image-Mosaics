
import cv2

# correspondence points lists
points_1 = []
points_2 = []

# Mouse click callback function
def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:

            # displaying the coordinates
            # on the Shell
            # print(x, ' ', y)

            # Save point (x,y)
            if '1' in params['img_name']:
                points_1.append([x,y])
            elif '2' in params['img_name']:
                points_2.append([x,y])

            # displaying a point 
            cv2.circle(params['img'], (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow(params['img_name'], params['img'])    

# Plot the input images and get mouse click positions
class Images:

    def __init__(self,paths):
        self.points_1 = []
        self.points_2 = []
        self.imgs= []
        # check number of images
        if(len(paths)<2):
            raise Exception('2 images are required')
        # read images
        for path in paths:
            self.imgs.append(cv2.imread(path))
    
    def show(self):
        i=1
        for img in self.imgs:
            cv2.imshow('Image'+str(i), img)
            params = {
                "img": img,
                "img_name": 'Image'+str(i)
            }
            cv2.setMouseCallback('Image'+str(i), click_event, param=params)
            i+=1
        cv2.waitKey(0)
        print('Image 1 points:', points_1)
        print('Image 2 points:', points_2)
        return points_1, points_2

    
        
    
    

        
        

