import numpy as np

def convertToHomogeneous(points):
    heterogeneous = np.array(points)
    n=heterogeneous.shape[0]
    homogeneous = np.append(heterogeneous, np.ones((n,1)), axis = 1)  
    return homogeneous

def convertToHeterogeneous(points):
    pass

class Homography():
    def __init__(self,points):
        self.points_1 = convertToHomogeneous(points[0])
        self.points_2 = convertToHomogeneous(points[1])
        n = self.points_1[0]
        self.A = np.array([],dtype=float)
    
    def createAi(self, point_1,point_2):

        x = point_1[0]
        y = point_1[1]
        x_= point_2[0]
        y_=point_2[1]

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
        self.createA()
        print(self.A)


# TEST
points = [
    [
        [1,1],[2,2],[3,3]
    ],
    [
        [2,2],[4,4],[6,6]
    ]
]
homography = Homography(points)
homography.solve()
