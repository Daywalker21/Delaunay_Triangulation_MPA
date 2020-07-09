
# ## Important Python Libraries used
# >cv2, numpy, math, sys

# In[2]:


import cv2
import numpy as np
import math
import sys


# ### circumcenter(p1,p2,p3)
# 
# #### Use
# >To Calculate the circumcenter of the three points given. 
# 
# #### Arguments
# >This function takes 3 arguments as p1, p2 and p3.
# 
# #### return type 
# >coordinate of the the circumcenter of the point made by p1, p2 and p3.



def circumcenter(p1,p2,p3):
    
    #using the mathemathical formula for circumcenter for three points
    #Source for formula wikipedia
    
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * 
             (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    
    det_p1 = p1[0]**2 + p1[1]**2 
    det_p2 = p2[0]**2 + p2[1]**2
    det_p3 = p3[0]** 2 + p3[1]**2
    
    cx = (det_p1 * (p2[1] - p3[1]) 
          + det_p2 * (p3[1] - p1[1]) 
          + det_p3 * (p1[1] - p2[1])) / d
    
    cy = (det_p1 * (p3[0] - p2[0]) 
          + det_p2 * (p1[0] - p3[0]) 
          + det_p3 * (p2[0] - p1[0])) / d
    
    return (cx, cy)


# ### dist(p1,p2)
# 
# #### Use
# >To Calculate distance between two points 
# 
# #### Arguments
# >This function takes 2 arguments as p1 and p2.
# 
# #### return type 
# >float value of distance between p1 and p2 calculated using euclidean distance formula
#     
# ### area(x1,y1,x2,y2,x3,y3)
# #### Use
# >To Calculate Area of triangle
# 
# #### Arguments
# >This function takes 4 arguments such that (x1,y1), (x2,y2), (x3,y3) are co-ordinates of the triangle.
# 
# #### return type 
# >float value of area of triangle


def dist(p1,p2):
    
    # Euclidean Distance formula
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

def area(x1, y1, x2, y2, x3, y3): 
    # formula to calculate area of triangle
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


# ### delaunay(pts,coord)
# 
# #### Use
# >To check whether the three points passed in pts form a triangle according to delaunay condition or not 
# 
# #### Arguments
# >This function takes 2 arguments as pts and coord.<br>
# >- pts - contains three points that represent the coordinates of the triangle on which the condition is to be checked.<br>
# >- coord - all the points the are input by the user as well as the boundary points.
# 
# #### return type 
# >boolean value - whether the passed triangle points fulfill the delauny conditon or not<br>
# 
# >- True - Fulfiled<br>
# >- False - Not Fulfiled



def delaunay(pts,coord):
    
    # Checking weather the points are collinear or not
    area_tri = area(pts[0][0],pts[0][1],pts[1][0],pts[1][1],pts[2][0],pts[2][1])
    if area_tri == 0:
        return False
        
    # Finding the circumcenter of the selected points for triangle
    c = circumcenter(pts[0],pts[1],pts[2])
    
    #radius of the circumcircle by p1,p2,p3
    radius = dist(c,pts[0])

    #Getting the distance of remaining points and checking delaunay condition
    
    for i in range(len(coord)):
        if coord[i] not in pts:
            d = dist(c,coord[i])
            if d < radius:
                return False
    
    return True



# ### combinations(coord)
# 
# #### Use
# >To generate the combinations of points to create triangles.
# 
# #### Arguments
# >This function takes 1 arguments coord<br>
# > - coord - the list of points provided by the user for the delaunay triangulation.
#     
# #### return type 
# >returns the list of all the combinations of the points.
#     


def combinations(coord):
    comb = []
    for i in range(len(coord)):
        for j in range(len(coord)):
            for k in range(len(coord)):
                if(i!=j&j!=k&k!=i):
                    t = (coord[i],coord[j],coord[k])
                    comb.append(t)
    return comb
                    


# ### get_Triangle_List(coord)
# 
# #### Use
# >To check which combinations of points satisfy the delaunay condition. 
# 
# #### Arguments
# >This function takes 1 arguments coord<br>
# > - coord - the list of points provided by the user for the delaunay triangulation.
#     
# #### return type 
# >returns the list of the combinations of the points that satisfy the delaunay condition
#     


def get_Triangle_List(coord):  
    comb = combinations(coord)
    triangle_list = []
    for i in comb:
        cond = delaunay(i,coord)
        if cond == True:
            triangle_list.append(i)
            
    return triangle_list

# ### CallBackFuncForimg1 
# >To get callback coordinates and draw circle at points clicked in source image which are the control points in source image 
# 
# ### CallBackFuncForimg2 
# >To get callback coordinates and draw circle at points clicked in destination image which are the control points in source image
# 
# ### getcoord
# >To get the coordinates from the user using left mouse button click and storing that values in a list. One can select as many control points along with border points but it should be greater then 3


def CallBackFuncForimg1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im1, (x,y), 1, (0, 0, 255), 2)
        coordSrc.append((y,x))

def CallBackFuncForimg2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im2, (x,y), 1, (255, 0, 0), 2)
        coordDest.append((y,x))
        
def getcoord(window,image):
    while (True):
        cv2.imshow(window, image)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()

# ### draw_delauany(img,triangleList,delaunay_color)
# 
# #### Use
# >To display the valid delaunany triangle in the image.
# 
# #### Arguments
# >This function takes 3 arguments as img, triangleList and delaunay_color<br>
# >- img - The image on which we have to draw the triangles<br>
# >- triangleList - The list coordinates of the valid triangles.<br>
# >- delauany_color - the color of the lines of the triangles.
# 
# #### return type 
# >returns the image having the triangle.
#     


def draw_delaunay(img, triangleList,delaunay_color):
    tri=[]
    
    for t in triangleList :
        
        pt1 = t[0]
        pt2 = t[1]
        pt3 = t[2]

        cv2.line(img, (pt1[1],pt1[0]), (pt2[1],pt2[0]), delaunay_color, 1)
        cv2.line(img, (pt2[1],pt2[0]), (pt3[1],pt3[0]), delaunay_color, 1)
        cv2.line(img, (pt3[1],pt3[0]), (pt1[1],pt1[0]), delaunay_color, 1)
        a=[]
        a.append(pt1)
        a.append(pt2)
        a.append(pt3)
        tri.append(a)
    return tri


# ### showTriangulated(img1,img2)
# 
# #### Use
# >To display the valid delaunany triangle in the image.
# 
# #### Arguments
# >This function takes 2 arguments as img1 and img2<br>
# >- img1 = source image<br>
# >- img2 = destination image
# 
# #### return type 
# >returns the image having the triangle for further process.
#     


def showTriangulated(img1,img2):
    size = img1.shape
    r = (0, 0, size[1], size[0])
    
    coord = coordSrc.copy()
    
    triangleList = get_Triangle_List(coord)
    
    tri1 = draw_delaunay(img1,triangleList,(255,0,0))
    
    # Matching the point p0,..,pn of the source and destination image
    tri2 = []
    for i in range(len(tri1)):
        a = []
        for j in range(len(tri1[i])):
            a.append(coordDest[coordSrc.index(tri1[i][j])])
        tri2.append(a)
        
    tri2 = draw_delaunay(img2,tri2,(0,255,255))
    
    cv2.imshow("src",img1)
    cv2.imshow("dest",img2)
    cv2.imwrite("src.jpg",img1)
    cv2.imwrite("dest.jpg",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return tri1,tri2


# ### isInsideTriangle(p1,p2,p3,x,y)
# 
# #### Use
# >To check if some point lie inside the triangle or not. 
# 
# #### Arguments
# >This function takes 8 arguments such that (x1,y1), (x2,y2), (x3,y3) are co-ordinates of the triangle and (x,y) is the point which we want to check.
# 
# #### return type 
# >bool value<br>
# >- True - point lies inside the triangle.<br>
# >- False - point doesn't lie inside the triangle.


def isInsideTriangle(p1,p2,p3,x,y):
 
    A = area (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    A1 = area (x, y, p2[0], p2[1], p3[0], p3[1])  
    A2 = area (p1[0], p1[1], x, y, p3[0], p3[1])  
    A3 = area (p1[0], p1[1], p2[0], p2[1], x, y)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False


# ### get_affine_basis(coord)
# 
# #### Use
# >To Calculate the affine basis
# 
# #### Arguments
# >This function takes only 1 argument which contains the co-ordinates of the triangle 
# 
# #### return type 
# >float value of x and y component of both the affine basis of a triangle
#     


def get_affine_basis(coord):
    e1x = coord[1][0]-coord[0][0]
    e1y = coord[1][1]-coord[0][1]
    e2x = coord[2][0]-coord[0][0]
    e2y = coord[2][1]-coord[0][1]
    return e1x,e1y,e2x,e2y


# ### get_intermediate_triangles(srcTri,destTri,k,n)
# 
# #### Use
# >To find the co-ordinates of the triangle in kth intermediate image corresponding to the triangle in Source and Destination Image.
# 
# #### Arguments
# >This function take 4 arguments <br>
# >- srcTri - coordinates of triangle in source image<br>
# >- destTri - coordinates of triangle in destination image<br>
# >- k - kth intermediate immage<br>
# >- n - k+2
# 
# #### return type
# >return the co-ordinates of the triangle in intermediate image by calculating it as:
#     
# <!-- $ \mathbf{Pk}= \left( \frac{n-k}{n} \right) \mathbf{P1}+\left(\frac{k}{n}\right)\mathbf{P2}$
# 
# $\mathbf{Pk}$ is calculated coordinate of triangle in intermediate kth image <br>
# $\mathbf{P1}$ is triangle coordinate in Source image<br>
# $\mathbf{P2}$ is triangle coordinate in Destination image      -->
# 


def get_intermediate_triangles(srcTri , destTri , k , n):
    intTri=[]
    for (st,dt) in zip(srcTri,destTri):
        a=[]
        for (coordS,coordD) in zip(st,dt):
            
            xi=int(((n-k)/n)*coordS[0]+(k/n)*coordD[0])
            yi=int(((n-k)/n)*coordS[1]+(k/n)*coordD[1])
            a.append((xi,yi))
        intTri.append(a)
    return intTri


# ### checkRange(sx,sy,dx,dy)
# 
# #### Use
# >if sx,sy,dx,dy are out of range i.e if they are negative or greater than the size of image so this function normalize them
# 
# #### Arguments
# >This function take 4 arguments <br>
# >- (sx,sy) - coordinate in source image
# >- (dx,dy) - coordinate in destination image
# 
# #### return type
# >return the normalize co-ordinates    



def checkRange(sx , sy , dx , dy):
    if sx<0:
        sx=0
    if dx<0:
        dx=0
    if sy<0:
        sy=0
    if dy<0:
        dy=0
    if sx>img1.shape[0]-1:
        sx=img1.shape[0]-1
    if dx>img2.shape[0]-1:
        dx=img2.shape[0]-1
    if sy>img1.shape[1]-1:
        sy=img1.shape[1]-1
    if dy>img2.shape[1]-1:
        dy=img2.shape[1]-1
    return sx,sy,dx,dy

# ### morph(no_of_intermed)
# 
# #### Use
# >To do affine Transformation from source image to destination image by making some intermediate images in which pixel value are calculated by combination of pixel value in source and destination image
# 
# #### Arguments
# >This function take only 1 argument which is the how many number of intermediate images we want to make.
# 
#     


def morph(no_of_intermed):
    n=no_of_intermed+2
    
    for k in range(1,no_of_intermed+1):
        
        print(str(k)+" intermediate is generating it may take some time Please Wait...")
        inter=np.zeros_like(img1,dtype=np.uint8)
        row,col,channel=inter.shape

        intTri=get_intermediate_triangles(tri1,tri2,k,n)

        for ( s_tri , i_tri , d_tri ) in zip( tri1 , intTri , tri2 ):

            src_e1x , src_e1y , src_e2x , src_e2y = get_affine_basis(s_tri)
            int_e1x , int_e1y , int_e2x , int_e2y = get_affine_basis(i_tri)
            dest_e1x , dest_e1y , dest_e2x , dest_e2y = get_affine_basis(d_tri)

            for r in range(row):
                for c in range(col):
                    if isInsideTriangle(i_tri[0],i_tri[1],i_tri[2],r,c):
                        
                        X = r-i_tri[0][0]
                        Y = c-i_tri[0][1]

                        alpha=((int_e2y*X)-(Y*int_e2x))/((int_e1x*int_e2y)-(int_e2x*int_e1y))
                        beta=((int_e1y*X)-(Y*int_e1x))/((int_e1y*int_e2x)-(int_e2y*int_e1x))

                        dest_x=int(alpha*dest_e1x+beta*dest_e2x+d_tri[0][0])
                        dest_y=int(alpha*dest_e1y+beta*dest_e2y+d_tri[0][1])

                        src_x=int(alpha*src_e1x+beta*src_e2x+s_tri[0][0])
                        src_y=int(alpha*src_e1y+beta*src_e2y+s_tri[0][1])

                        src_x,src_y,dest_x,dest_y=checkRange(src_x,src_y,dest_x,dest_y)

                        inter[r][c][0]=int(((n-k)/n)*img1[src_x][src_y][0]
                                           +(k/n)*img2[dest_x][dest_y][0])
                        inter[r][c][1]=int(((n-k)/n)*img1[src_x][src_y][1]
                                           +(k/n)*img2[dest_x][dest_y][1])
                        inter[r][c][2]=int(((n-k)/n)*img1[src_x][src_y][2]
                                           +(k/n)*img2[dest_x][dest_y][2])

#         cv2.imshow("inter"+str(k),inter)
        name="inter"+str(k)+".jpg"
        cv2.imwrite(name, inter) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# # Reading of input images and resizing them to same size

# img1=cv2.imread("bush.jpg")
# img2=cv2.imread("clinton.jpg")

img1=cv2.imread(str(sys.argv[1]))
img2=cv2.imread(str(sys.argv[2]))
img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))
im1=np.copy(img1)
im2=np.copy(img2)
window1 = 'image1'
window2= 'image2'
coordSrc=[]
coordDest=[]


# # Getting control points on images using mouse click

cv2.namedWindow(window1)
cv2.setMouseCallback(window1, CallBackFuncForimg1)
getcoord(window1,im1)

cv2.namedWindow(window2)
cv2.setMouseCallback(window2, CallBackFuncForimg2)
getcoord(window2,im2)

r1,c1,ch1= img1.shape
r2,c2,ch2 = img2.shape


# # Triangulating the images and applying affine transformation

tri1,tri2 = showTriangulated(im1,im2)
morph(int(input("Enter number of intermediate you want ")))



