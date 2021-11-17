import cv2
import numpy as np

## 

def deNormalizePoints(datas):
    """ 
        function to denormalize points
        input: 
            only json containing image data
        format of json:
            [{
                "id":,
                "type":,
                "value":{ 
                    "points":[
                        [x,y],
                        [],
                    ],
                    "polygonlabels":[ ]
                },
                "to_name": name,
                "from_name": name,
                "image_rotation": 0,
                "original_width": 1280,
                "original_height": 720
                }]
    """
    
    coordinates = []
    for data in datas:
    
        new_coord = []
    
        image_height = data['original_height']
        image_width = data['original_width']
    
        for x,y in data['value']['points'][:]:
            # formula used to denormalize
            # denormalized_x = (x*image_width)/100
            # denormalized_y = (y*image_height)/100
            new_coord.append([(x*image_width)/100,(y*image_height)/100 ])
    
        # changing coordinate float to int32
        new_coord = np.array(new_coord, np.int32)
        coordinates.append(new_coord)
    
    return coordinates

def polygonLabels(image_datas):
    """ function to get labels form image data """
    labels = []
    for data in image_datas:
        labels.append(data['value']['polygonlabels'][0])
    return labels

def maskImage(image, coordinates, alpha = 0.4, color = [(255, 0, 0),(0,255, 0),(0, 0,255),(255,255, 0),(0,255,255),(255, 0,255)]):
    """ function to mask the image 

        alpha:
            initially alpha is 0.4
        color:
            we have 6 colors which can be changed by input
    """
    # making 2 copies of img
    overlay = image.copy()
    img = image.copy()
    
    cnt = 0
    for point in coordinates:
        # this will draw solid polygon on the image
        cv2.fillPoly(overlay, [point], color[cnt%6])
        cnt+=1

    # blending overlay and img
    cv2.addWeighted(overlay, alpha, img,1-alpha, 0, img)
    
    cnt = 0
    for point in coordinates:
        # adding border to the polygon
        cv2.polylines(img, [point],True, color[cnt%6], 2)
        cnt+=1
    
    return img


def boundingBoxCoordinates(coordinates,height, width):
    """ function to get bounding box coordinates """
    # a blank image of size same as that of image
    blank_img = np.zeros((height, width, 3), np.uint8)
    
    color = (255,255,255)

    boundingBox = []
    for point in coordinates:
        blank_img[:,:] = (0,0,0)
        cv2.fillPoly(blank_img, [point], color)

        gray = cv2.cvtColor(blank_img,cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
        
        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            # saving the coordinates for bounding box as [x1,y1,x2,y2]
            boundingBox.append([x,y,x+w,y+h])
        
    return boundingBox


def DrawBox(img,coords,labels,font_scale = 1, font_thickness=1,color=[(255, 0, 0),(0,255, 0),(0, 0,255),(255,255, 0),(0,255,255),(255, 0,255)]):
    """ function to draw bounding box and put label
        color can be changed
    """
    num = 0
    n_img = img.copy()
    
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_color_bg = (0,0,0)
    for point in coords:
        # drawing bounding rectangle
        c1 = int(point[0]), int(point[1])
        c2 = int(point[2]), int(point[3])
        cv2.rectangle(n_img,c1,c2,color[num%6],2,2)
        
        # getting text size to fill text background this is set to black
        text_size, _ = cv2.getTextSize(labels[num],font,font_scale, font_thickness)
        text_w, text_h = text_size

        c1 = int(point[0]), int(point[1])
        c2 = int(point[0]+text_w+5), int(point[1]+text_h+10)
        cv2.rectangle(n_img, c1,c2,text_color_bg,-1)
        
        # putting text
        cv2.putText(n_img,labels[num],(point[0]+5,point[1]+18),font,font_scale,color[num%6],2)
        # print(labels[num])
        num+=1

    return n_img