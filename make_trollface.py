import sys 
import numpy as np 
import argparse
import cv2
import dlib
import imutils
from imutils import face_utils
import xml.etree.ElementTree as ET
import math
import time

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def get_avg_point(p1, p2):
    x = abs(p1[0] + p2[0]) / 2
    y = abs(p1[1] + p2[1]) / 2 
    return (x, y)

def get_min_point(points, ind):
    min = points[0]
    for p in points:
        if p[ind] < min[ind]:
            min = p

    return min

def get_dist(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return math.sqrt(a*a + b*b)

def get_trollface_landmarks(mouthOpen):
    # Get facial landmark info from trollface picture
    tree = ET.parse('data/trollface_landmarks.xml')
    root = tree.getroot()

    image = list(root.iter('image'))[0 if mouthOpen else 1]

    parts = []
    for part in image.iter('part'):
        parts.append( (int(part.attrib['x']), int(part.attrib['y'])) )

    return np.array(parts)

def get_face_angle(points):
    p_top = points[27]
    p_bot = points[8]
    x = p_top[0] - p_bot[0]
    y = p_top[1] - p_bot[1]
    angle = math.atan2(y, x) * 180/math.pi

    if (p_top[1] > p_bot[1]):
        angle = 90 - angle
    else:
        angle = 90 + angle

    return int(angle)

def scale_rotate_point_with_image(point, img, size, angle):
    img_h, img_w = img.shape[0], img.shape[1]
    w, h = size

    # Scale point first
    w_ratio = w / img_w
    h_ratio = h / img_h
    scaled_point = (int(point[0] * w_ratio), int(point[1] * h_ratio))

    # Rotate the point by making it a point in an image, then use rotate_bound()
    point_image = np.zeros((h, w, 1), dtype=img.dtype)
    point_image[scaled_point[1], scaled_point[0], 0] = 255
    point_image = imutils.rotate_bound(point_image, angle)
    point_index = np.where(point_image > 0)
    orig_point = (point_index[1][0], point_index[0][0])

    return orig_point

def put_image_alpha(src, output, x, y):
    # Puts image src onto output at (x, y) using the image's alpha channels

    x1 = x 
    y1 = y
    x2 = x + src.shape[1]
    y2 = y + src.shape[0]

    # Fix x and y values if out of range
    s_x1 = 0
    s_y1 = 0
    s_x2 = src.shape[1]
    s_y2 = src.shape[0]
    
    if x1 < 0:
        s_x1 = x1*-1 # Only start drawing trollface x1*-1 pixels over from the start
        x1 = 0
    
    if x2 > output.shape[1]:
        s_x2 -= x2-output.shape[1]
        x2 = output.shape[1]

    if y1 < 0:
        s_y1 = y1*-1 # Only start drawing trollface y1*-1 pixels over from the start
        y1 = 0
    
    if y2 > output.shape[0]:
        s_y2 -= y2-output.shape[0]
        y2 = output.shape[0]


    alpha_m = src[:, :, 3] / 255.0
    alpha_o = 1.0 - alpha_m
    
    for c in range(3):
        output[y1:y2, x1:x2, c] = alpha_m[s_y1:s_y2, s_x1:s_x2] * src[s_y1:s_y2, s_x1:s_x2, c] + alpha_o[s_y1:s_y2, s_x1:s_x2] * output[y1:y2, x1:x2, c]


'''
def rotate_point_with_image(point, img, angle):
    # Rotates the point as if rotating an image with imutils.rotate_bound()
    angle = math.radians(angle) # Don't need to multiply by -1 because coordinate system is different (y goes down)
    h, w = img.shape

    # Rotate point around center of image
    ox = w/2
    oy = h/2
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
'''



def flip_landmarks(points, img):
    new_points = []
    new_points = [(img.shape[1]-p[0], p[1]) for p in points]

    swap = (
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), #outer face
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), #eyebrows
        (36, 45), (37, 44), (38, 43), (39, 42), #upper eyelid
        (40, 47), (41, 46), #lower eyelid
        (31, 35), (32, 34), #nose
        (48, 54), (49, 53), (50, 52), #upper outer lip
        (55, 59), (56, 58), #lower outer lip
        (60, 64), (61, 63), #upper inner lip
        (65, 67) #lower inner lip
    )

    for (p1,p2) in swap:
        temp = new_points[p1]
        new_points[p1] = new_points[p2]
        new_points[p2] = temp

    return new_points

def calculateDelaunayTriangles(img, points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for p in points:
        if (rect_contains(rect, p)):
            subdiv.insert(p)

    return subdiv.getTriangleList()

def drawDelaunayTriangles(img, tris, name, wait=True):
    for t in tris:
    
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if image_contains_tri(img, [pt1, pt2, pt3]):
        
            cv2.line(img, pt1, pt2, (255, 0, 0))
            cv2.line(img, pt2, pt3, (255, 0, 0))
            cv2.line(img, pt3, pt1, (255, 0, 0))

    cv2.imshow(name, np.uint8(img))
    if wait: cv2.waitKey(0)

def image_contains_tri(img, tri):
    s = img.shape
    r = (0, 0, s[1], s[0])

    return rect_contains(r, tri[0]) and rect_contains(r, tri[1]) and rect_contains(r, tri[2])

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 4), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img1Rect = cv2.cvtColor(img1Rect, cv2.COLOR_BGR2BGRA)
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    #img2Rect = cv2.cvtColor(img2Rect, cv2.COLOR_BGR2BGRA)

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def make_trollface(img_import_path, img_export_path, show_times=False):
    start = time.time()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
    end = time.time()
    if show_times: print("beginning thing time: ", end-start)
    
    start = time.time()
    image = cv2.imread(img_import_path)
    #image_orig_h, image_orig_w = image.shape[:2]
    #image = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    trollface_image = cv2.imread('data/TrollFace_s.jpg')
    trollface_image = cv2.cvtColor(trollface_image, cv2.COLOR_BGR2BGRA)
    #wrinkle_image = cv2.imread('data/wrinkle.png', cv2.IMREAD_UNCHANGED)
    
    # Copy the image and add an alpha channel
    output_image = image.copy()
    #output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    end = time.time()
    if show_times: print("Load image time: ", end-start)
    

    start = time.time()
    rects = detector(gray, 1)
    end = time.time()
    if show_times: print('Detector time: ', end-start)

    image = np.float32(image)
    trollface_image = np.float32(trollface_image)

    start = time.time()
    landmarks = []
    for (i, rect) in enumerate(rects):
        #cv2.rectangle(output_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0))

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)
        #landmarks.append(list(map(tuple,shape)))
        landmarks.append(shape)

        #for (x, y) in shape:
        #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    landmarks = np.array(landmarks)
    end = time.time()
    if show_times: print("Landmark recog time: ", end-start)

    start = time.time()
    for (i, points) in enumerate(landmarks):
        # Clear image_morph
        image_morph = np.zeros((trollface_image.shape[0], trollface_image.shape[1], 4), dtype=trollface_image.dtype)

        # Display face with open mouth if mouth opening is greater than 20% of mouth
        mouth_opening_height = get_dist(points[66],points[62])
        mouth_height = get_dist(points[51], points[57])
        mouth_opening_ratio = mouth_opening_height/mouth_height
        trollface_points = get_trollface_landmarks(True if mouth_opening_ratio > .2 else False)

        # Flip face if facing left
        flip = True if get_dist(points[0],points[27]) < get_dist(points[16],points[27]) else False
        if flip:
            trollface_points = flip_landmarks(trollface_points, trollface_image)
            trollface_image_flip = np.fliplr(trollface_image)
        else:
            trollface_image_flip = trollface_image

        # Delaunay triangulation
        # tris = calculateDelaunayTriangles(image, points)
        # trollface_tris = calculateDelaunayTriangles(trollface_image_flip, trollface_points)

        #drawDelaunayTriangles(np.uint8(trollface_image_flip), trollface_tris, 'trolol', wait=False)
        #drawDelaunayTriangles(image, tris, 'image', wait=True)
        
        with open('data/tri.txt') as file:
            for line in file:
                x, y, z = line.split()
                x = int(x)
                y = int(y)
                z = int(z)

                r = (0, 0, image.shape[1], image.shape[0])
                if x < 68 and y < 68 and z < 68 and rect_contains(r, points[x]) and rect_contains(r, points[y]) and rect_contains(r, points[z]):
                    t1 = [points[x], points[y], points[z]]
                    t2 = [trollface_points[x], trollface_points[y], trollface_points[z]]
                    morphTriangle(image, trollface_image_flip, image_morph, t1, t2, t2, 0)

        #cv2.imshow('Output', np.uint8(image_morph))
        #cv2.waitKey(0)
        #cv2.imwrite('img/o%d.png' % (i), np.uint8(image_morph))

        ## Put the trollface onto the original image
        image_morph_uint8 = np.uint8(image_morph)

        # Feather edges
        trollface_mask = cv2.imread('data/TrollFace_mask2.jpg', cv2.IMREAD_GRAYSCALE)
        if flip: trollface_mask = np.fliplr(trollface_mask)
        image_morph_uint8[:, :, 3] = trollface_mask
        
        # Set the size of trollface to cover the original face
        face_height_fact = 1.25
        face_height = int((face_height_fact) * (get_dist( points[27], points[8] ) + (image_morph_uint8.shape[0] - get_dist( trollface_points[27], trollface_points[8] ))  ) )
        face_width_fact = 1.5
        face_width = int((face_width_fact) * get_dist(points[0], points[16]))

        # Get the angle to rotate trollface to fit the angle of target face
        face_angle = get_face_angle(points)
        
        # Perform the actual scaling and rotation on the trollface image and 27th point and wrinkle
        transformed_point = scale_rotate_point_with_image(trollface_points[27], image_morph_uint8, (face_width, face_height), face_angle)
        image_morph_uint8 = cv2.resize(image_morph_uint8, (face_width, face_height))
        image_morph_uint8 = imutils.rotate_bound(image_morph_uint8, face_angle)
        #wrinkle_image = cv2.resize(wrinkle_image, (int(0.6*face_width), int(0.2*face_height)))
        #wrinkle_image = imutils.rotate_bound(wrinkle_image, face_angle)

        # Get x and y coordinates to place morphed trollface image
        x = points[27][0] - transformed_point[0]
        y = points[27][1] - transformed_point[1]

        # Seamless clone wrinkle on
        #mask = 255*np.ones(wrinkle_image.shape, dtype=wrinkle_image.dtype)
        #image_morph_uint8 = cv2.cvtColor(image_morph_uint8, cv2.COLOR_BGRA2BGR)
        #output_image = cv2.cvtColor(output_image, cv2.COLOR_BGRA2BGR)
        #center = (int(x1+image_morph_uint8.shape[1]/2), int(y1))
        #output_image = cv2.seamlessClone(wrinkle_image, output_image, mask, center, cv2.MIXED_CLONE)

        put_image_alpha(image_morph_uint8, output_image, x, y)
        #put_image_alpha(wrinkle_image, output_image, int(x+wrinkle_image.shape[1]*.4), int(y-wrinkle_image.shape[0]/2))
    end = time.time()
    if show_times: print("image morph time: ", end-start)
        
    img_path = img_export_path
    cv2.imwrite(img_path, output_image)
    print('{"trollface_count": "%s"}' % (len(landmarks)))
    #output_image = imutils.resize(output_image, width=800)
    #cv2.imshow('Output', output_image)
    #cv2.imshow('Original', imutils.resize(np.uint8(image), width=800))
    #cv2.waitKey(0)
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #ap.add_argument('-p', '--shape-predictor', required=False, help='path to facial landmark predictor')
    #ap.add_argument('-i', '--image', required=False, help='path to input image')
    #args = vars(ap.parse_args())

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        print('ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    if len(sys.argv) < 3:
        print('ERROR: Need to supply image parameters\nEx: python main.py IMAGE_PATH OUTPUT_PATH')
        sys.exit(1)

    start = time.time()
    show_times = False
    make_trollface(sys.argv[1], sys.argv[2], show_times=show_times)
    end = time.time()
    if show_times: print('Total Time: ', end-start)