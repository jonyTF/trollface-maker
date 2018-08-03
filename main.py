import sys 
import numpy as np 
import argparse
import cv2
import dlib
import imutils
from imutils import face_utils
import xml.etree.ElementTree as ET
import math

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
        parts.append((int(part.attrib['x']), int(part.attrib['y'])))

    return parts

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

    return angle

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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #ap.add_argument('-p', '--shape-predictor', required=False, help='path to facial landmark predictor')
    #ap.add_argument('-i', '--image', required=False, help='path to input image')
    #args = vars(ap.parse_args())

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        print('ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    if len(sys.argv) < 2:
        print('ERROR: Need to supply image parameter\nEx: python main.py IMAGE')
        sys.exit(1)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(sys.argv[1])
    image = imutils.resize(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    trollface_image = cv2.imread('img/TrollFace_s.jpg')
    trollface_image = cv2.cvtColor(trollface_image, cv2.COLOR_BGR2BGRA)
    output_image = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    image = np.float32(image)
    trollface_image = np.float32(trollface_image)

    image_morph = np.zeros((trollface_image.shape[0], trollface_image.shape[1], 4), dtype=trollface_image.dtype)

    landmarks = []
    for (i, rect) in enumerate(rects):
        #cv2.rectangle(output_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0))

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)
        landmarks.append(list(map(tuple,shape)))

        '''for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)'''
    
    for (i, points) in enumerate(landmarks):
        # Display face with open mouth if mouth opening is greater than 5
        diff = abs(points[66][1]-points[62][1])
        trollface_points = get_trollface_landmarks(True if diff > 5 else False)

        # Delaunay triangulation
        tris = calculateDelaunayTriangles(image, points)
        trollface_tris = calculateDelaunayTriangles(trollface_image, trollface_points)

        #drawDelaunayTriangles(trollface_image, trollface_tris, 'trolol', wait=False)
        #drawDelaunayTriangles(image, tris, 'image', wait=False)
        
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
                    morphTriangle(image, trollface_image, image_morph, t1, t2, t2, 0)

        #cv2.imshow('Output', np.uint8(image_morph))
        #cv2.waitKey(0)
        #cv2.imwrite('img/o%d.png' % (i), np.uint8(image_morph))

        # Put the trollface onto the original image
        image_morph_uint8 = np.uint8(image_morph)

        face_height_fact = 1.25
        face_height = int((face_height_fact+.25) * get_dist( (points[27][0], get_min_point(points, 1)[1]), points[8] ))
        face_width_fact = 1.25
        face_width = int((face_width_fact+.25) * get_dist(points[0], points[16]))
        image_morph_uint8 = cv2.resize(image_morph_uint8, (face_width, face_height))
        image_morph_uint8 = imutils.rotate_bound(image_morph_uint8, get_face_angle(points))

        x1 = int(get_avg_point(points[0], points[16])[0] - image_morph_uint8.shape[1]/2)
        y1 = int(points[30][1] - image_morph_uint8.shape[0]/2)
        x2 = x1 + image_morph_uint8.shape[1]
        y2 = y1 + image_morph_uint8.shape[0]

        # Fix x and y values if out of range
        t_x1 = 0
        t_y1 = 0
        t_x2 = image_morph_uint8.shape[1]
        t_y2 = image_morph_uint8.shape[0]
        
        if x1 < 0:
            t_x1 = x1*-1 # Only start drawing trollface x1*-1 pixels over from the start
            x1 = 0
        
        if x2 > output_image.shape[1]:
            t_x2 -= x2-output_image.shape[1]
            x2 = output_image.shape[1]

        if y1 < 0:
            t_y1 = y1*-1 # Only start drawing trollface y1*-1 pixels over from the start
            y1 = 0
        
        if y2 > output_image.shape[0]:
            t_y2 -= y2-output_image.shape[0]
            y2 = output_image.shape[0]


        alpha_m = image_morph_uint8[:, :, 3] / 255.0
        alpha_o = 1.0 - alpha_m
        
        for c in range(3):
            output_image[y1:y2, x1:x2, c] = alpha_m[t_y1:t_y2, t_x1:t_x2] * image_morph_uint8[t_y1:t_y2, t_x1:t_x2, c] + alpha_o[t_y1:t_y2, t_x1:t_x2] * output_image[y1:y2, x1:x2, c]

    img_path = 'img/output.jpg'
    cv2.imwrite(img_path, output_image)
    print('{"img_path": "%s"}' % (img_path))
    #output_image = imutils.resize(output_image, width=800)
    #cv2.imshow('Output', output_image)
    #cv2.imshow('Original', imutils.resize(np.uint8(image), width=800))
    #cv2.waitKey(0)
    