from pylab import *
import cv2
import pickle
from glob import glob
import sys

start=int(sys.argv[1])
step=int(sys.argv[2])

print start,step

videos_dir="/run/shm/videos/"
file_list=glob(videos_dir+'*')

sub_list=[file_list[i] for i in range(start,len(file_list),step)]
print 'files to be processed:'
print '\n'.join(sub_list)
print

frontal='/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(frontal)

## Read videos and detect faces
output_dir=" /run/shm/output/"

for path in sub_list:
    start=path.find('videos/')+7
    filename=path[start:-4]
    print filename
    print "reading and detecting faces"
    vid = cv2.VideoCapture(videos_dir+filename+".mp4")
    i=0;
    Faces_List=[]
    while True:
        flag,frame=vid.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        Faces_List.append(faces)
        #print 'index=',i,'faces=',faces
        i+=1
        if i % 10 == 0:
            print '\r',i,

    pickle.dump(Faces_List,open(output_dir+filename+'_Faces.pkl','w'),protocol=2);