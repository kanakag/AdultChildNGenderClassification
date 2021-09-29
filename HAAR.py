# USAGE
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import cv
import argparse
from imutils.object_detection import non_max_suppression
import time
import math
import dlib
import threading
from multiprocessing import Process

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
person=cv2.CascadeClassifier('/Users/Kanak/opencv/data/haarcascades/haarcascade_fullbody.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
fgbg = cv2.BackgroundSubtractorMOG2()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to videos")
ap.add_argument('--verbose', action='store_true')
ap.set_defaults(verbose=True)
args = vars(ap.parse_args())

#global variables
num_threads=0


threads=[]
index={}
it={}
i=-1
printFrames1=[[] for _ in xrange(100)]
for i in range(40):
	it[i]=-1
	index[i]=-1
currentFaceID = 0
points=[]
fidsToDelete=[]

count=0
initial=[]
trackersAll = {}
printFrames=[]



class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        self.start()
    def stop(self):
    	print "stopping"
    	self.__stop = True 
#


def showFrame(frame):
	global currentFrame,printFrames1,count,index,num_threads,it
	if (len(printFrames1))>0:
		#print "m"
		for j in range(num_threads):
			mx,my,mw,mh=0,0,0,0
			fx,fy,fw,fh=0,0,0,0
			d=it[j]
			if d>0:
				for k in range(d):
					if (printFrames1[j]):
						(x,y,w,h)=printFrames1[j].pop()
						#cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
						mx+=x
						my+=y
						mw+=w
						mh+=h
				fx=mx/d
				fy=my/d
				fw=mw/d
				fh=mh/d
				pad_w=int(0.15*fw)
				pad_h =  int(0.05*fh)
				cv2.rectangle(frame, (fx+pad_w, fy+pad_h), (fx+fw, fy+fh), (0, 255, 0), 2)
		for i in range(len(it)):
			it[i]=0
		for i in range(len(index)):
			index[i]=-1
	cv2.imshow("Result", frame)
	
def track(tracker,(t_x,t_y,t_w,t_h),frame,l):
	iterator=0
	global currentFrame,printFrames1, index,count,it,i, fidsToDelete
	print l," Thread executing"
	i=-1
	it[l]=0
	t=True
	while(t):
		try:
			iterator+=1
			cv2.waitKey(25)
			frame=currentFrame.copy()
			tracker.update(frame)
			tracked_position=tracker.get_position()
			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())
			if not ((t_y,t_x)< frame.shape[:2]):
				if count>0:
					count-=1
				print "stop"
				fidsToDelete.append( l)

				
				break
			i+=1
			with threading.Lock():
				cropped= frame[t_y:t_y+t_h,t_x:t_x+t_w]
				index[i]=l
				it[l]=it[l]+1
				printFrames1[l].append((t_x,t_y,t_w,t_h))
			if (iterator % 40)==0:
				cropped= frame[t_y:t_y+t_h,t_x:t_x+t_w]
				if detect(cropped):
					t=False
					print "stop"
					fidsToDelete.append( l )

					if count>0:
						count-=1
						break
		except:
			break			


  			
def detect(source):
	global count
	ret=False
	r = person.detectMultiScale(source, 1.05, 5)
	if len(r)==0:
		ret=True
			
	return ret
	
def detect_person(source,frame, fno):
	cframe= frame.copy()
	(_imageHeight, _imageWidth) = frame.shape[:2]
	iw=(_imageWidth/3)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#-----

	r = person.detectMultiScale(source, 1.05, 2)
	global printFrames1, count, trackersAll, points,initial,tracker,currentFrame,num_threads
	del initial[:]
	for (x, y, w, h) in r:
	
		if len(r) !=0 :
				for (hx, hy, hw, hh) in r:
						x = int(hx)
						y = int(hy)
						w = int(hw)
						h = int(hh)
						pad_w, pad_h = int(0.15*w), int(0.05*h)
						if (x,y,w,h) in points:
							break
						x_bar = x + 0.5 * w
						y_bar = y + 0.5 * h
						y_length = abs((hy+pad_h)-(hy+hh-pad_h))
						x_length = abs((hx+pad_w)-(hx+hw-pad_w))
						area = x_length * y_length
						matchedFid = None
	            		for fid in trackersAll.keys():
	            			tracked_position=trackersAll[fid].get_position()
	            			t_x = int(tracked_position.left())
	            			t_y = int(tracked_position.top())
	            			t_w = int(tracked_position.width())
	            			t_h = int(tracked_position.height())#cal centerpoint
	            			t_x_bar = t_x + 0.25 * t_w
	            			t_y_bar = t_y + 0.25 * t_h
	            			a=( t_x <= x_bar   <= (t_x + t_w))
	            			b=( t_y <= y_bar   <= (t_y + t_h))
	            			c=( x   <= t_x_bar <= (x   + w  ))
	            			d=( y   <= t_y_bar <= (y   + h  ))
	            			if(a and b and c and d):
	            				matchedFid=fid

	            				if threads[matchedFid].isAlive():
	            					break
	            				else:
	            					matchedFid=None
	            					
	            		if matchedFid is None:
	            			
	            			print("Creating new tracker " + str(detect_person.counter))
	            			points.append(( x+pad_w,y+pad_h,x+w-pad_w,y+h-pad_h))
	            			tracker = dlib.correlation_tracker()
	            			tracker.start_track(frame, dlib.rectangle(x+pad_w,y+pad_h,x+w-pad_w,y+h-pad_h))
	            			trackersAll[detect_person.counter]=tracker
	            			t = Thread(track, tracker,(x+pad_w,y+pad_h,x+w-pad_w,y+h-pad_h),frame,detect_person.counter)
	            			threads.append(t)
	            			detect_person.counter+=1
	            			num_threads+=1
	            			count +=1
	            		if (50<h):
								cropped= cframe[y+pad_h:y+h-pad_h, x+pad_w:x+w-pad_w]
								cv2.imwrite("video/frame%d.jpg" % detect_person.counter, cropped)
		

        
def getInfo(sourcePath):
    cap = cv2.VideoCapture(sourcePath)
    info = {
        "framecount": cap.get(cv.CV_CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv.CV_CAP_PROP_FPS),
        "width": int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv.CV_CAP_PROP_FOURCC))
    }
    cap.release()
    return info


def extractFrames(sourcePath, verbose=True):
    info = getInfo(sourcePath)
    print sourcePath
    cap = cv2.VideoCapture(sourcePath)
    firstFrame=None
    ret = True
    global currentFrame,tracker,trackersAll,printFrames1,count,fidsToDelete 
    while(cap.isOpened() and ret):
        ret, frame = cap.read()
        if ret:
			currentFrame=frame.copy()
			frame_number = cap.get(cv.CV_CAP_PROP_POS_FRAMES) - 1

			for fid in trackersAll.keys():
				trackingQuality = trackersAll[ fid ].update( frame )
				if trackingQuality < 7:
					fidsToDelete.append( fid )
			#If the tracking quality is not good enough, we must delete this tracker
			for fid in fidsToDelete:
				print("Removing fid " + str(fid) + " from list of trackers")
				trackersAll.pop( fid , None )
				
			if frame_number % 5== 0:
				#gray=cv2.resize(frame, (288,360))#----
				#cv2.imshow("h1",gray)
						#resized= cv2.resize(cropped, (64,128))

				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#-----
				gray = cv2.GaussianBlur(frame, (21, 21), 0)#?
				if firstFrame is None:
					firstFrame = gray
					continue
				frameDelta = cv2.absdiff(firstFrame, gray)
				thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]	
				thresh = cv2.dilate(thresh, None, iterations=2)			
				(_imageHeight, _imageWidth) = frame.shape[:2]
				fgmask = fgbg.apply(frame)
				fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
				#cv2.fastNlMeansDenoising(fgmask,fgmask,20,7,21)

				currentFrame=frame.copy()
				cv2.imshow("h",fgmask)
				detect_person(fgmask,frame,frame_number)
				txt = "count is {}".format(count)
				cv2.putText(frame, txt , (0, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
				showFrame(frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

    cap.release()
    cv2.destroyAllWindows()

detect_person.counter=0
if args["verbose"]:
    info = getInfo(args["video"])
    print("Source Info: ", info)

extractFrames(args["video"], args["verbose"])


        
        
        
        
        
        
        
#createsamples -info positives.txt -bg negatives.txt -vec training.vec -w 20 -h 40
#opencv_traincascade -data ~/Desktop -vec trainf.vec -bg neg.txt numPos 1550 -numNeg 4187 -numStages 14 -featureType HAAR -minHitRate 0.995 -maxFalseAlarmRate 0.5 -w 50 -h 120
        
        
        	
	