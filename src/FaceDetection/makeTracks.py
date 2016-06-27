def check_position_size(box):
    Xpos=1.*(box[0]>400) * 1.*(box[0]<1000)
    Ypos=1.*(box)[1]>50 * 1.*(box[1]<220)
    Size=1.*(box[2])>130
    return Xpos*Ypos*Size

def is_match(f1,f2):
    diff=max(abs(f1-f2))
    return diff<15

from numpy import *
from glob import glob
import pickle

for filename in glob('../output/*_Faces.pkl'):
    print '#'*50,'processing', filename
    Faces =pickle.load(open(filename,'r'))

    # remove faces whose parameters are too atypical
    GoodFaces=[]
    remove=0
    for F in Faces:
        L=[]
        for f in F:
            if check_position_size(f):
                L.append(f)
            else:
                remove+=1
        GoodFaces.append(L)
 
    print '\nremoved',remove,'atypical faces'
    # Organize Face detections into tracks
    Tracks=[]
    for i in range(len(GoodFaces)):
        print '\r',i,
        F=GoodFaces[i]

        for f in F:
            attached=False
            for track in Tracks:
                j,fprev=track[-1]
                if i-j<10 and is_match(f,fprev):
                    track.append((i,f))
                    attached=True
                    break
            if not attached:
                Tracks.append([(i,f)])
    all_tracks=len(Tracks)
    Tracks=[f for f in Tracks if len(f)>100]
    print '\nall tracks:',all_tracks,'long tracks:',len(Tracks)

    #calc some statistics of the tracks
    n=0
    for track in Tracks:
        gaps=0
        diffs=[]
        for i in range(1,len(track)):
            gap=track[i][0]-track[i-1][0]
            gaps+=gap-1
            diffs.append((track[i][1]-track[i-1][1])/gap)
        print n,track[0][0],gaps,len(track)
        print 'mean=',mean(stack([t[1] for t in track]),axis=0)[:3],
        print 'std of diffs=',std(stack(diffs),axis=0)[:3]
        n+=1

    #smooth the tracks
    k=2
    SmoothTracks=[]
    print '\n smoothing the tracks'
    print 'start end length'
    for track in Tracks:
        start=track[0][0]
        end=track[-1][0]
        length=end-start+1
        print start,end,length
        Track1=empty([length,5])
        Track1[:,:]=nan
        #put the entries in track into the array Track1, leaving nan values in the holes
        for i in range(len(track)):
            index=track[i][0]-start
            Track1[index,1:5]=track[i][1]

        Track2=empty([length,5])
        Track2[:,:]=nan
        for i in range(length):
            Track2[i][0]=start+i
            Track2[i][1:5]=nanmean(Track1[max(i-k,0):min(i+k,length),1:5],axis=0)

        Track3=array(Track2,dtype=int16)  

        SmoothTracks.append(Track3)
    pickle.dump(SmoothTracks,open(filename[:-9]+'Tracks.pkl','w'))