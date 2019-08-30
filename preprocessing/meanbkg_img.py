from Generic.filedialogs import BatchProcess
from Generic.video import ReadVideo
import numpy as np
from Generic.images import display
import cv2

if __name__ == '__main__':



    for file in BatchProcess(pathfilter='/media/ppzmis/data/ActiveMatter/Microscopy/190820bacteriaand500nmparticles/videos/joined/StreamDIC???.mp4'):
        readvid=ReadVideo(filename=file)
        frame_init=readvid.read_next_frame()#.astype(np.int32)
        counter=1
        sz=np.shape(frame_init)
        frame_assemble=np.reshape(frame_init,(sz[0],sz[1]*sz[2]))

        for i in range(readvid.num_frames-1):
            frame=readvid.read_next_frame().astype(np.int32)
            new_frame=np.reshape(frame,(sz[0],sz[1]*sz[2]))
            frame_assemble=np.sum((frame_assemble,new_frame),axis=0, dtype=np.int32)
            counter=counter+1
        frame = (frame_assemble / counter).astype(np.uint8)
        frame_assemble = np.reshape(frame,(sz[0],sz[1],sz[2]))
        cv2.imwrite(file[:-4]+'_bkgimg.png', frame_assemble)


        readvid.close()
