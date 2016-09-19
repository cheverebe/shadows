import cv2
from datetime import datetime

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

class VideoRecorder(object):

    def __init__(self, capture):
        w=int(capture.get(CV_CAP_PROP_FRAME_WIDTH))
        h=int(capture.get(CV_CAP_PROP_FRAME_HEIGHT))
        # video recorder
        fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
        now = datetime.now()
        now_str = "%d-%d-%d" % (now.year, now.month, now.day)
        self.video_writer = cv2.VideoWriter("output-%s.mpeg" % now_str, fourcc, 25, (w, h))

    def write(self, frame):
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()