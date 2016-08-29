import cv2
vidcap = cv2.VideoCapture('video/WP_20151203_001.mp4')
success, image = vidcap.read()
count = 0
success = True
export_path = "img/sequences/5/"
while success:
    cv2.imwrite(export_path+"%d.png" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print 'Read a new frame: ', success
    count += 1
print "FINISHED"