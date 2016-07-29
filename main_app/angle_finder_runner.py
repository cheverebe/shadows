import cv2

from main_app.angle_finder import AngleFinder

ma = AngleFinder()
ma.run()
cv2.destroyAllWindows()