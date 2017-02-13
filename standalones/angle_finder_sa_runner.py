import cv2

from standalones.angle_finder_standalone import AngleFinderStandalone
from standalones.entropy_angle_finder_standalone import EntropyAngleFinderStandalone

# EntropyAngleFinderStandalone().run()
AngleFinderStandalone().run()
cv2.destroyAllWindows()