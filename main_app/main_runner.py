import cv2

from main_app.main import MainApp

ma = MainApp()
ma.run()
if ma.source_folder == 'camera':
    ma.cap.release()
cv2.destroyAllWindows()
