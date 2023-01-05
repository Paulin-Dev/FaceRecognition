
from os.path import abspath, join, dirname

import cv2
from screeninfo import get_monitors


DIR_PATH      = dirname(abspath(__file__))
IMG_PATH      = join(DIR_PATH, 'images')
MODELS_PATH   = join(DIR_PATH, 'models', 'haarcascades')

SCREEN_HEIGHT = get_monitors()[0].height
SCREEN_WIDTH  = get_monitors()[0].width


class Camera:
    def __init__(self) -> None:
        try:
            self.camera = cv2.VideoCapture(0)
        except KeyboardInterrupt:
            exit(1)

        self.title = 'Love Calculator'

        self.frontalCascade = cv2.CascadeClassifier(join(MODELS_PATH, 'haarcascade_frontalface_default.xml'))
        self.profileCascade = cv2.CascadeClassifier(join(MODELS_PATH, 'haarcascade_profileface.xml'))
        
        cv2.namedWindow(self.title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def detect_faces(self) -> None:
        while True:
            try:
                ret, frame = self.camera.read()
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # minSize could be modified according to the frame's size    minSize=(70, 70)    minNeighbors=20
                faces = self.frontalCascade.detectMultiScale(
                    grayscale,
                    scaleFactor=1.1,
                    minNeighbors=40,
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                if isinstance(faces, tuple):
                    faces = self.profileCascade.detectMultiScale(
                        grayscale,
                        scaleFactor=1.1,
                        minNeighbors=40,
                        flags = cv2.CASCADE_SCALE_IMAGE
                    )
                
                # Draw rectangles 
                for (x, y, width, height) in faces:
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (71, 46, 231), 2)     # BGR color
                    
                # Resize to fit the screen's height
                frame = cv2.resize(frame, (round(frame.shape[0]*(SCREEN_WIDTH/frame.shape[1])), SCREEN_HEIGHT))
                
                # Add borders (img, topBorderWidth, bottomBorderWidth, leftBorderWidth, rightBorderWidth, borderStyle, color)
                x_margin = (SCREEN_WIDTH - frame.shape[1])//2
                frame = cv2.copyMakeBorder(frame, 0, 0, x_margin, x_margin, cv2.BORDER_CONSTANT, value=(71, 46, 231))
                
                cv2.imshow(self.title, frame)
                
                # check if escape key or close button is pressed 
                if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    self.camera.release()
                    cv2.destroyAllWindows()
                    break
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    camera = Camera()
    camera.detect_faces()
















'''

def main():

    camera = cv2.VideoCapture(0)
    foreground = cv2.imread(join(IMG_PATH, "fg_1.png"))
    faceCascade = cv2.CascadeClassifier(CASC_PATH)
    while True:
        ret, frame = camera.read()
        #frame = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2))    # scale x2
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # minSize could be modified according to the frame's size
        faces = faceCascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,
            minNeighbors=20,
            minSize=(70, 70),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles on faces
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (71, 46, 231), 2)     # /!\ BGR color

        # prevent to rescale the foreground multiple times
        try:
            output = cv2.addWeighted(frame, 1, foreground, 0.5, 0)
        except:
            foreground = cv2.resize(foreground, (frame.shape[1], frame.shape[0]))
            output = cv2.addWeighted(frame, 1, foreground, 0.5, 0)

        image = cv2.imshow("Love Calculator", output)

        # check if escape key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

    camera.release()

'''