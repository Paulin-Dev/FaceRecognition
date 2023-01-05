
# Built-in imports
from os.path import abspath, join, dirname
from random import randint
from time import perf_counter

# 3rd party imports
import cv2
from screeninfo import get_monitors

# constants
DIR_PATH      = dirname(abspath(__file__))
IMG_PATH      = join(DIR_PATH, 'images')
MODELS_PATH   = join(DIR_PATH, 'models', 'haarcascades')

SCREEN_HEIGHT = get_monitors()[0].height
SCREEN_WIDTH  = get_monitors()[0].width

FACES_NEEDED  = 1 


class Camera:
    def __init__(self, title: str) -> None:
        try:
            self.__camera = cv2.VideoCapture(0)
        except KeyboardInterrupt:
            exit(1)

        self.__title = title

        self.__frontalCascade = cv2.CascadeClassifier(join(MODELS_PATH, 'haarcascade_frontalface_default.xml'))
        self.__profileCascade = cv2.CascadeClassifier(join(MODELS_PATH, 'haarcascade_profileface.xml'))
        
        cv2.namedWindow(self.__title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.__title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.__countdown      = 0
        self.__step           = 0
        self.__love           = 0

    def __draw_values(self, faces, frame) -> None:
        if self.__countdown == 0:
            self.__countdown = perf_counter()

        elif perf_counter()-self.__countdown >= 1:
            self.__countdown = 0

            self.__step += 1
            if self.__step == 4:
                self.__love = randint(50, 100)

        if len(faces) == FACES_NEEDED:
            if 1 <= self.__step <= 3:
                cv2.putText(frame, f'{randint(0, 100)}%', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            elif self.__step > 10:
                self.__step = 0

        elif len(faces) < FACES_NEEDED and 1 <= self.__step <= 3:
            self.__step = 0

        if 4 <= self.__step <= 9:
            cv2.putText(frame, f'{self.__love}%', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    def __draw_rectangles(self, faces, frame) -> None:
        for index, (x, y, width, height) in enumerate(faces):
            if index < 2:
                # /!\ BGR color
                cv2.rectangle(frame, (x, y), (x+width, y+height), (71, 46, 231), 2)    

    def __resize_frame(self, frame):
        # Resize to fit the screen's height
        frame = cv2.resize(frame, (round(frame.shape[0]*(SCREEN_WIDTH/frame.shape[1])), SCREEN_HEIGHT))

        # Add left n right black borders (img, topBorderWidth, bottomBorderWidth, leftBorderWidth, rightBorderWidth, borderStyle, color)
        x_margin = (SCREEN_WIDTH - frame.shape[1])//2
        return cv2.copyMakeBorder(frame, 0, 0, x_margin, x_margin, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def detect_faces(self) -> None:
        while True:
            try:
                ret, frame = self.__camera.read()
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # minSize could be modified according to the frame's size              minSize=(70, 70)
                faces = self.__frontalCascade.detectMultiScale(
                    grayscale,
                    scaleFactor=1.04,
                    minNeighbors=40,
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                if isinstance(faces, tuple):
                    faces = self.__profileCascade.detectMultiScale(
                        grayscale,
                        scaleFactor=1.1,
                        minNeighbors=15,
                        flags = cv2.CASCADE_SCALE_IMAGE
                    )

                self.__draw_values(faces, frame)
                self.__draw_rectangles(faces, frame)
                frame = self.__resize_frame(frame)

                cv2.imshow(self.__title, frame)

                # Check if escape key or close button is pressed 
                if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(self.__title, cv2.WND_PROP_VISIBLE) < 1:
                    self.__camera.release()
                    cv2.destroyAllWindows()
                    break

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    camera = Camera('Love Calculator')
    camera.detect_faces()









'''
Adds a foreground
=================

foreground = cv2.imread(join(IMG_PATH, "fg_1.png"))

# prevent to rescale the foreground multiple times
try:
    output = cv2.addWeighted(frame, 1, foreground, 0.5, 0)
except Exception:
    foreground = cv2.resize(foreground, (frame.shape[1], frame.shape[0]))
    output = cv2.addWeighted(frame, 1, foreground, 0.5, 0)
'''
