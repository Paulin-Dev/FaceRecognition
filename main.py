import cv2
from os.path import abspath, join, dirname


DIR_PATH = dirname(abspath(__file__))
IMG_PATH = join(DIR_PATH, "images")

# path to the cascade (XML file that contains the data to detect faces)
CASC_PATH = join(DIR_PATH, "haarcascade_frontalface_default.xml")


if __name__ == "__main__":

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
