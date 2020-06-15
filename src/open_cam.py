"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('My Room', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        print("CAMERA CLOSED")
    cv2.destroyAllWindows()


def main():
    print("PRESS ESC TO QUIT")
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()