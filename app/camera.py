import cv2 
import time

class CameraStream:
    def __init__(self):
        self.cap = None

        # try different camera indexes
        for index in range(3):

            print(f"Trying camera index: {index}")

            cap = cv2.VideoCapture(index)

            time.sleep(2)


            if cap.isOpened():

                ret, frame = cap.read()

                if ret:

                    print(f"Camera found at index {index}")

                    self.cap = cap

                    break

                cap.release()


        if self.cap is None:
            raise Exception("No webcam found")


        
    def get_frame(self):

        if self.cap is None:
            return None
        
    
        for _ in range(5):

            ret, frame = self.cap.read()

        
            if ret:
                return frame
        
        print("Failed to grab frame")

           
        
        return None
    
    def release(self):
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()