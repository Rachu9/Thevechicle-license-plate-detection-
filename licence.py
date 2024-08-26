import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

TESSERACT_CMD = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update with your path
CASCADE_FILE = './indian_license_plate.xml'

class LicensePlateRecognizer:
    def __init__(self, master):
        self.master = master
        master.title("License Plate Recognizer")

        # GUI setup
        self.create_widgets()
        
        # Initialize camera and license plate detector
        self.cam = cv2.VideoCapture(0)  # 0 for default webcam
        self.license_cascade = cv2.CascadeClassifier(CASCADE_FILE)
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        # Start processing
        self.update_frame()

    def create_widgets(self):
        # Video display
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack(pady=10)

        # Output label
        ttk.Label(self.master, text="Detected Plate:").pack()
        self.output_label = ttk.Label(self.master, text="", font=("Arial", 16))
        self.output_label.pack(pady=5)

    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding to improve OCR accuracy
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Apply dilation and erosion to close gaps in between object edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        return eroded

    def update_frame(self):
        ret, frame = self.cam.read()

        if ret:
            # Detect license plates
            plates = self.license_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in plates:
                # Extract plate region
                plate_region = frame[y:y+h, x:x+w]

                # Preprocess the plate region
                preprocessed_plate = self.preprocess_image(plate_region)

                # Perform OCR on plate region
                plate_text = pytesseract.image_to_string(preprocessed_plate, lang='eng', config='--psm 8')

                # Draw rectangle around the detected plate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Update output label
                self.output_label.config(text=plate_text.strip())

            # Display frame on canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Tkinter
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img  

        # Schedule the next frame update
        self.master.after(10, self.update_frame)

    def __del__(self):
        self.cam.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognizer(root)
    root.mainloop()
