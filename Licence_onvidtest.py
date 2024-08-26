import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

TESSERACT_CMD = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
CASCADE_FILE = './indian_license_plate.xml'
DEFAULT_VIDEO = './video1.mp4'

class LicensePlateRecognizer:
    def __init__(self, master):
        self.master = master
        master.title("License Plate Recognizer")
        self.master.configure(background='#2e3440')  # Dark background
        self.video_source = DEFAULT_VIDEO
        self.cam = None
        self.paused = False

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        self.create_widgets()
        self.start_video()

    def create_widgets(self):
        # Video source selection
        self.control_frame = ttk.Frame(self.master, style="TFrame")
        self.control_frame.pack(pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", background='#4c566a', foreground='white', font=('Helvetica', 10, 'bold'))
        style.configure("TFrame", background='#2e3440')
        style.configure("TLabel", background='#2e3440', foreground='white', font=('Helvetica', 10, 'bold'))

        ttk.Button(self.control_frame, text="Select Video", command=self.select_video, style="TButton").grid(row=0, column=0, padx=5)
        ttk.Button(self.control_frame, text="Pause/Resume", command=self.toggle_pause, style="TButton").grid(row=0, column=1, padx=5)
        ttk.Button(self.control_frame, text="Exit", command=self.master.quit, style="TButton").grid(row=0, column=2, padx=5)

        # Canvas for video display
        self.canvas = tk.Canvas(self.master, width=640, height=480, bg='#3b4252', highlightthickness=0)
        self.canvas.pack()

        # Output frame for detected plate number
        self.output_frame = ttk.Frame(self.master, style="TFrame")
        self.output_frame.pack(pady=10)

        ttk.Label(self.output_frame, text="Detected Plate:", width=15, style="TLabel").pack(side=tk.LEFT)
        self.output_label = ttk.Label(self.output_frame, text="", width=20, style="TLabel")
        self.output_label.pack(side=tk.LEFT)

    def select_video(self):
        video_file = filedialog.askopenfilename(
            initialdir="./", title="Select Video", filetypes=[("Video files", "*.mp4 *.avi")]
        )
        if video_file:
            self.video_source = video_file
            self.start_video()

    def start_video(self):
        if self.cam:
            self.cam.release()
        self.cam = cv2.VideoCapture(self.video_source)
        self.cascade = cv2.CascadeClassifier(CASCADE_FILE)
        self.update_frame()

    def toggle_pause(self):
        self.paused = not self.paused
    def recognize_plate(self, frame):
        plates = self.cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=7)
        for (x, y, w, h) in plates:
            plate_img = Image.fromarray(frame[y:y+h, x:x+w])
            plate_number = pytesseract.image_to_string(plate_img, lang='eng').strip()
            return plate_number
        return None
    def update_frame(self):
        if not self.paused:
            ret, frame = self.cam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plate_number = self.recognize_plate(frame)
                if plate_number:
                    self.output_label.config(text=f"{plate_number}")
                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img
        self.master.after(10, self.update_frame)
    def __del__(self):
        if self.cam:
            self.cam.release()
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognizer(root)
    root.mainloop()