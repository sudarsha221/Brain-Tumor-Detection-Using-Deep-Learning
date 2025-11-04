import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained brain tumor model
model = load_model("best_model.h5")  # change path if needed
IMG_SIZE = 224  # or 128, 64 depending on your model input size

class BrainTumorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Brain Tumor Detection")
        self.window.geometry("600x600")
        self.image_path = None

        self.label = Label(window, text="Brain Tumor Detection", font=("Arial", 20))
        self.label.pack(pady=20)

        self.img_label = Label(window)
        self.img_label.pack(pady=10)

        self.result_label = Label(window, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=10)

        self.choose_btn = Button(window, text="Choose Image", command=self.choose_image, font=("Arial", 14))
        self.choose_btn.pack(pady=5)

        self.predict_btn = Button(window, text="Predict Tumor", command=self.predict, font=("Arial", 14))
        self.predict_btn.pack(pady=5)

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.img_label.config(image=photo)
            self.img_label.image = photo
            self.result_label.config(text="")

    def predict(self):
        if self.image_path is None:
            self.result_label.config(text="Please select an image first.", fg="red")
            return

        img = cv2.imread(self.image_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]  # adjust for your model's output
        result = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
        color = "red" if prediction > 0.5 else "green"

        self.result_label.config(text=f"Prediction: {result}", fg=color)

# Run the app
root = tk.Tk()
app = BrainTumorApp(root)
root.mainloop()
