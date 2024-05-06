import multiprocessing
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import customtkinter
from glob import glob
from model.config import *
import threading
from model.evaluation import individual_score
from model.model import load_unet_model
from model.data import load_ground_truth, load_image_list_from_stage, load_image, read_image
from model.inference import tta
from model.pre_processing import preprocess
from model.visualization import plot_ground_truth, plot_image, plot_prediction
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import numpy as np

class InferenceFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.model_name = None
        self.model = None
        self.index = 0
        self.loading_thread = None

        self.first_frame = customtkinter.CTkFrame(self)
        self.second_frame = customtkinter.CTkFrame(self)

        # self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=1)
        

        self.first_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.second_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")

        self.third_frame = customtkinter.CTkFrame(self)
        self.third_frame.grid(row=2, column=0, padx=10, pady=(10, 0), sticky="nsew")

        self.third_frame.grid_rowconfigure(0, weight=1)
        self.third_frame.grid_columnconfigure(1, weight=1)
        self.third_frame.grid_columnconfigure(2, weight=1)

        self.fourth_frame = customtkinter.CTkFrame(self)
        self.fourth_frame.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="nsew")

        self.fifth_frame = customtkinter.CTkFrame(self)
        self.fifth_frame.grid(row=4, column=0, padx=10, pady=(10, 10), sticky="nsew")

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)

        models_list = sorted(glob(MODELS_PATH + '*'))
        models_list = [model.split('/')[-1] for model in models_list]
        self.model_combobox = customtkinter.CTkComboBox(self.first_frame, values=models_list)
        self.model_combobox.set("Select a model")
        self.model_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.load_button = customtkinter.CTkButton(self.first_frame, text="Load", command=lambda:load_callback(self))
        self.load_button.grid(row=0, column=2, padx=10, pady=10)

        self.label = customtkinter.CTkLabel(self.first_frame, text="No model loaded")
        self.label.grid(row=0, column=3, padx=10, pady=10)

        self.progressbar_1 = customtkinter.CTkProgressBar(self.first_frame)

        self.stage_label = customtkinter.CTkLabel(self.second_frame, text="Stage:")
        self.stage_label.grid(row=0, column=0, padx=10, pady=10)

        self.image_combobox = ttk.Combobox(self.second_frame, width=70)
        self.image_combobox.set("Select an image")
        self.image_combobox.bind("<<ComboboxSelected>>", lambda event: image_combobox_callback(self))

        self.stage_combobox = customtkinter.CTkComboBox(self.second_frame, values=["Train", "Stage 1", "Stage 2"], command=lambda x: stage_callback(self, 0))
        self.stage_combobox.set("Select stage")
        self.stage_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.image_label = customtkinter.CTkLabel(self.second_frame, text="Image:")
        self.image_label.grid(row=0, column=2, padx=10, pady=10)

        self.image_combobox.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.predict_button = customtkinter.CTkButton(self.second_frame, text="Predict", command=lambda:predict_callback(self))
        self.predict_button.grid(row=0, column=4, padx=10, pady=10)

        self.import_predict_button = customtkinter.CTkButton(self.second_frame, text="Import and predict", command=lambda:import_and_predict_callback(self))
        self.import_predict_button.grid(row=0, column=5, padx=10, pady=10)

        self.score_label = customtkinter.CTkLabel(self.fourth_frame, text="Score:")
        self.score_label.grid(row=0, column=0, padx=10, pady=10)

        self.score = customtkinter.CTkLabel(self.fourth_frame, text="0.0")
        self.score.grid(row=0, column=1, padx=10, pady=10)

        self.true_objects_label = customtkinter.CTkLabel(self.fourth_frame, text="True objects:")
        self.true_objects_label.grid(row=0, column=2, padx=10, pady=10)

        self.true_objects_number = customtkinter.CTkLabel(self.fourth_frame, text="0")
        self.true_objects_number.grid(row=0, column=3, padx=10, pady=10)

        self.pred_objects_label = customtkinter.CTkLabel(self.fourth_frame, text="Predicted objects:")
        self.pred_objects_label.grid(row=0, column=4, padx=10, pady=10)

        self.pred_objects_number = customtkinter.CTkLabel(self.fourth_frame, text="0")
        self.pred_objects_number.grid(row=0, column=5, padx=10, pady=10)

        self.fifth_frame.grid_columnconfigure(0, weight=1)
        self.fifth_frame.grid_columnconfigure(1, weight=1)

        self.previous_button = customtkinter.CTkButton(self.fifth_frame, text="<", command=lambda:stage_callback(self, self.index - 1))
        self.previous_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.next_button = customtkinter.CTkButton(self.fifth_frame, text=">", command=lambda:stage_callback(self, self.index + 1))
        self.next_button.grid(row=0, column=1, padx=10, pady=10, sticky="e")

    def popup_message(message):
        messagebox.showinfo("Information", message)

def load_unet_model_thread(self):
    try:
        self.model = load_unet_model(self.model_name)
    except Exception as e:
        print("Error loading model:", e)

def load_callback(self):
    if self.model_combobox.get() == 'Select a model':
        return None
    if self.model_name == self.model_combobox.get():
        self.label.configure(text=f"Model {self.model_name} already loaded")
    else:
        self.label.grid_remove()
        self.progressbar_1.grid(row=0, column=3, padx=(10, 10), pady=(10, 10), sticky="ew")
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        self.progressbar_1.update()
        self.model_name = self.model_combobox.get()
        print(self.model_name)
        self.loading_thread = threading.Thread(target=load_unet_model_thread, args=(self,))
        self.loading_thread.setDaemon(True)
        self.loading_thread.start()
        while self.loading_thread.is_alive():
            self.update()
        self.progressbar_1.stop()
        self.progressbar_1.grid_remove()
        self.label.configure(text=f"Model {self.model_name} loaded successfully !")
        self.label.grid(row=0, column=3, padx=10, pady=10)
        self.label.update()

def stage_callback(self, index):
    if self.stage_combobox.get() == 'Select stage':
        return None
    
    if index >=0 and index < len(load_image_list_from_stage(self.stage_combobox.get())):
        self.index = index

    images_list = load_image_list_from_stage(self.stage_combobox.get())
    self.image_combobox.configure(values=images_list)
    self.image_combobox.set(images_list[self.index])
    image_combobox_callback(self)

def image_combobox_callback(self):
    if self.image_combobox.get() == 'Select an image':
        return None
    self.index = self.image_combobox.current()
    image = load_image(self.stage_combobox.get(), self.index)
    fig = plot_image(image)
    canvas = FigureCanvasTkAgg(fig, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    if self.model is not None:
        predict_callback(self)

def predict_callback(self):
    if self.model is None or self.image_combobox.get() == 'Select an image' or self.stage_combobox.get() == 'Select stage':
        print("Model, image or stage not selected")
        return None

    gt_mask = load_ground_truth(self.stage_combobox.get(), self.index)
    fig1 = plot_ground_truth(gt_mask)

    image = np.array([preprocess(load_image(self.stage_combobox.get(), self.index), IMAGE_SHAPE)])
    prediction = tta(self.model, image)
    fig2 = plot_prediction(image[0], prediction[0])

    a, b, c = individual_score(gt_mask, np.squeeze(prediction[0]))
    self.score.configure(text=str(round(c, 2)))
    self.true_objects_number.configure(text=str(a))
    self.pred_objects_number.configure(text=str(b))

    canvas = FigureCanvasTkAgg(fig1, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

    canvas = FigureCanvasTkAgg(fig2, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=2, sticky="nsew")

    self.update()
    self.update_idletasks()

def import_and_predict_callback(self):
    if self.model is None:
        print("Model not selected")
        return None

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
    if not file_path:
        return None
    
    fig = plot_image(read_image(file_path))
    canvas = FigureCanvasTkAgg(fig, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    
    image = np.array([preprocess(read_image(file_path), IMAGE_SHAPE)])
    prediction = tta(self.model, image)
    fig = plot_prediction(image[0], prediction[0])
    canvas = FigureCanvasTkAgg(fig, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
    self.update()
    self.update_idletasks()

