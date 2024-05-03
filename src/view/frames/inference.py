from tkinter import ttk
import customtkinter # type: ignore
from glob import glob
from model.config import MODELS_PATH
from model.model import load_unet_model
from model.data import load_image_list_from_stage, load_image
from model.visualization import plot_image
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class InferenceFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.model_name = None
        self.model = None
        self.index = 0

        self.first_frame = customtkinter.CTkFrame(self)
        self.first_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.second_frame = customtkinter.CTkFrame(self)
        self.second_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.third_frame = customtkinter.CTkFrame(self)
        self.third_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.fourth_frame = customtkinter.CTkFrame(self)
        self.fourth_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)

        models_list = sorted(glob(MODELS_PATH + '*'))
        models_list = [model.split('/')[-1] for model in models_list]
        self.model_combobox = customtkinter.CTkComboBox(self.first_frame, values=models_list)
        self.model_combobox.set("Select a model")
        self.model_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.load_button = customtkinter.CTkButton(self.first_frame, text="Load", command=lambda x:load_callback(self))
        self.load_button.grid(row=0, column=2, padx=10, pady=10)

        self.label = customtkinter.CTkLabel(self.first_frame, text="No model loaded")
        self.label.grid(row=0, column=3, padx=10, pady=10)

        self.stage_label = customtkinter.CTkLabel(self.second_frame, text="Stage:")
        self.stage_label.grid(row=0, column=0, padx=10, pady=10)

        self.image_combobox = ttk.Combobox(self.second_frame, width=30)
        self.image_combobox.set("Select an image")

        self.stage_combobox = customtkinter.CTkComboBox(self.second_frame, values=["Train", "Stage 1", "Stage 2"], command=lambda x: stage_callback(self, 0))
        self.stage_combobox.set("Select stage")
        self.stage_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.image_label = customtkinter.CTkLabel(self.second_frame, text="Image:")
        self.image_label.grid(row=0, column=2, padx=10, pady=10)

        self.image_combobox.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.predict_button = customtkinter.CTkButton(self.second_frame, text="Predict")
        self.predict_button.grid(row=0, column=4, padx=10, pady=10)

        self.import_predict_button = customtkinter.CTkButton(self.second_frame, text="Import and predict")
        self.import_predict_button.grid(row=0, column=5, padx=10, pady=10)

        self.previous_button = customtkinter.CTkButton(self.fourth_frame, text="<", command=lambda:stage_callback(self, self.index - 1))
        self.previous_button.grid(row=0, column=0, padx=10, pady=10)

        self.next_button = customtkinter.CTkButton(self.fourth_frame, text=">", command=lambda:stage_callback(self, self.index + 1))
        self.next_button.grid(row=0, column=2, padx=10, pady=10)

def load_callback(self):
    if self.model_combobox.get() == 'Select a model':
        return None
    if self.model_name == self.model_combobox.get():
        self.label.configure(text=f"Model {self.model_name} already loaded")
    else:
        self.label.configure(text=f"Loading model {self.model_combobox.get()} ...")
        self.model_name = self.model_combobox.get()
        print(self.model_name)
        self.model = load_unet_model(self.model_name)
        self.label.configure(text=f"Model {self.model_name} loaded successfully !")

def stage_callback(self, index):
    if self.stage_combobox.get() == 'Select stage':
        return None
    
    if index >=0 and index < len(load_image_list_from_stage(self.stage_combobox.get())):
        self.index = index

    
    images_list = load_image_list_from_stage(self.stage_combobox.get())
    self.image_combobox.configure(values=images_list, width=max([len(image) for image in images_list]))
    image = load_image(self.stage_combobox.get(), self.index)
    fig = plot_image(image)
    canvas = FigureCanvasTkAgg(fig, master=self.third_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
