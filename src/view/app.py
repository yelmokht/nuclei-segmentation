from contextlib import closing
import customtkinter
from CTkMessagebox import CTkMessagebox
from view.frames.inference import InferenceFrame
from view.frames.performance import PerformanceFrame
from view.frames.train import TrainFrame
import matplotlib.pyplot as plt
import sys
import os

customtkinter.set_appearance_mode("System") 
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Nuclei segmentation")
        self.geometry(f"{1445}x{695}")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.content_frame = customtkinter.CTkFrame(self)
        self.content_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.current_frame = None

        self.create_sidebar()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sidebar(self):

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Nuclei", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Train a new model", command=lambda: self.show_frame("Train"))
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Predict with a model", command=lambda: self.show_frame("Predict"))
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Information about a model", command=lambda: self.show_frame("Information"))
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        # self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        # self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))

        # self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        # self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

    def show_frame(self, frame_name):
        if self.current_frame:
            if isinstance(self.current_frame, InferenceFrame) and self.current_frame.loading_thread:
                if self.current_frame.loading_thread.is_alive():        
                        msg = CTkMessagebox(master=self, title="Exit?", message="A model is being loaded. Are you sure you want to exit?",
                        icon="question", option_1="Cancel", option_2="No", option_3="Yes", sound=True)
                        response = msg.get()
                        if response=="No" or response=="Cancel":
                            return
                        else:
                            print("Loading model stopped.")
                        
            if isinstance(self.current_frame, TrainFrame) and self.current_frame.training_thread:
                if self.current_frame.training_thread.is_alive():        
                        msg = CTkMessagebox(master=self, title="Exit?", message="A model is being trained. Are you sure you want to exit?",
                        icon="question", option_1="Cancel", option_2="No", option_3="Yes", sound=True)
                        response = msg.get()
                        if response=="No" or response=="Cancel":
                            return
                        else:
                            self.training_thread = None
                            print("Training model stopped.")

        if frame_name == "Train":
            self.current_frame = TrainFrame(self.content_frame)
        elif frame_name == "Predict":
            self.current_frame = InferenceFrame(self.content_frame)
        elif frame_name == "Information":
            self.current_frame = PerformanceFrame(self.content_frame)

        self.current_frame.grid(row=0, column=0, sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    
    def on_closing(self):
        plt.close()
        self.quit()
        self.destroy()
        os._exit(0)
