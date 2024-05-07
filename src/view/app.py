import customtkinter
from tkinter import messagebox
from view.frames.inference_frame import InferenceFrame
from view.frames.performance_frame import PerformanceFrame
from view.frames.train_frame import TrainFrame
import matplotlib.pyplot as plt
import os
import sys
import keras.backend as K

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

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, padx=10, pady= 10, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(0, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        self.content_frame = customtkinter.CTkFrame(self)
        self.content_frame.grid(row=0, column=1, padx=(0, 10), pady=10, rowspan=4, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.current_frame = None

        self.create_sidebar()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_sidebar(self):

        self.internal_frame = customtkinter.CTkFrame(self.sidebar_frame)
        self.internal_frame.grid(row=0, column=0, sticky="nsew")
        self.internal_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.internal_frame, text="Nuclei", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = customtkinter.CTkButton(self.internal_frame, text="Train a new model", command=lambda: self.show_frame("Train"))
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.internal_frame, text="Predict with a model", command=lambda: self.show_frame("Predict"))
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_3 = customtkinter.CTkButton(self.internal_frame, text="Information about a model", command=lambda: self.show_frame("Information"))
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.internal_frame, text="Appearance:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))

        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.internal_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(20, 20))

    def is_thread_running(self):
        if self.current_frame:
            if isinstance(self.current_frame, InferenceFrame) and self.current_frame.loading_thread:
                if self.current_frame.loading_thread.is_alive():        
                        response  = messagebox.askquestion("Exit ?", "A model is being loaded. Are you sure you want to exit ?")
                        if response == "no":
                            return True
                        else:
                            self.current_frame.loading_thread_flag = True
                            print('Model loading stopped')
                            return False
                        
            if isinstance(self.current_frame, TrainFrame) and self.current_frame.training_thread:
                if self.current_frame.training_thread.is_alive():        
                        response = messagebox.askquestion("Exit ?", "A model is being trained. Are you sure you want to exit ? This will close the application !") 
                        if response == "no":
                            return True
                        else:
                            sys.stdout = sys.__stdout__
                            sys.stderr = sys.__stderr__
                            print('Model training stopped')
                            self.close()
                            return False

    def show_frame(self, frame_name):
        if self.is_thread_running():
            return
        K.clear_session()
        if frame_name == "Train":
            self.current_frame = TrainFrame(self.content_frame)
        elif frame_name == "Predict":
            self.current_frame = InferenceFrame(self.content_frame)
        elif frame_name == "Information":
            self.current_frame = PerformanceFrame(self.content_frame)

        self.current_frame.grid(row=0, column=0, sticky="nsew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self):
        if self.is_thread_running():
            return
        self.close()
        
    def close(self):
        print("Closing application.")
        K.clear_session()
        plt.close()
        self.quit()
        self.destroy()
        os._exit(0)
