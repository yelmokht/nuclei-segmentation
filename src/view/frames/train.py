import customtkinter
import os
import sys
import threading
import time
from model.model import modified_unet_model, train, train_model, save_model, save_history

class TrainFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.model_name = None
        self.training_thread = None
        self.train_textbox_thread = None

        self.first_frame = customtkinter.CTkFrame(self)
        self.first_frame.grid(row=0, column=0, columnspan=7, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.first_frame.rowconfigure(0, weight=1)
        self.first_frame.columnconfigure(0, weight=1)
        self.second_frame = customtkinter.CTkFrame(self)
        self.second_frame.grid(row=1, column=0, columnspan=7, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.second_frame.rowconfigure(0, weight=1)
        self.second_frame.columnconfigure(0, weight=1)

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model name:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)
        self.model_entry = customtkinter.CTkEntry(self.first_frame)
        self.model_entry.grid(row=0, column=1, padx=10, pady=10)

        self.batch_size_label = customtkinter.CTkLabel(self.first_frame, text="Batch size:")
        self.batch_size_label.grid(row=0, column=2, padx=10, pady=10)
        self.batch_size_combobox = customtkinter.CTkComboBox(self.first_frame, values=["8", "16", "32", "64"])
        self.batch_size_combobox.set("32")
        self.batch_size_combobox.grid(row=0, column=3, padx=10, pady=10)

        self.epochs_label = customtkinter.CTkLabel(self.first_frame, text="Epochs:", text_color_disabled="black")
        self.epochs_label.grid(row=0, column=4, padx=10, pady=10)
        self.epochs_entry = customtkinter.CTkEntry(self.first_frame)
        self.epochs_entry.grid(row=0, column=5, padx=10, pady=10)
        self.epochs_entry.insert(0, "20")

        self.train_button = customtkinter.CTkButton(self.first_frame, text="Train", command=lambda:self.train_callback())
        self.train_button.grid(row=0, column=6, columnspan=2, padx=10, pady=10)

        self.train_textbox = customtkinter.CTkTextbox(self.second_frame, width=250)
        self.train_textbox.grid(row=0, column=0, columnspan=7, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.train_textbox.insert("0.0", "CTkTextbox\n\n" + "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.\n\n" * 20)
    
    def training_thread(self):
        self.model_name = self.model_entry.get()
        print(f"Training model {self.model_name}...")
        dir_path = f"./models/{self.model_name}"
        os.makedirs(dir_path)
        batch_size = self.batch_size_combobox.get()
        epochs = self.epochs_entry.get()
        try:
            train(model_name=self.model_name, batch_size=batch_size, epochs=epochs)
        except Exception as e:
            print("Error training model:", e)


    def disable_all(self):
        self.model_entry.configure(state="disabled")
        self.batch_size_combobox.configure(state="disabled")
        self.epochs_entry.configure(state="disabled")
        self.train_button.configure(state="disabled")
        
        

    def enable_all(self):
        for child in self.winfo_children():
            child.configure(state="enable")

    def train_textbox_thread(self):
        self.disable_all()
        while self.training_thread.is_alive():
            self.train_textbox.insert("end", sys.stdout)
            self.train_textbox.see("end")
            self.train_textbox.update()
            time.sleep(0.1)
        
        self.train_textbox.insert("end", "Training...\n")
        self.train_textbox.see("end")
        self.train_textbox.update()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.enable_all()
        
    def train_callback(self):
        if self.model_entry.get() == "":
            print("Please enter a model name.")
            return
        
        if self.model_entry.get() in os.listdir("./models"):
            print("Model name already exists.")
            print("Please enter a different model name.")
            return
        
        if self.epochs_entry.get() == "":
            print("Please enter the number of epochs.")
            return
        
        train(self.model_entry.get(), self.batch_size_combobox.get(), self.epochs_entry.get())
        # self.training_thread = threading.Thread(target=self.training_thread)
        # self.training_thread.start()
        # self.train_textbox_thread = threading.Thread(target=self.train_textbox_thread)
        # self.train_textbox_thread.start()



