import customtkinter
import signal
from tkinter import messagebox
import os
import sys
import threading
import time
from model.train import train

class TrainFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.model_name = None
        self.training_thread = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.first_frame = customtkinter.CTkFrame(self)
        self.first_frame.grid(row=0, column=0, columnspan=7, padx=10, pady=(10, 0), sticky="nsew")
        
        self.second_frame = customtkinter.CTkFrame(self)
        self.second_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.second_frame.grid_rowconfigure(0, weight=1)
        self.second_frame.grid_columnconfigure(0, weight=1)

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model name:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)
        self.model_entry = customtkinter.CTkEntry(self.first_frame)
        self.model_entry.grid(row=0, column=1, padx=10, pady=10)

        self.batch_size_label = customtkinter.CTkLabel(self.first_frame, text="Batch size:")
        self.batch_size_label.grid(row=0, column=2, padx=10, pady=10)
        self.batch_size_combobox = customtkinter.CTkComboBox(self.first_frame, values=["8", "16", "32", "64"])
        self.batch_size_combobox.grid(row=0, column=3, padx=10, pady=10)

        self.epochs_label = customtkinter.CTkLabel(self.first_frame, text="Epochs:", text_color_disabled="black")
        self.epochs_label.grid(row=0, column=4, padx=10, pady=10)
        self.epochs_entry = customtkinter.CTkEntry(self.first_frame)
        self.epochs_entry.grid(row=0, column=5, padx=10, pady=10)
        self.epochs_entry.insert(0, "5")

        self.train_button = customtkinter.CTkButton(self.first_frame, text="Train", command=lambda:self.train_callback())
        self.train_button.grid(row=0, column=6, columnspan=2, padx=10, pady=10)

        self.train_textbox = customtkinter.CTkTextbox(self.second_frame)
        self.train_textbox.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def decorator(self, func):
        def inner(str):
            try:
                if '\r' in str:
                    cursor_index = self.train_textbox.index('insert')
                    line_number = cursor_index.split('.')[0]
                    line_start_index = f"{line_number}.0"
                    line_end_index = f"{line_number}.end"
                    self.train_textbox.delete(line_start_index, line_end_index)
                    self.train_textbox.insert(line_end_index, str.replace('\b', '').replace('\r', ''))
                    return func(str)
                self.train_textbox.insert("end", str)
                return func(str)
            except:
                return func(str)
        return inner

    def train_callback(self):
        if self.model_entry.get() == "":
            messagebox.showerror("Error", "Please enter a model name.")
            return
        
        if self.model_entry.get() in os.listdir("./models"):
            messagebox.showerror("Error", "Model name already exists. Please enter a different model name.")
            return
        
        if self.epochs_entry.get() == "":
            messagebox.showerror("Error", "Please enter the number of epochs.")
            return
        
        sys.stdout.write=self.decorator(sys.stdout.write)
        sys.stderr.write=self.decorator(sys.stdout.write)

        self.training_thread = threading.Thread(target=training_thread, args=(self,))
        self.training_thread.daemon = True
        self.training_thread.start()


def training_thread(self):
    train(self.model_entry.get(), int(self.batch_size_combobox.get()), int(self.epochs_entry.get()))
    messagebox.showinfo("Training", "Model training completed.")