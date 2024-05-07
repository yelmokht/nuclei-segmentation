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
        self.batch_size_combobox.set("16")
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

        self.training_thread = threading.Thread(target=training_thread, args=(self,))
        self.training_thread.daemon = True

        self.train_textbox_thread = threading.Thread(target=train_textbox_thread, args=(self,))
        self.train_textbox_thread.daemon = True

        self.training_thread.start()
        self.train_textbox_thread.start()


def training_thread(self):
    train(self.model_entry.get(), int(self.batch_size_combobox.get()), int(self.epochs_entry.get()))

def train_textbox_thread(self):
    open("temp.txt", "w").close()
    last_modification_time = os.path.getmtime("temp.txt")
    last_file_size = 0
    
    with open("temp.txt", "a") as stdout_file, open("temp.txt", "a") as stderr_file:
        sys.stdout = stdout_file
        sys.stderr = stderr_file
        last_content = ""
        try:
            while self.training_thread.is_alive():
                current_modification_time = os.path.getmtime("temp.txt")
                current_file_size = os.path.getsize("temp.txt")
                
                if current_modification_time != last_modification_time or current_file_size > last_file_size:
                    with open("temp.txt") as f:
                        f.seek(last_file_size)
                        new_content = f.read()
                        last_content = new_content
                        if ('\b' or '\r' in new_content) or ('\b' or '\r' in last_content):
                            self.train_textbox.delete("end-1l", "end")
                            self.train_textbox.insert("end", new_content.replace('\b', '').replace('\r', ''))                        
                        if self.train_textbox.yview()[1] > 0.85:
                            self.train_textbox.see("end")
                        self.train_textbox.update()

                    last_modification_time = current_modification_time
                    last_file_size = current_file_size
                
                time.sleep(0.01)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            messagebox.showinfo("Training", "Model training completed.")