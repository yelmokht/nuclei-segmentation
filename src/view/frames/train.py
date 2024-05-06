import multiprocessing
import customtkinter
import os
import sys
import threading
import time
from model.model import modified_unet_model, train, train_model, save_model, save_history
import io

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

        # self.train_textbox = customtkinter.CTkTextbox(self.second_frame, width=1000)
        # self.train_textbox.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

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
        
        print(f"Training model {self.model_entry.get()}...")
        print(f"Batch size: {self.batch_size_combobox.get()}")

        self.training_thread = multiprocessing.Process(target=training_thread, args=(self,))
        self.training_thread.start()

        # self.train_textbox_thread = threading.Thread(target=train_textbox_thread, args=(self,))
        # self.train_textbox_thread.start()

def training_thread(self):
    try:
        train(self.model_entry.get(), int(self.batch_size_combobox.get()), int(self.epochs_entry.get()))
    except Exception as e:
        print("Error training model:", e)

# def train_textbox_thread(self):
#     open("temp.txt", "w").close()
#     last_modification_time = os.path.getmtime("temp.txt")
#     last_file_size = 0
    
#     with open("temp.txt", "a") as stdout_file, open("temp.txt", "a") as stderr_file:
#         sys.stdout = stdout_file
#         sys.stderr = stderr_file
#         try:
#             while self.training_thread.is_alive():
#                 current_modification_time = os.path.getmtime("temp.txt")
#                 current_file_size = os.path.getsize("temp.txt")
                
#                 if current_modification_time != last_modification_time or current_file_size > last_file_size:
#                     with open("temp.txt") as f:
#                         f.seek(last_file_size)
#                         new_content = f.read()
#                         self.train_textbox.insert("end", new_content.replace('\b', ''))                        
#                         if self.train_textbox.yview()[1] > 0.9:
#                             self.train_textbox.see("end")
#                         self.train_textbox.update()

#                     last_modification_time = current_modification_time
#                     last_file_size = current_file_size
                
#                 time.sleep(0.05)
#         finally:
#             sys.stdout = sys.__stdout__
#             sys.stderr = sys.__stderr__
#             print("Training complete")
#             # Delete the temporary file
#             os.remove("temp.txt")