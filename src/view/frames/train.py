import customtkinter # type: ignore

class TrainFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        batch_size_label = customtkinter.CTkLabel(self, text="Batch size:")
        batch_size_label.grid(row=0, column=0, padx=10, pady=10)
        batch_size_combobox = customtkinter.CTkComboBox(self, values=["8", "16", "32", "64"])
        batch_size_combobox.set("32")
        batch_size_combobox.grid(row=0, column=1, padx=10, pady=10)

        epochs_label = customtkinter.CTkLabel(self, text="Epochs:")
        epochs_label.grid(row=0, column=2, padx=10, pady=10)
        epochs_entry = customtkinter.CTkEntry(self)
        epochs_entry.grid(row=0, column=3, padx=10, pady=10)
        epochs_entry.insert(0, "20")

        train_button = customtkinter.CTkButton(self, text="Train")
        train_button.grid(row=0, column=4, padx=10, pady=10)

        train_textbox = customtkinter.CTkTextbox(self, width=250)
        train_textbox.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        train_textbox.insert("0.0", "CTkTextbox\n\n" + "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.\n\n" * 20)