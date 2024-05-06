import customtkinter
from glob import glob
from model.config import MODELS_PATH
from model.model import load_history
from model.visualization import print_plot_history
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class PerformanceFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.first_frame = customtkinter.CTkFrame(self)
        self.second_frame = customtkinter.CTkFrame(self)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.first_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.second_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)

        models_list = sorted(glob(MODELS_PATH + '*'))
        models_list = [model.split('/')[-1] for model in models_list]
        self.model_combobox = customtkinter.CTkComboBox(self.first_frame, values=models_list, command=lambda x:self.history_callback(self))
        self.model_combobox.set("Select a model")
        self.model_combobox.grid(row=0, column=1, padx=10, pady=10)

    def history_callback(self, x):
        history = load_history(self.model_combobox.get())
        fig = print_plot_history(history)
        canvas = FigureCanvasTkAgg(fig, master=self.second_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

