import customtkinter
from glob import glob
from model.config import HISTORY_FORMAT, MODELS_PATH
from model.model import load_history
from model.visualization import print_plot_history
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class PerformanceFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.model_name = None
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.first_frame = customtkinter.CTkFrame(self)
        self.second_frame = customtkinter.CTkFrame(self)

        self.first_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.second_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.second_frame.grid_rowconfigure(0, weight=1)
        self.second_frame.grid_columnconfigure(0, weight=1)

        self.model_label = customtkinter.CTkLabel(self.first_frame, text="Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)

        models_list = [model.split('/')[-1] for model in sorted(glob(MODELS_PATH + '*'))]
        self.model_combobox = customtkinter.CTkComboBox(self.first_frame, values=models_list, command=lambda x:self.history_callback(self))
        self.model_combobox.set("Select a model")
        self.model_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.history_label = customtkinter.CTkLabel(self.first_frame, text="")
        self.history_label.grid(row=0, column=2, padx=10, pady=10)

    def history_callback(self, x):
        self.model_name = self.model_combobox.get()
        self.history_label.configure(text=f'Loading history of {self.model_name} ...')
        try:
            history = load_history(self.model_combobox.get())
            fig = print_plot_history(history)
            canvas = FigureCanvasTkAgg(fig, master=self.second_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.history_label.configure(text=f'History of {self.model_name} loaded successfully !')
        except:
            canvas = FigureCanvasTkAgg(matplotlib.figure.Figure(), master=self.second_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.history_label.configure(text=f'No history found for {self.model_name}. Make sure ' + HISTORY_FORMAT + f' exists in {self.model_name} folder.')
            

