from model.data import load_data, load_models
from view.app import App

def main():
    load_data()
    load_models()
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()