from model.data import load_data
from view.app import App
from model.model import train

def main():
    load_data()
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()