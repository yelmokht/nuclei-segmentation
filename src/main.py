from model.data import load_data
from view.app import App

def main():
    load_data()
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()