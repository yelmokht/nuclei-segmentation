import tkinter as tk

def say_hello():
    print("Hello, tkinter!")

# Create the main application window
root = tk.Tk()

# Add a label to the window
label = tk.Label(root, text="Welcome to tkinter!")
label.pack()

# Add a button to the window
button = tk.Button(root, text="Say Hello", command=say_hello)
button.pack()

# Run the Tkinter event loop
root.mainloop()
