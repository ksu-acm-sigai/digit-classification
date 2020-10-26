import tkinter as tk


def draw(event):
    print("Received left-click event")
    canvas.create_oval(event.x, event.y, event.x + 25, event.y + 25, outline="#000000", fill="#ff0000", width=2)

top = tk.Tk()

canvas = tk.Canvas(top, width=1280, height=960)
canvas.create_line(15, 25, 200, 25)
canvas.pack()
canvas.bind('<Button-1>', draw)

top.mainloop()
