import tkinter as tk

CELL_SIZE = 16
GRID_SIZE = 32
RESOLUTION = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)

grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
top = tk.Tk()
canvas = tk.Canvas(top, width=RESOLUTION[0], height=RESOLUTION[1])


def get_binary_array(arr):
    binary_arr = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] != -1:
                binary_arr[i][j] = 1

    return binary_arr


def snap(x, y):
    new_x = (x // CELL_SIZE) * CELL_SIZE
    new_y = (y // CELL_SIZE) * CELL_SIZE
    return new_x, new_y


def draw(event):
    x, y = snap(event.x, event.y)
    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
    grid[grid_y][grid_x] = canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="black", outline='')
    print(type(grid[grid_y][grid_x]))


def erase(event):
    x, y = snap(event.x, event.y)
    grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
    canvas.delete(grid[grid_y][grid_x])
    canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="white", outline='')
    grid[grid_y][grid_x] = -1


if __name__ == '__main__':
    canvas.pack()
    canvas.configure(bg='white')
    canvas.bind('<Button-1>', draw)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<Button-3>', erase)
    canvas.bind('<B3-Motion>', erase)

    top.mainloop()
