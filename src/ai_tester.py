import math
import numpy as np
import customtkinter as ctk
import csv
from tkinter import filedialog

gradient = ["#232323","#5C5C5C","#666666","#707070","#7A7A7A","#8F8F8F","#999999","#A3A3A3","#ADADAD","#B8B8B8","#C2C2C2","#CCCCCC","#D6D6D6","#E0E0E0","#EBEBEB","#F5F5F5","#FFFFFF"]
radius = 0.75
grid_size = 28
box_size = 560 / grid_size

hidden_sizes = [32,32]
input_size = 28*28
output_size = 10

alpha = 0.01


class network:
    def __init__(self):
        self.w0 = np.zeros((hidden_sizes[0],input_size))
        self.w1 = np.zeros((hidden_sizes[1],hidden_sizes[0]))
        self.w2 = np.zeros((output_size,hidden_sizes[1]))
        self.b0 = np.zeros((hidden_sizes[0]))
        self.b1 = np.zeros((hidden_sizes[1]))
        self.b2 = np.zeros((output_size))

    def fromCSV(self,fileName):
        with open(fileName, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            rows = [row for row in reader]

            row_count = 0

            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                for i, num in enumerate(rows[row_count]):
                    self.w0[j, i] = float(num)
                j += 1
                row_count += 1
            
            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                for i, num in enumerate(rows[row_count]):
                    self.w1[j, i] = float(num)
                j += 1
                row_count += 1

            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                for i, num in enumerate(rows[row_count]):
                    self.w2[j, i] = float(num)
                j += 1
                row_count += 1

            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                self.b0[j] = float(rows[row_count][0])
                j += 1
                row_count += 1

            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                self.b1[j] = float(rows[row_count][0])
                j += 1
                row_count += 1

            j = 0
            while row_count < len(rows):
                if "BEGIN" in rows[row_count][0]:
                    row_count += 1
                    break

                self.b2[j] = float(rows[row_count][0])
                j += 1
                row_count += 1

    def leaky_ReLU(self,vec):
        return np.where(vec > 0, vec, vec * alpha)
    
    def softMax(self,vec):
        stable = vec - np.max(vec)
        exp = np.exp(stable)
        return exp / np.sum(exp)


    def neural(self,mat):
        
        a0 = self.leaky_ReLU(self.w0 @ mat + self.b0)
        a1 = self.leaky_ReLU(self.w1 @ a0 + self.b1)
        a2 = self.softMax(self.w2 @ a1 + self.b2)
        sorted = np.sort(a2)[::-1]
        print("________")
        for num in sorted:
            print("{:d} | {:.0f}%".format(np.where(a2 == num)[0][0], num * 100))
        print("________\n\n")
        return np.argmax(a2)

class main_window(ctk.CTk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.geometry("560x560")
        self.canvas = ctk.CTkCanvas()
        self.canvas.place(relx = 0.5, rely = 0.5, relheight = 1, relwidth = 1, anchor = "center")
        self.matrix = np.zeros((grid_size, grid_size))
        self.create_grid()
        self.text = self.canvas.create_text(280,280,font = ('Lato',120), text = "", fill = 'white')
        self.bind("<B1-Motion>", self.on_move)
        self.bind("<Button-1>", self.on_click)
        self.bind("<ButtonRelease>", self.on_release)

        self.net = network()

        file_path = filedialog.askopenfilename()
        if file_path:
            self.net.fromCSV(file_path)

    def f_to_c(self, num):
        return gradient[math.floor(num * (len(gradient) - 1))]

    def create_grid(self):
        self.sqs = []
        for i in range(grid_size):
            self.sqs.append([])
            for j in range(grid_size):
                sq = self.canvas.create_rectangle(i * box_size, j * box_size, (i + 1) * box_size, (j + 1) * box_size, fill = self.f_to_c(self.matrix[j][i]), width=0)
                self.sqs[i].append(sq)

    def update_grid(self):
        for i in range(grid_size):
            for j in range(grid_size):
                self.canvas.itemconfigure(self.sqs[i][j], fill = self.f_to_c(self.matrix[j][i]))

    def on_release(self,event):
        num = self.net.neural(self.prep_matrix(self.matrix).flatten())
        self.canvas.itemconfigure(self.text, text = num)
        self.matrix = np.zeros((grid_size,grid_size))
        self.update_grid()

    def prep_matrix(self,mat):
        mincol = 28
        maxcol = -1
        minrow = 28
        maxrow = -1
        for i in range(28):
            if np.any(mat[i] != 0):
                minrow = min(minrow,i)
                maxrow = max(maxrow,i)
            if np.any(mat[:,i] != 0):
                mincol = min(mincol,i)
                maxcol = max(maxcol,i)

        sub = mat[minrow:(maxrow), mincol:(maxcol)]

        new_mat = np.zeros((28,28))
        
        newminrow = max(0,14 - (maxrow - minrow) // 2)
        newmaxrow = min(28,14 + ((maxrow - minrow) + 1) // 2)
        newmincol = max(0,14 - (maxcol - mincol) // 2)
        newmaxcol = min(28,14 + ((maxcol - mincol) + 1) // 2)

        new_mat[newminrow:(newmaxrow), newmincol:(newmaxcol)] = sub
        return new_mat

    def on_click(self,event):
        self.canvas.itemconfigure(self.text, text = "")

    def on_move(self,event):
        for i in range(grid_size):
            for j in range(grid_size):
                if ((i * box_size - event.x)**2 + (j * box_size - event.y)**2 <= (radius*box_size)**2):
                    self.matrix[j][i] = min(1, self.matrix[j][i] + 0.9)
                elif ((i * box_size - event.x)**2 + (j * box_size - event.y)**2 <= ((radius + 1)*box_size)**2):
                    self.matrix[j][i] = min(1, self.matrix[j][i] + 0.3)
        self.update_grid()
        
if __name__ == "__main__":
    app = main_window()
    app.mainloop()