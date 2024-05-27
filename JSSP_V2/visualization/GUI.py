from visualization.Gantt import Gantt
import io
from tkinter import *
from PIL import Image, ImageTk

simmode = ''

class GUI():
    def __init__(self, image_bytes):
        self.tk = Tk()
        self.tk.title("Job Shop Scheduler - Hyunjin Oh")
        self.tk.geometry("1280x720")
        self.tk.resizable(False, False)
        self.image_bytes = image_bytes
        self.image = Image.open(io.BytesIO(self.image_bytes)) # 0으로 초기화하긴 했지만 byte가 들어올거라 괜찮음
        self.tk_images = ImageTk.PhotoImage(self.image)

        self.frame1 = LabelFrame(self.tk, text=simmode)
        self.frame1.grid(column=0, row=0)
        self.gantt = Label(self.frame1, text=simmode)
        self.gantt.grid(column=0, row=0, sticky=N + E + W + S)
        self.gantt.config(image=self.tk_images)
        self.tk.mainloop()


class GUI_Update():
    def __init__(self, machine_log, config):
        self.config = None
        self.RANDOM_SEED = 42
        self.tk = Tk()
        self.tk.title("Job Shop Scheduler - Hyunjin Oh")
        self.tk.geometry("1680x720+400+200")
        self.tk.resizable(False, False)
        self.image_bytes = [1 for i in range(self.config.n_show)]
        self.image = [1 for i in range(self.config.n_show)]
        self.tk_images = [1 for i in range(self.config.n_show)]
        self.current_image = 1

        for i in range(1,self.config.n_show):
            self.image_bytes[i] = Gantt(machine_log, i)
            self.image[i] = Image.open(io.BytesIO(self.image_bytes[i])) # 0으로 초기화하긴 했지만 byte가 들어올거라 괜찮음
            self.tk_images[i] = ImageTk.PhotoImage(self.image[i])

        self.frame1 = LabelFrame(self.tk, text=simmode)
        self.frame1.grid(column=0, row=0)
        self.gantt = Label(self.frame1, text=simmode)
        self.gantt.grid(column=0, row=0, sticky=N + E + W + S)

        self.update()
        self.tk.mainloop()

    def update(self):

        total_images = self.config.n_op # slide show처럼 돌릴 이미지 개수

        self.gantt.config(image=self.tk_images[self.current_image])

        if self.current_image < self.config.n_op:
            self.current_image += 1
        # self.current_image = (self.current_image + 1) % total_images  # Cycle through images

        # tk.after는 인자로 받은 함수를 계속해서 callback함
        # 재귀적으로 호출함으로써 계속 동작하게 만들 수 있음
        # while self.current_image<n_show:

        if self.current_image == 1:
            self.tk.after(self.config.show_interval_time, self.update)  # Pause for 2000 milliseconds (2 seconds)
        else:
            self.tk.after(self.config.finished_pause_time, self.update)  # Pause for 2000 milliseconds (2 seconds)

