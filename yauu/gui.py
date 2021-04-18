from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfile
from tkinter.ttk import Progressbar

from super_res import Upscaler


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.master = master
        self.pack()

        Button(self, text='choose file', command=self.choose_input).pack(side='right')
        Button(self, text='choose model', command=self.choose_model).pack(side='right')

        Button(self, text='run', command=self.run).pack(side='bottom')

        self.progress_bar = Progressbar(self, length=100, mode='determinate')
        self.progress_bar.pack(side='bottom')

    def choose_input(self):
        filepath = askopenfilename()
        self.input_file = filepath

    def choose_model(self):
        filepath = askopenfilename()
        self.model = filepath

    def run(self):
        upscaler = Upscaler(self.input_file, self.model, 4, 64, 2, 'cuda')
        for i in range(len(upscaler)):
            print(i)
            upscaler.upscale_tile(i)
            self.progress_bar['value'] = (i / len(upscaler)) * 100
            self.update_idletasks()
        upscaler.result.save('result_from_gui.png')



root = Tk()
app = Application(master=root)
app.mainloop()
