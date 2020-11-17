import tkinter as tk
import pandas as pd

from skincare import p,some_predict,selected_columns
from tkinter.filedialog import askopenfile 
from tkinter import *
window =tk.Tk()

entry =tk.Entry()
text_box =tk.Text()

product =StringVar(window)
product_entry = tk.Entry()

retail_entry=tk.Entry()


def open_file(): 
    file = askopenfile(mode ='r') 
    s = pd.read_csv(file)
    open_file.pred_s = s[selected_columns].copy()
    pred_s = s[selected_columns].copy()

    text_box.insert("1.0",pred_s)

   
    if file is not None: 
        content = file.read() 
        print(content) 
    

        
def predict():
   


    product = product_entry.get()
    retail = retail_entry.get()
    standards = standard.get()
    data = {'Slno':[2],'Product':[product],'Standard':[standards],'retailprice':[retail]}
    df = pd.DataFrame(data)
    prediction = p.predict(df)
    print(product,retail)
    text_box.insert(tk.END,prediction)


standard = StringVar(window)
standard.set("30ml") # initial value

option = OptionMenu(window, standard, "20ml", "30ml", "40ml", "50ml","60ml","110ml","175ml","200ml")
option.pack()

btn = tk.Button(window, text ='Open', command = lambda:open_file()) 
predictbtn = tk.Button(window, text ='Predict', command = lambda:predict()) 

btn.pack() 
predictbtn.pack()
  

text_box.pack()
product_entry.pack()
retail_entry.pack()

window.mainloop()
