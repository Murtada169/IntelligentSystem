#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 21:02:43 2022

@author: iman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:24:47 2022

@author: iman
"""

from tkinter import *
from tkcalendar import *
import pygame
import time
import speech_recognition as sr 

# ----- Voon Tao ----- #
from MinutesManager import MinutesManager

ws = Tk()
ws.title("Smart Meeting Minutes")
ws.geometry("820x600")

hour_string=StringVar()
min_string=StringVar()
last_value_sec = ""
last_value = ""        
f = ('Arial', 40)

# ----- Voon Tao ----- #
def createTxtMinutes(start_date,start_time):
    manager = MinutesManager()
    manager.createMinutes(start_date,start_time)
# ----- Voon Tao ----- #

def display_msg():
    date = cal.get_date()
    m = min_sb.get()
    h = sec_hour.get()
    s = sec.get()
    t = f"Your appointment is booked for {date} at {m}:{h}:{s}."
    createTxtMinutes(date,f'{m}:{h}:{s}') # create txt file minutes
    msg_display.config(text=t)
    ws.destroy()
    import options

if last_value == "59" and min_string.get() == "0":
    hour_string.set(int(hour_string.get())+1 if hour_string.get() !="23" else 0)   
    last_value = min_string.get()

if last_value_sec == "59" and sec_hour.get() == "0":
    min_string.set(int(min_string.get())+1 if min_string.get() !="59" else 0)
    
if last_value == "59":
    hour_string.set(int(hour_string.get())+1 if hour_string.get() !="23" else 0)            
    last_value_sec = sec_hour.get()

fone = Frame(ws)
ftwo = Frame(ws)

fone.pack(pady=50)
ftwo.pack(pady=10)

cal = Calendar(
    fone, 
    selectmode="day", 
    year=2021, 
    month=2,
    day=3
    )
cal.pack()

min_sb = Spinbox(
    ftwo,
    from_=0,
    to=23,
    wrap=True,
    textvariable=hour_string,
    width=2,
    state="readonly",
    font=f,
    justify=CENTER
    )

sec_hour = Spinbox(
    ftwo,
    from_=0,
    to=59,
    wrap=True,
    textvariable=min_string,
    font=f,
    width=2,
    justify=CENTER
    )

sec = Spinbox(
    ftwo,
    from_=0,
    to=59,
    wrap=True,
    textvariable=sec_hour,
    width=2,
    font=f,
    justify=CENTER
    )

min_sb.pack(side=LEFT, fill=X, expand=True)
sec_hour.pack(side=LEFT, fill=X, expand=True)
sec.pack(side=LEFT, fill=X, expand=True)

msg = Label(
    ws, 
    text="Hour            Minute           Seconds",
    font=("Arial", 12)
    )

msg.pack(side=TOP)

actionBtn = Button(
    ws,
    text="Smart Meeting Minutes",
    padx=10,
    pady=10,
    command=display_msg
)

actionBtn.pack(pady=10)

msg_display = Label(
    ws,
    text=""
)

msg_display.pack(pady=10)

ws.mainloop()
