# ----- Voon Tao ----- #
import os
from datetime import datetime

class MinutesManager():
    def createMinutes(self,start_date,start_time):
        
        global filename
        global filepath

        date_time_str = f'{start_date} {start_time}' # 18/09/19 01:55:19
        date_time_obj = datetime.strptime(date_time_str, '%m/%d/%y %H:%M:%S') # 2019-09-18 01:55:19
        report_title = f'{date_time_obj} Meeting Minutes\n'

        start_date_time = str(date_time_obj).replace(":", "") # avoid create filename with ':'
        filename = f"{start_date_time}.txt"

        filepath = os.path.join('Minutes', filename)
        f = open(filepath,"w")

        # add title to first line 
        f = open(filepath,"a")
        f.write(report_title)
        f.close()

    def recordMinutes(self,text):
        global filename
        global filepath
        f = open(filepath,"a")
        currTime = (datetime.now()).strftime("%H:%M:%S")
        text = f"[{currTime}] {text}\n"
        f.write(text)
        f.close()

    def getFilePath(self):
        global filepath
        return filepath