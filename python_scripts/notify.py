import os 

def notify():
    os.system('echo "Your script has run." | mail -s "Script Notification" cl2729@york.ac.uk')

