
import paho.mqtt.client as mqtt
import time
import pandas as pd

client = mqtt.Client()
client.connect("localhost", 1883, 60)
#%%
def uploaddingFile(filename):
    df = pd.read_csv(filename+".csv")
    return df["pm10"], df["pm25"], df["pm1"] 


def sendToGama(data, data1, data2, index):
    payload = "mydatais="+data
    payload1 = "mydatais="+data1
    payload2 = "mydatais="+data2
    print("pushing pm10="+data+" to Gama reste=>"+index)
    client.publish("pm10", payload, 0, False)
    client.publish("pm25", payload1, 0, False)
    client.publish("pm1", payload2, 0, False)
    time.sleep(0.84)

def run(filename):
    pm10,pm25,pm1 = uploaddingFile(filename)
    total = 4027
    index = 1
    for i in range(total):
        sendToGama(str(pm10[i]), str(pm25[i]), str(pm1[i]), str(total-index))
        index = index+1
    print("finish")



#%%

#run("/Users/admin/Documents/ML/Thesis/data/PM_Data/2018-01-04")
run("/Users/admin/Documents/ML/Thesis/pm/data/data_pm")

#%%
