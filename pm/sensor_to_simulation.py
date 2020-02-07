
import paho.mqtt.client as mqtt
import time
import pandas as pd

# #%%
# def reading(date_file):
#     file = open("data/"+date_file+".txt", "r")
#     data = []
#     index = 1
#     for ligne in file: 
#         tab = ligne.split('\t')
#         last = tab[1]
#         if(last[len(last)-1] == '\n'):
#             data.append([tab[0], date_file+" "+last[:len(last)-1]])
#         else:
#             data.append([tab[0], date_file+" "+last])
#     file.close()
#     #data.reverse()
#     return data

# #%%
# client = mqtt.Client()

# client.connect("localhost", 1883, 60)
# mydata = reading("2018-11-27")

# print(mydata)

# for i in mydata:
#    payload = "data="+i[0]+"&date="+i[1]
#    client.publish("temperature", payload, 0, False)
#    time.sleep(0.5)

client = mqtt.Client()
client.connect("localhost", 1883, 60)
#%%
def uploaddingFile(filename):
    df = pd.read_csv(filename+".csv")
    return df["temperature"].iloc[::-1]


def sendToGama(data, index):
    payload = "datawithallinfo="+data
    # payload2 = "datawithallinfo="+data2
    # payload3 = "datawithallinfo="+data3
    # payload4 = "datawithallinfo="+data4
    print("pushing i="+data+" to Gama reste=>"+index)
    client.publish("temperature", payload, 0, False)
    # client.publish("temperature2", payload2, 0, False)
    # client.publish("temperature3", payload3, 0, False)
    # client.publish("temperature4", payload4, 0, False)
    time.sleep(1)

def run(filename ):
    data = uploaddingFile(filename)
    total = 795
    index = 1
    for i in data:
        sendToGama(str(i),str(total-index))
        index = index+1
    print("finish")



#%%

#run("/Users/admin/Documents/ML/my_data")
run("/Users/admin/Documents/ML/Thesis/data/temp_data/my_data")

