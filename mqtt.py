#%%
import paho.mqtt.client as mqtt

#%%
client = mqtt.Client()
client.connect("localhost", 1883, 60)

#%%
payload1 = "25C"
client.publish("/temperature", payload1, 0, False)