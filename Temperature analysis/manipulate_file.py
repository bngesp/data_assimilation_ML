#ma_liste=[]
data = []
fichier = open("data.txt","r")
#output = open("output.txt", "w")
index = 0
for ligne in fichier:
    if(index>0):
        b=ligne.split('\t')
        if(b[1] != '\t'):
            temp = open(b[1]+".txt", "a")
            temp.write(b[0]+"\t"+b[2])
            temp.close()
    index=index+1

print(data)
 
 
fichier.close()
#output.close()

#%%
import pandas as pd


#%%
df = pd.read_excel('/Users/admin/Documents/ML/Thesis/data.xlsx')
df['date'].head
df.to_csv("my_data.csv")
#%%
for i in range(30, 31):
    path = "2019-08-"+str(i)
    a = df[df['date']==path]
    a.to_csv(path+".csv")
    

#%%
