from serial import Serial
serial_port = Serial(port='/dev/cu.usbserial-1420', baudrate=9600)
fichier = open("data_pm.txt","w")
header = 0;
while True:
    message = serial_port.read_until(b'\r')
    ele = message.decode("utf-8")
    print(ele)
    if header!= 0:
        el = ele.replace(':',',')
        fichier.write(el)
    else:
        entete = "PM1,PM25,PM10"
        fichier.write(entete)
    header+=1
    #formation initiale
    