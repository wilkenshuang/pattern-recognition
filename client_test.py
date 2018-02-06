from xmlrpc.client import ServerProxy
import xmlrpc.client
import cv2
import matplotlib.pyplot as plt
import time
import json
import base64


with open('C:/Users/gd/Desktop/test1.txt') as txtfile:
    string1=txtfile.read()
    txtfile.close()

with open('C:/Users/gd/Desktop/test.txt') as txtfile:
    string=txtfile.read()
    txtfile.close()

with open('C:/Users/gd/Desktop/test2.txt') as txtfile:
    string2=txtfile.read()
    txtfile.close()

#strs=','.join([string,string2,string3])

#f=open('C:/Users/gd/Desktop/test/230.jpg','rb')
#ls_f=base64.b64encode(f.read())

port=8080
server = ServerProxy("http://10.0.0.247:8080")#,verbose=True)

#res=server.shibie(string3,0)
mark=server.RedNum(string2)
print(mark)

#res=server.recognize_multi(strs,)
#print(res)
