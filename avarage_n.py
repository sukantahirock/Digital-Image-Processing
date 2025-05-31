import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('lena.png',0)
h,w=img.shape

#Average_Filtering
def fun(i,j,k,mid):
    val=0.0
    for y in range (0,k):
        for z in range (0,k):
            a=i-mid+y
            b=j-mid+z
            val+=float(img[a,b]*mask[y,z])
    return round(val/(k*k))

fig = plt.figure()
fig.add_subplot(3,2,1)
plt.imshow(img)
plt.axis('off')
plt.title("Orginal Image")

g=1
#Kernel
mask=np.ones([3,3], dtype=int)
mask=mask/9

for k in range(1,6):
   img_new= np.zeros([h, w])
   
   for i in range(1, h-1):
    for j in range(1, w-1):
     img_new[i,j]= fun(i,j,3,1)

   sq_sum=0;
   abs_sum=0;
   for i in range (0,h):
       for j in range (0,w):
           sq_sum+=((img[i,j])*(img[i,j]))
           abs_sum+=(img[i,j]-img_new[i,j])*(img[i,j]-img_new[i,j])
           
       
   #MSE
   mse=(abs_sum/(h*w));
   print("MSE of Kernel size", k,"x",k, "is", mse)
   
   #SNR
   snr=20* math.log10(sq_sum/abs_sum)
   print("SNR of Kernel size", k,"x",k, "is", snr)
   
   #PSNR
   psnr=20 * math.log10 (((255*255)*(h*w)) / abs_sum)
   print("PSNR of Kernel size", k,"x",k, "is", psnr)
   print()
   g=g+1
   
   fig.add_subplot(3,2,g)
   plt.imshow(img_new)
   plt.axis('off')
   plt.title(str(k) +" Times")

print('End')
