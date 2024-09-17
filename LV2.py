#1 

import matplotlib.pyplot as plt
import numpy as np


x =np.array([1,2,3,3,1])
y =np.array([1,2,2,1,1])


plt.xlabel('x os')
plt.ylabel('y os')
plt.axis((0,4,0,4))
plt.plot(x, y, 'b', linewidth=5, marker=".", markersize=25, markerfacecolor='0.5', color='black')

plt.title('Primjer')


plt.show()

#2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = np.loadtxt(open(r"C:\Users\Filip\IdeaProjects\psuLV2\drugi\mtcars.csv", "rb"), usecols=(1,4,6,2),
                  delimiter=",", skiprows=1) 


mpg =data[:, 0] 
hp = data[:, 1] 
wt=data[:, 2] 
cyl=data[:,3]

plt.scatter(mpg, hp, s=wt*15) 
plt.xlabel("mpg")
plt.ylabel("hp")
plt.title("ovisnost potrosnje automobila o konjskim snagama")
plt.show()

min_mpg=mpg.min()
max_mpg=mpg.max()
avg_mpg=mpg.mean()

print(min_mpg)
print(max_mpg)
print(avg_mpg)
print("\n")

cilindri6 = np.where(cyl == 6)[0]
mpg6= mpg[cilindri6]

min_mpg6=np.min(mpg6)
max_mpg6=np.max(mpg6)
avg_mpg6=np.mean(mpg6)

print(min_mpg6)
print(max_mpg6)
print(avg_mpg6)

#3

import numpy as np
import matplotlib.pyplot as plt
img = plt.imread("tiger.png")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)

broj = 150
brightness = img + broj 
brightness = np.clip(brightness, 0, 255) 

plt.figure() #prikaz slike
plt.imshow(brightness, cmap="gray")
plt.title("brightness")


plt.figure()
plt.imshow(img, cmap="gray")
plt.title("oridžidži")

rotirana = np.rot90(img, k= 1) 
plt.figure()
plt.imshow(rotirana, cmap="gray")
plt.title("rotirano za 90")

mirror = img[:, ::-1] 
plt.figure()
plt.imshow(mirror, cmap="gray")
plt.title("zrcaljena slika")

length = img.shape[0]
height = img.shape[1]

smanjena = img.reshape(img.shape[0] // 10, 10, img.shape[1] // 10, 10).mean(axis=(1,3)) 

plt.figure()
plt.imshow(smanjena, cmap="gray")
plt.title("smanjena slika")

druga_cetvrtina = img[0 : img.shape[0], img.shape[1] // 4 : img.shape[1] // 2].copy()

plt.show()

#4

import numpy as np
import matplotlib.pyplot as plt

def ploca(kvadrat, redak, stupac):
    crna = np.zeros((kvadrat, kvadrat))
    bijela = np.ones((kvadrat, kvadrat)) * 255

    redak1 = np.hstack([crna, bijela] * (stupac // 2))
    if stupac % 2 != 0:
        redak1 = np.hstack([redak1, crna])

    redak2 = np.hstack([bijela, crna] * (stupac // 2))
    if stupac % 2 != 0:
        redak2 = np.hstack([redak2, bijela])

    matrix = np.vstack([redak1, redak2] * (redak // 2))
    if redak % 2 != 0:
        matrix = np.vstack([matrix, redak1])

    return matrix

img = ploca(50, 4, 5)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
