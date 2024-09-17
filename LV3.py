#1

import pandas as pd
import numpy as np

df = pd.read_csv('mtcars.csv')

print("Top 5 automobila s najvecom potrosnjom:")
print(df[['car', 'mpg']].sort_values('mpg').tail(5))

print("\nTri automobila s 8 cilindara i najmanjom potrosnjom:")
print(df[['car', 'mpg', 'cyl']][df.cyl == 8].sort_values('mpg').head(3))

cil6_srednja_potrosnja = df[df.cyl == 6].mpg.mean()
print("\nSrednja potrosnja automobila s 6 cilindara:")
print(df[df.cyl == 6].mpg.mean())

cil4_srednja_potrosnja = df[(df.cyl == 4) & (df.wt >= 2.0) & (df.wt <= 2.2)].mpg.mean()
print("\nSrednja potrosnja automobila s 4 cilindra mase izmedu 2000 i 2200 lbs:")
print(cil4_srednja_potrosnja)

broj_rucni_mjenjac = df[df['am'] == 1].shape[0]
broj_automatski_mjenjac = df[df['am'] == 0].shape[0]
print("\nBroj automobila s rucnim mjenjacem: {}".format(broj_rucni_mjenjac))
print("Broj automobila s automatskim mjenjacem: {}".format(broj_automatski_mjenjac))

print("\nBroj automobila s automatskim mjenjacem i snagom preko 100 konjskih snaga:")
print(df[(df.am == 0) & (df.hp> 100)].shape[0])

df['wt_kg'] = df['wt'] * 1000
print("\nMasa svakog automobila u kilogramima:")
print(df[['car', 'wt', 'wt_kg']])

#2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mtcars.csv')

potrosnja_po_cilindrima = df.groupby('cyl')['mpg'].mean()
potrosnja_po_cilindrima.plot(kind='bar', rot=0)
plt.title('Potrošnja automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja (mpg)')
plt.show()

df[df['cyl'].isin([4, 6, 8])].boxplot(column='wt', by='cyl')
plt.title('Težina automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

potrosnja_po_mjenjacu = df.groupby('am')['mpg'].mean()
potrosnja_po_mjenjacu.plot(kind='bar', rot=0)
plt.title('Potrošnja automobila po vrsti mjenjača')
plt.xlabel('Vrsta mjenjača (0 - automatski, 1 - ručni)')
plt.ylabel('Potrošnja (mpg)')
plt.xticks(ticks=[0, 1], labels=['Automatski', 'Ručni'])
plt.show()

ubrzanje = df['qsec']
snaga = df['hp']
mjenjac = df['am']
colors = np.where(mjenjac == 1, 'r', 'b')

plt.scatter(snaga, ubrzanje, c=colors)
plt.title('Odnos ubrzanja i snage automobila')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (1/4 mile)')
plt.legend(['Ručni mjenjač', 'Automatski mjenjač'])
plt.show()

#3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mtcars.csv')

potrosnja_po_cilindrima = df.groupby('cyl')['mpg'].mean()
potrosnja_po_cilindrima.plot(kind='bar', rot=0)
plt.title('Potrošnja automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja (mpg)')
plt.show()

df[df['cyl'].isin([4, 6, 8])].boxplot(column='wt', by='cyl')
plt.title('Težina automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

potrosnja_po_mjenjacu = df.groupby('am')['mpg'].mean()
potrosnja_po_mjenjacu.plot(kind='bar', rot=0)
plt.title('Potrošnja automobila po vrsti mjenjača')
plt.xlabel('Vrsta mjenjača (0 - automatski, 1 - ručni)')
plt.ylabel('Potrošnja (mpg)')
plt.xticks(ticks=[0, 1], labels=['Automatski', 'Ručni'])
plt.show()

ubrzanje = df['qsec']
snaga = df['hp']
mjenjac = df['am']
colors = np.where(mjenjac == 1, 'r', 'b')

plt.scatter(snaga, ubrzanje, c=colors)
plt.title('Odnos ubrzanja i snage automobila')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (1/4 mile)')
plt.legend(['Ručni mjenjač', 'Automatski mjenjač'])
plt.show()
