#1 

def total_euro(sati, satnica):
    return sati*satnica

sati = input('radni sati: ')
satnica = input('eura/h: ')

sati = float(sati)
satnica = float(satnica)

ukupno = total_euro(sati, satnica)
print("Ukupno: ", ukupno)

#2

try:
    score = float(input("unesi ocjeni izmedju 0.0 i 1.0: "))

    if 0.0 <= score <= 1.0:
        if score >= 0.9:
            print("A")
        elif score >= 0.8:
            print("B")
        elif score >= 0.7:
            print("C")
        elif score >= 0.6:
            print("D")
        else:
            print("F")
    else:
        print("uneseni broj je izvan intervala")
except ValueError:

    print("unos nije broj")

#3 

num = []
while True:
    unos = input("unesi broj ili 'Done' za kraj: ")
    if unos == "Done":
        break
    try:
        broj = float(unos)
        num.append(broj)
    except ValueError:
        print("nisi unio broj")

if num:
    print(f"ukupno brojeva: {len(num)}")
    print(f"sr vrj brojeva: {sum(num) / len(num)}")
    print(f"min vrj brojeva: {min(num)}")
    print(f"max vrj brojeva: {max(num)}")

    num.sort()
    print(num)
else:
    print("nije unesen broj")

#4

def izracunaj_srj_spam_confidence(ime_fajla):
    try:
        with open(ime_fajla, 'r') as file:
            total_confidence = 0
            brojac = 0
            for line in file:
                if line.startswith('X-DSPAM-Confidence: '):

                    confidence = float(line.strip().split(':')[1])
                    total_confidence+=confidence
                    brojac += 1
            if brojac > 0 :
                srj_confidence = total_confidence / brojac
                print("srednja X-DSPAM-Confidence: ", srj_confidence)
            else:
                print("nema 'X-DSPAM-Confidence:' ")
    except FileNotFoundError:
        print("file nije pronadjen")
    except Exception as e:
        print("doslo je do errora:", e)
    finally:
        try:
            file.close()
        except:
            pass

ime_fajla = input("ime datoteke: ")
izracunaj_srj_spam_confidence(ime_fajla)

#5

def ucitaj_i_prebroji_rijeci(ime_datoteke):
    try:
        with open(ime_datoteke, 'r', encoding='utf-8') as datoteka:
            brojac_rijeci = {}
            for line in datoteka:
                rijeci = line.strip().split()
                for rijec in rijeci:
                    rijec = rijec.strip(',').lower()
                    if rijec:
                        if rijec in brojac_rijeci:
                            brojac_rijeci[rijec] += 1
                        else:
                            brojac_rijeci[rijec] = 1
            return brojac_rijeci
    except FileNotFoundError:
        print(f"datoteka '{ime_datoteke}' nije pronadena")
        return None

def ispisi_rijeci_samo_jednom(brojac_rijeci):
    rijeci_samo_jednom = [rijec for rijec, broj in brojac_rijeci.items() if broj == 1]
    print(f"broj rijeci koje se pojavljuju jednom: {len(rijeci_samo_jednom)}")
    print("ove rijecu su:")
    for rijec in rijeci_samo_jednom:
        print(rijec)


ime_datoteke = 'song.txt'
brojac_rijeci = ucitaj_i_prebroji_rijeci(ime_datoteke)
if brojac_rijeci is not None:
    ispisi_rijeci_samo_jednom(brojac_rijeci)

#6

def izracunaj_statistiku(file):
    broj_ham_poruka = 0
    broj_spam_poruka = 0
    ukupno_rijeci_ham = 0
    ukupno_rijeci_spam = 0
    spam_zavrsava_usklicnikom = 0

    with open(file, 'r', encoding='utf-8') as f:
        for linija in f:
            tip, poruka = linija.split('\t', 1)
            broj_rijeci = len(poruka.split())

            if tip == 'ham':
                broj_ham_poruka += 1
                ukupno_rijeci_ham += broj_rijeci
            elif tip == 'spam':
                broj_spam_poruka += 1
                ukupno_rijeci_spam += broj_rijeci
                if poruka.strip().endswith('!'):
                    spam_zavrsava_usklicnikom += 1

    prosjek_rijeci_ham = ukupno_rijeci_ham / broj_ham_poruka if broj_ham_poruka > 0 else 0
    prosjek_rijeci_spam = ukupno_rijeci_spam / broj_spam_poruka if broj_spam_poruka > 0 else 0

    return prosjek_rijeci_ham, prosjek_rijeci_spam, spam_zavrsava_usklicnikom

datoteka = 'SMSSpamCollection.txt'
prosjek_ham, prosjek_spam, broj_spam_usklicnik = izracunaj_statistiku(datoteka)

print(f"Prosjek riječi u ham porukama: {prosjek_ham:.2f}")
print(f"Prosjek riječi u spam porukama: {prosjek_spam:.2f}")
print(f"Broj spam poruka koje završavaju uskličnikom: {broj_spam_usklicnik}")
