import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Python 3.7.9
# Keras 2.3.1
# Tensorflow 1.14.0
# Numpy 1.17.4
# Matplotlib 3.3.4
# OpenCV 4.5.1.48
# sklearn

# potrebne biblioteke
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import glob

# Obrada slike uz pomocu cv2 (openCV) biblioteke
def processImage(imagePath):
    # Citanje file-a iz zadane putanje kao NumPy niz red(visina) x stupac(sirina) x boja(3) 
    # (redoslijed boja je tada BGR)
    image = cv2.imread(imagePath) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Pretvorba iz BGR u RGB redoslijed
    image = image / 255 # Pretvorba u raspon 0-1 da bi odgovarala matplotlib

    return image

# Ucitavanje slike tablice nad kojom ce se izvrsiti segmentacija
LicencePlate = processImage("PlateExamples/1.jpg")
#LicencePlate = processImage("PlateExamples/2.jpg")
#LicencePlate = processImage("PlateExamples/3.jpg")
#LicencePlate = processImage("PlateExamples/4.jpg")
#LicencePlate = processImage("PlateExamples/5.jpg")
#LicencePlate = processImage("PlateExamples/6.jpg")
#LicencePlate = processImage("PlateExamples/7.jpg")
#LicencePlate = processImage("PlateExamples/8.png")
#LicencePlate = processImage("PlateExamples/9.png")
#LicencePlate = processImage("PlateExamples/10.jpg")
#LicencePlate = processImage("PlateExamples/11.png")
#LicencePlate = processImage("PlateExamples/12.jpg")

# Prikaz originalne slike koristeci matplotlib biblioteku
fig = plt.figure(figsize=(12,6))
plt.axis(False)
plt.imshow(LicencePlate)

# Pretvorba slike u 8-bitnu 'skalu', gdje je faktor skaliranja alpha
# 3 bita za crvenu, 3 bita za zelenu i 2 bita za plavu boju (rgb)
# Smanjujuci vrijednost alphe rezultantna slika postaje sve tamnija (alpha = 0.0 - Slika je cista crna 
# pozadina)
LicencePlateImage = cv2.convertScaleAbs(LicencePlate, alpha=(350.0))

# Boja nije nuzna za prepoznavanje tablice pa je uklanjamo sa slike i pretvaramo u sivu - 'grayscale'
# Tehnika zamucenja - 'blur' provodi za radi uklanjanja smetnji (buke) i nebitnih informacija
# Gaussian blur je jedna od vrsta zamucenja, a velicina jezgre (7,5) se moze promijeniti s obzirom na sliku 
# Velicina jezgre (sirina, visina) treba biti pozitivna i neparna
# Parametar koji slijedi iza velicine jezgre je standardno odstupanje u smjerovima X i Y, ondosno samo u 
# smjeru X, a Y se podrazumijeva da je 0 ako nije naveden
# S obzirom da su oba odstupanja jednaka 0, izracunavaju se iz velicine jezgre
# Povecavanjem velicine jezgre smanjuje se buka, ali i gubi vise podataka
gray = cv2.cvtColor(LicencePlateImage, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,5),0)

# Inverzno binarno pragiranje - cv2.THRESH_BINARY_INV
# Postavi se vrijednost praga tako da svaki piksel manje vrijednosti od te pretvori u maksimum (255), 
# a svaka veca u 0
# cv2.THRESH_OTSU poziva algoritam koji odabire optimalnu vrijednost praga
binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
# Tehnika prosirenja bijelog podrucja slike - za poboljsanje bijele konture slike
# Traze se 'zrna' pravokutnog oblika funkcijom cv2.getStructuringElement(), kojoj se proslijede oblik i 
# velicina jezgre, a dobije se zeljena jezgra
# U funkciji cv2.morphologyEx() poziva se izvrsavanje tehnike dilatacije
# Ako je barem 1 piksel ispod jezgre '1', tada je element piksela '1'.
# Tom tehnikom se povecava bijela regija na slici ili se povecava velicina predmeta u prvom planu
# Korisno je za spajanje slomljenih dijelova predmeta na slici
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dilation = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

# Prikaz primijenjenih tehnika na pocetnu sliku koristeci biblioteku matplotlib
fig = plt.figure(figsize=(12,7))
plt.rcParams.update({"font.size":18})
grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig) #prikaz tih 5 slika u 3 reda i 2 stupca
plotImage = [LicencePlateImage, gray, blur, binary, dilation]
plotName = ["LicencePlateImage","gray","blur","binary","dilation"]

for i in range(len(plotImage)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.title(plotName[i])
    if i ==0:
        plt.imshow(plotImage[i])
    else:
        plt.imshow(plotImage[i],cmap="gray")

# Dohvaca i sortira utemeljene konture u kojima su 'uhvaceni' znakovi tablice slijeva nadesno 
# (reverse=False, i=0) jer ih je bitno rasporediti u ispravnom redoslijedu
# Prvi argument je lista kontura koje zelimo sortirati, a drugi metoda, tj. redoslijed sortiranja
def sortContours(cnts,reverse = False):
    i = 0
    # Izracun granicnih okvira svake konture, sto su početne(x, y)-koordinate granicnog okvira 
    # pracene sirinom i visinom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # Granicni okviri omogucuju sortiranje stvarnih okvira uz dodani python kod koji zajedno sortira 
    # dva popisa (okvire, konture)
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts # Lista sortiranih kontura

# Funkcija za identifikaciju koordinata znaka
# Kontura je krivulja koja spaja sve kontinuirane tocke koje dijele istu boju i intezitet
# Koristeci cv2.CHAIN_APPROX_SIMPLE algoritam stedi se memorija,
# umjesto da se spremaju sve tocke neke linije konture, spremaju se samo po dvije krajnje tocke linije
cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Izraduje se kopija pocetne ulazne slike, kako bi se na njoj mogli iscrtati svi granicni okviri oko 
# znakova za potrebe vizualizacije
testLicencePlateImage = LicencePlateImage.copy()

# Incijalizacija liste koja ce se koristiti za dodavanje slike znakova
cropedCharacters = []

# Definirane standardna sirina i visina znaka na tablici (obicno je visina vece vrijednosti zbog 
# standardnog oblika slova)
resizeWidth, resizeHeight = 55, 75

for c in sortContours(cont):
    (x, y, w, h) = cv2.boundingRect(c) # Citamo pozicije okvira, visinu i sirinu
    ratio = h/w # Omjer visine i sirine
    # Uzimamo u obzir samo one konture kojima je omjer visina/sirina u rasponu od 1 do 8
    if 1<=ratio<=8:
        # Uzimamo u obzir samo one konture kojima je visina u rasponu 55% do 85% visine slike tablice
        if 0.85>=h/LicencePlateImage.shape[0]>=0.55: 
            # Nacrtati okvir (zelene boje, debljine 3) oko znamenke ili slova
            cv2.rectangle(testLicencePlateImage, (x, y), (x + w, y + h), (0, 255,0), 3)
            # Odvajanje znakova i predvidanje
            currentCharacter = dilation[y:y+h,x:x+w]
            currentCharacter = cv2.resize(currentCharacter, dsize=(resizeWidth, resizeHeight))
            # Inverzno binarno pragiranje
            _, currentCharacter = cv2.threshold(currentCharacter, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Dodavanje svih segmentiranih znakova u listu
            cropedCharacters.append(currentCharacter)

print("[INFO] Detected {} letters...".format(len(cropedCharacters)))

# Prikaz tablice s okvirima oko svakog znaka
fig = plt.figure(figsize=(10,6))
plt.axis(False)
plt.imshow(testLicencePlateImage)

# Ucitavaju se koristena NN arhitektura modela 'MobileNets' i vec dobivene tezine nakon zavrsene faze 
# treniranja uz klase znakova
# Zato se unutar foldera projekta nalaze MobileNets_character_recognition JSON file, 
# License_character_recognition_weight H5 datoteka i license_character_classes NPY file
# NN koristi set podataka slova i brojeva, koji je podijeljen na set treninga(90%) i set za validaciju(10%),
# sto omogucuje izbjegavanje overfittanja i pracenje tocnosti modela
jsonFile = open('MobileNets_character_recognition.json', 'r')
loadedModel = jsonFile.read()
jsonFile.close()
model = model_from_json(loadedModel)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

# Ulazni sloj modela NN je konfiguriran za primanje slika u obliku(80,80,3) pa se i nasa slika pretvara u 
# takav oblik
def predictFromModel(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    # Implementacija ucitanih klasa znakova(labels) u inverzne jednokratne oznake kodiranja dobivene 
    # od modela do digitalnih znakova
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

# Prikaz odvojenih znakova uz predikciju svakog znaka iznad konture koristeci matplotlib
plt.style.use('classic')
fig = plt.figure(figsize=(15,3))
cols = len(cropedCharacters)
grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

# Petlja se generira nad svakom slikom znaka iz cropedCharacters
# Isrtava se slika svakog od znakova tablice s pripadajucim predvidanjima
licencePlateCharacters = ''
for i,character in enumerate(cropedCharacters):
    fig.add_subplot(grid[i])
    title = np.array2string(predictFromModel(character,model,labels)) # Poziv funkcije za predvidanje
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    licencePlateCharacters+=title.strip("'[]")
    plt.axis(False)
    plt.imshow(character,cmap='gray')

print("\n\n\n[INFO] Result:",licencePlateCharacters) # ispis konacnog predvidenog niza znakova cijele tablice

plt.show()