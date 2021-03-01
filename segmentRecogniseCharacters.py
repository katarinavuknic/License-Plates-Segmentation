import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# potrebne biblioteke
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob

#predobrada slike uz pomoc cv2 (openCV) biblioteke
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path) #citanje file-a iz zadane putanje kao NumPy niz red(visina) x stupac(sirina) x boja(3) (redoslijed boja je tada BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #pretvorba iz BGR u RGB redoslijed
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224)) #postavljanje velicine slike na 224 x 224
    return img

#ucitavanje slike tablice nad kojom ce se izvrsiti segmentacija
#LpImg = preprocess_image("PlateExamples/1.jpg")
#LpImg = preprocess_image("PlateExamples/2.jpg")
#LpImg = preprocess_image("PlateExamples/3.jpg")
#LpImg = preprocess_image("PlateExamples/4.jpg")
#LpImg = preprocess_image("PlateExamples/5.jpg")
#LpImg = preprocess_image("PlateExamples/6.jpg")
#LpImg = preprocess_image("PlateExamples/7.jpg")
#LpImg = preprocess_image("PlateExamples/8.jpg")
#LpImg = preprocess_image("PlateExamples/9.png")
#LpImg = preprocess_image("PlateExamples/10.png")
#LpImg = preprocess_image("PlateExamples/11.png")
#LpImg = preprocess_image("PlateExamples/12.jpg") #slika ima neko 'nevidljivo' zamucenje u desnom kraju
#LpImg = preprocess_image("PlateExamples/13.jpg")
LpImg = preprocess_image("PlateExamples/15.png")


fig = plt.figure(figsize=(12,6))
plt.axis(False)
plt.imshow(LpImg) #prikaz originalne slike koristeci matplotlib biblioteku

#pretvorba slike u 8-bitnu 'skalu', gdje je faktor skaliranja alpha
#3 bita za crvenu, 3 bita za zelenu i 2 bita za plavu boju (rgb)
#smanjujuci vrijednost alphe rezultantna slika postaje sve tamnija(alpha=0.0 - slika je cista crna pozadina)
plate_image = cv2.convertScaleAbs(LpImg, alpha=(255.0))

#boja nije nuzna za prepoznavanje tablice pa je uklanjamo sa slike i pretvaramo u sivu - 'grayscale'
#tehnika zamucenja - 'blur' provodi za radi uklanjanja smetnji (buke) i nebitnih informacija
#Gaussian blur je jedna od vrsta zamucenja, a velicina jezgre (13,5) se moze promijeniti s obzirom na sliku 
#velicina jezgre (sirina, visina) treba biti pozitivna i neparna
#parametar koji slijedi iza velicine jezgre je standardno odstupanje u smjerovima X i Y, ondosno samo u smjeru X, a Y se podrazumijeva da je 0 ako nije naveden
#s obzirom da su oba odstupanja jednaka 0, izracunavaju se iz velicine jezgre
#povecavanjem velicine jezgre smanjuje se buka, ali i gubi vise podataka
gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(13,5),0)

#inverzno binarno pragiranje - cv2.THRESH_BINARY_INV
#postavi se vrijednost praga tako da svaki piksel manje vrijednosti od te pretvori u maksimum (255), a svaka veca u 0
#cv2.THRESH_OTSU poziva algoritam koji odabire optimalnu vrijednost praga
ret,binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#tehnika prosirenja bijelog podrucja slike - za poboljsanje bijele konture slike
#traze se 'zrna' pravokutnog oblika funkcijom cv2.getStructuringElement(), kojoj se proslijede oblik i veličina jezgre, a dobije se željena jezgra
#u funkciji cv2.morphologyEx() poziva se izvrsavanje tehnike dilatacije
#ako je barem 1 piksel ispod jezgre '1', tada je element piksela '1'.
#tom tehnikom se povećava bijela regija na slici ili se povećava veličina predmeta u prvom planu
#korisno je za spajanje slomljenih dijelova predmeta na slici
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

#prikaz primijenjenih tehnika na pocetnu sliku koristeci biblioteku matplotlib
fig = plt.figure(figsize=(12,7))
plt.rcParams.update({"font.size":18})
grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig) #prikaz tih 5 slika u 3 reda i 2 stupca
plot_image = [plate_image, gray, blur, binary,thre_mor]
plot_name = ["plate_image","gray","blur","binary","dilation"]

for i in range(len(plot_image)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.title(plot_name[i])
    if i ==0:
        plt.imshow(plot_image[i])
    else:
        plt.imshow(plot_image[i],cmap="gray")

# plt.savefig("threshding.png", dpi=300)

#dohvaca i sortira utemeljene konture u kojima su 'uhvaceni' znakovi tablice slijeva nadesno(reverse=False, i=0) jer ih je bitno rasporediti u ispravnom redoslijedu
#prvi argument je lista kontura koje zelimo sortirati, a drugi metoda, tj. redoslijed sortiranja

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #izracun granicnih okvira svake konture, sto su početne(x, y)-koordinate granicnog okvira pracene sirinom i visinom
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), #granicni okviri omogucuju sortiranje stvarnih okvira uz dodani python kod koji zajedno sortira dva popisa (okvire, konture)
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts #lista sortiranih kontura

#funkcija za identifikaciju koordinata znaka
#kontura je krivulja koja spaja sve kontinuirane tocke koje dijele istu boju i intezitet
cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #koristeci cv2.CHAIN_APPROX_SIMPLE algoritam stedi se memorija,
#umjesto da se spremaju sve tocke neke linije konture, spremaju se samo po dvije krajnje tocke linije


#izraduje se kopija pocetne ulazne slike, kako bi se na njoj mogli iscrtati svi granicni okviri oko znakova
test_roi = plate_image.copy()

#incijalizacija liste koja ce se koristiti za dodavanje slike znakova
crop_characters = []

# definirane standardna sirina i visina znaka na tablici (obicno je visina vece vrijednosti zbog standardnog oblika slova)
digit_w, digit_h = 55, 75

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w #omjer visine i sirine
    if 1<=ratio<=8: #gledati samo one konture kojima je visina od 1 do 8 puta sirina(filter 1), ovisno o obliku slova i omjeru konkretno njegove visine i sirine
        if 0.85>=h/plate_image.shape[0]>=0.5: #gledati samo one konture cija je visina veca od 50% visine tablice, a manja od 85% visine tablice(filter 2)

            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 3) #nacrtati okvir (zelene boje, debljine 3) oko znamenke ili slova
            #odvajanje znakova i predvidanje
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #inverzno binarno pragiranje
            crop_characters.append(curr_num) #dodavanje svih segmentiranih znakova u listu

print("Detect {} letters...".format(len(crop_characters))) #ispis prepoznatih znakova u terminal

#prikaz tablice s okvirima oko svakog znaka
fig = plt.figure(figsize=(10,6))
plt.axis(False)
plt.imshow(test_roi)

#prikaz odvojenih znakova(u 'binary' obliku) koristeci matplotlib 
plt.style.use('classic')
fig = plt.figure(figsize=(10,6))
grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")

#ucitavaju se koristena NN arhitektura modela 'MobileNets' i vec dobivene tezine nakon zavrsene faze treniranja uz klase znakova
#zato se unutar foldera projekta nalaze MobileNets_character_recognition JSON file, License_character_recognition_weight H5 datoteka i license_character_classes NPY file
#NN koristi set podataka slova i brojeva, koji je podijeljen na set treninga(90%) i set za validaciju(10%), 
#sto omogucuje izbjegavanje overfittanja i pracenje tocnosti modela
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

#ulazni sloj modela NN je konfiguriran za primanje slika u obliku(80,80,3) pa se nasa slika i pretvara u takav oblik (168. i 169.linija)
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))]) #implementacija ucitanih klasa znakova(labels) u inverzne jednokratne oznake kodiranja dobivene od modela do digitalnih znakova
    return prediction
#za prikaz odvojenih znakova uz predikciju svakog znaka iznad konture(u 'binary' obliku) koristeci matplotlib 
fig = plt.figure(figsize=(15,3))
cols = len(crop_characters)
grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

#petlja se generira nad svakom slikom znaka iz crop_characters
#isrtava se slika svakog od znakova tablice s pripadajucim predvidanjima
final_string = ''
for i,character in enumerate(crop_characters):
    fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character,model,labels)) #poziv funkcije za predvidanje
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    final_string+=title.strip("'[]")
    plt.axis(False)
    plt.imshow(character,cmap='gray')

print("\n\n\n",final_string) #ispis konacnog predvidenog niza znakova cijele tablice

plt.show()