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

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

#ucitavanje slike tablice nad kojom ce se izvrsiti segmentacija
LpImg = preprocess_image("PlateExamples/15.png")
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
#LpImg = preprocess_image("PlateExamples/12.jpg")
#LpImg = preprocess_image("PlateExamples/13.jpg")
#LpImg = preprocess_image("PlateExamples/15.png")


fig = plt.figure(figsize=(12,6))
plt.axis(False)
plt.imshow(LpImg)

#skaliranje, izracun apsolutnih vrijednosti i pretvorba slike u 8-bitnu 'skalu' 
#3 bita za crvenu, 3 bita za zelenu i 2 bita za plavu boju (rgb)
plate_image = cv2.convertScaleAbs(LpImg, alpha=(255.0))

#boja nije nuzna za prepoznavanje tablice pa je uklanjamo sa slike i pretvaramo u sivu - 'grayscale'
#tehnika zamucenja - 'blur' provodi za radi uklanjanja smetnji (buke) i nebitnih informacija
#Gaussian blur je jedna od vrsta zamucenja, a velicina jezgre (7,7) se moze promijeniti s obzirom na sliku 
#povecavanjem velicine jezgre smanjuje se buka, ali i gubi vise podataka
gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(13,5),0)
    
#postavi se vrijednost praga tako da se svaka manja vrijednost piksela od te pretvori u 255 i obratno
#inverzno binarno pragiranje
ret,binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#tehnika prosirenja bijelog podrucja slike - za poboljsanje bijele konture slike
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

# vizualizacija primijenjenih tehnika na pocetnu sliku    
fig = plt.figure(figsize=(12,7))
plt.rcParams.update({"font.size":18})
grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
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

#dohvaca i sortira utemeljene konture slijeva udesno jer ih je bitno rasporediti u ispravnom redoslijedu
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

#funkcija za identifikaciju koordinata znaka tablice
#teorija: kontura je krivulja koja spaja sve kontinuirane tocke koje dijele istu boju i intezitet
cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

#incijalizacija liste koja ce se koristiti za dodavanje slike znakova
crop_characters = []

# definirane standardna sirina i visina znaka na tablici (obicno je visina vece vrijendosti)
digit_w, digit_h = 55, 75

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w #omjer visine i sirine
    if 1<=ratio<=8: #gledati samo one konture kojima je visina od 1 do 8 puta sirina (filter 1)
        if 0.85>=h/plate_image.shape[0]>=0.5: #gledati samo one konture cija je visina veca od 50% visine tablice (filter 2)
            #nacrtati okvir oko broja znamenke ili slova

            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
            #odvajanje znakova i predvidanje
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #inverzno binarno pragiranje
            crop_characters.append(curr_num) #dodavanje svih segmentiranih znakova u listu

print("Detect {} letters...".format(len(crop_characters))) 

#vizualizacija koristeci matplotlib 
fig = plt.figure(figsize=(10,6))
plt.axis(False)
plt.imshow(test_roi)
#plt.savefig('grab_digit_contour.png',dpi=300)

plt.style.use('classic')
fig = plt.figure(figsize=(10,6))
grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")
#plt.savefig("segmented_leter.png",dpi=300)

#ucitavaju se koristena NN arhitektura modela i tezine nakon zavrsene faze treniranja te klase oznaka
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

fig = plt.figure(figsize=(15,3))
cols = len(crop_characters)
grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)

final_string = ''
for i,character in enumerate(crop_characters):
    fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character,model,labels))
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    final_string+=title.strip("'[]")
    plt.axis(False)
    plt.imshow(character,cmap='gray')

print("\n\n\n",final_string)
#plt.savefig('final_result.png', dpi=300)

plt.show()