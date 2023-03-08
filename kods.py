from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

bildes = []
irNavMaja = []


adrese = './images'


for nosaukums in os.listdir(adrese):
    bilde = Image.open(os.path.join(adrese, nosaukums)).resize((200,200), Image.Resampling.NEAREST)
    bildes.append(np.array(bilde))
    if 'maja' in nosaukums:
        irNavMaja.append(1)
    else:
        irNavMaja.append(0)

bildes = np.array(bildes)
irNavMaja = np.array(irNavMaja)   

#Jāsadala dati trenina datos un testa datos
x_train, x_test, y_train, y_test = train_test_split(bildes, irNavMaja, test_size = 0.2)

#datus pārstaisa 0-1, nevis 0-255, kas ir pikseļu 
x_test = x_test/255.0
x_train = x_train/255.0

#bildes DATI jasamazina
x_test = x_test.reshape(x_test.shape[0],-1)
x_train = x_train.reshape(x_train.shape[0],-1)


#trenējam modeli
modelis = RandomForestClassifier()
modelis.fit(x_train, y_train)


#pārbaudīt testa datus 
testa_atbilde = modelis.predict(x_test)


#pārbaudīt precizitāti
precizitate = accuracy_score(y_test, testa_atbilde)
print('Precizitāte:', precizitate)
