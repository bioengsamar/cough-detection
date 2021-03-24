from pathlib import Path
from scipy.io import wavfile
import random
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer


CATEGORIES = ["data/cough", "data/not cough"]

training_data = []  
def create_training_data():
    for category in CATEGORIES:
        paths = Path(category).glob('**/*.wav')
        class_num = CATEGORIES.index(category)
        #print(class_num)
        for path in paths:
            path_in_str = str(path)
            samplerate, data = wavfile.read(path_in_str)
            training_data.append([data, class_num])
       
create_training_data()
#print(training_data[88])
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
lb = LabelBinarizer()
y = lb.fit_transform(y)
y = to_categorical(y)
print(y)
#Let's save this data, so that we don't need to keep calculating it every time
pickle_out = open("features.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("label.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
    
            
        