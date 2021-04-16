import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier 




def load_data(filename1, filename2):
    pickle_in_x = open(filename1,"rb") 
    pickle_in_y = open(filename2,"rb")
    x = pickle.load(pickle_in_x)

    y = pickle.load(pickle_in_y)

    x= np.array(x)
    y= np.array(y)
    x = np.fft.fft(x)
    x = np.absolute(x)
    scaler = preprocessing.StandardScaler().fit(x) #standard scaling
    x = scaler.transform(x)
    #x = x - x.mean() #calculate mean to center the data.
    #x = (x - x.min()) / (x.max() - x.min()) #Standard normalization
    #x = (x * 2) -1 #scaling between 1, -1
    return x, y

def model(X_train, X_test, y_train, y_test):
    #model = svm.SVC(kernel='linear') # SVM for multi-class classification using built-in one-vs-one method
    #model.fit(X_train,y_train)
    
    #knn = KNeighborsClassifier(n_neighbors=5, weights='distance') 
    #knn.fit(X_train,y_train)
    #model = RandomForestClassifier(n_estimators=100 ,random_state=50 ,criterion="gini",max_depth=10)
    #model.fit(X_train,y_train)
    
    #mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=1000,  random_state=1) 
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
    mlp.fit(X_train,y_train)
    accuracy= mlp.score(X_test ,y_test)
    return accuracy


    

if __name__ == "__main__":
    filename1="features.pickle"
    filename2= "label.pickle"


    
    x, y=load_data(filename1, filename2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)
    accuracy= model(X_train, X_test, y_train, y_test)
    print('Model accuracy is: ', accuracy) #0.8779342723004695 without and with center data & without normalization, scaling
                                           #0.9295774647887324 with center data and normalization & without scaling
                                           #0.9389671361502347 with center data, normalization & scaling