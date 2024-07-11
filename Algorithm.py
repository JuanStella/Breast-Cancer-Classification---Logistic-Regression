import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler 



class Algorithm:

    def __init__(self):
        self.x = None
        self.y = None
        self.data = None
        

    def load_data(self, file):
        try:
            # Cargar datos numéricos
            numeric_data = np.loadtxt(file, delimiter=',', skiprows=1, usecols=range(2, 32))
            
            # Cargar diagnósticos separadamente
            diagnoses = np.loadtxt(file, delimiter=',', skiprows=1, usecols=1, dtype=str)
            
            # Combinar datos numéricos y diagnósticos
            self.data = np.column_stack((diagnoses, numeric_data))
            return self.data
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None
    

    def sigmoid(self,z):

        g = 1/(1+np.e**(-z))
        return g
    

    def predict (self, g, x = None):
        x = x*999/13
        x = int(x)
        probability = g[x]
    
        if probability >= 0.5:
            diagnosis = "maligno"
            certainty = probability * 100
        else:
            diagnosis = "benigno"
            certainty = (1 - probability) * 100
        return (f"Hay un {certainty:.2f}% de probabilidad de que el tumor sea {diagnosis}.")
        
    

    def fit (self, data):
        
        self.x = np.zeros((len(data), 2))
        self.y = np.zeros(len(data))

        self.x[:,0] = data[:, 1]
        self.x[:,1] = data[:, 2]

        for i in range(len(data)):
            if data[i,0] == "M":
                self.y[i] = 1
            elif data[i,0] == "B":
                self.y[i] = 0


        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        sgdr = SGDRegressor(max_iter=1000)
        sgdr.fit(self.x, self.y)

        b_norm = sgdr.intercept_
        w_norm = sgdr.coef_

        z = np.dot(self.x, w_norm) + b_norm
        X_features = ["Size", "Texture"]

        fig,ax=plt.subplots(1,2,figsize=(12,3),sharey=True)


        aprox = self.sigmoid(z)

        for i in range(len(ax)):
            ax[i].scatter(self.x[:,i],self.y, label = 'target')
            ax[i].set_xlabel(X_features[i])
            ax[i].scatter(self.x[:,i],z,color="red", label = 'predict')
        ax[0].set_ylabel("Price"); ax[0].legend();
        fig.suptitle("target versus prediction using z-score normalized model")
        plt.show()

        # Graficar la función sigmoidea
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        for i in range(2):
            max = (int(self.x[:,i].max()+4))
            min = (int(self.x[:,i].min()-4))
            print(max, min)
            #print(max, min)
            x_range = np.linspace(min, max, 1000)
            z_range = w_norm[i] * x_range + b_norm
            sigmoid = self.sigmoid(z_range)
        
            print(self.predict(sigmoid, 2))

            ax[i].scatter(self.x[:,i], self.y, label='Target', alpha=0.5)
            ax[i].plot(x_range, sigmoid, color='red', label='Función Sigmoidea')
        
            ax[i].set_xlabel(X_features[i])
            ax[i].set_ylabel("Probabilidad de Diagnóstico Maligno")
            ax[i].legend()
            ax[i].set_title(f"Función Sigmoidea: {X_features[i]} vs Probabilidad")

            ax[i].set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.show()

        return aprox
    

def main():

    alg = Algorithm()
    data = alg.load_data('D:\\ML\MyProgress\\Breast Cancer Classification - Logistic Regression\\breast-cancer.csv')
    if data is not None:
        alg.fit(data)
    else:
        print("No se pudieron cargar los datos.")
    return None

    


if __name__ == '__main__':
    main()