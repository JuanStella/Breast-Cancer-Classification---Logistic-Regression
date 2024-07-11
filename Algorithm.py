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
    

    def predict (self, x1, x2, bnorm):
        
        if (x1+x2) < bnorm:
            return "El tumor es benigno"
        else:
            return "El tumor es maligno"
    

    def fit (self, data):
        
        self.x = np.zeros((len(data), 2))
        self.y = np.zeros(len(data))

        '''for i in range(1,3):
            self.x[:,i-1] = data[:,i]'''
        
        self.x[:,0] = data[:,1]
        self.x[:,1] = data[:,2]


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
        X_features = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
                      'radius_se,texture_se,perimeter_se,area_se','smoothness_se,compactness_se','concavity_se,concave points_se','symmetry_se',
                      'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst'
                      ,'symmetry_worst','fractal_dimension_worst']

        fig,ax=plt.subplots(1,2,figsize=(12,3),sharey=True)


        aprox = self.sigmoid(z)

        for i in range(len(ax)):
            ax[i].scatter(self.x[:,i],self.y, label = 'target')
            ax[i].set_xlabel(X_features[i])
            ax[i].scatter(self.x[:,i],z,color="red", label = 'predict')
        ax[0].set_ylabel("Price"); ax[0].legend();
        fig.suptitle("target versus prediction using z-score normalized model")
        plt.show()

        # Graficar x1 (radius_mean) y x2 (texture_mean)
        x1 = self.x[:, 0]
        x2 = self.x[:, 1]

        xplot1 = np.linspace(x1.min(), x1.max(), 100)
        xplot2 = np.linspace(x2.min(), x2.max(), 100)


        xplot1 = b_norm - xplot2
        # Graficar puntos reales
        plt.figure(figsize=(10, 6))
        plt.scatter(x1[self.y == 1], x2[self.y == 1], color='red', alpha=0.7, label='Maligno (Real)')
        plt.scatter(x1[self.y == 0], x2[self.y == 0], color='blue', alpha=0.7, label='Benigno (Real)')
        plt.plot(xplot1, xplot2, color='green', label='Boundary')
        plt.scatter(2, 2, color='black', label='Unknown Tumor')
        plt.xlabel('radius_mean')
        plt.ylabel('texture_mean')
        plt.title('Visualización de datos de cáncer de mama')
        plt.legend()
        plt.show()

        print(self.predict (2,2,b_norm))

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