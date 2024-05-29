# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:35:49 2023

@author: pablo
"""


#Importamos librerías
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.datasets import make_classification,make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

#nombres de los clasificadores
names = [
    "Nearest Neighbors",

    "RBF SVM",

    "Perceptron",
    
    "MLP"
]
#Algoritmos de los clasificadores
classifiers = [
    KNeighborsClassifier(5),#5 vecinos

    SVC(kernel="rbf",gamma=0.1, C=10, random_state=42),
    #hiperparámetros C=10 y gamma = 0.1 para una SVC con kernel RBF

    
    SGDClassifier(loss="perceptron", max_iter=10000, learning_rate="constant" ,eta0=0.3),
#tasa de aprendizaje de 0.3 para el Perceptron con 10 épocas y función de activación sigmoide.
#Documentacion de este clasificador
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
#Documentación clasificador sklearn.linear_model.Perceptron    
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    MLPClassifier(max_iter=10000,learning_rate='constant',learning_rate_init=0.3,
                  activation="logistic",hidden_layer_sizes=(2))
    #2 neuronas en capa de entrada, 2 en capa oculta y 1 en la capa de salida
    #En el parámetro hidden_layer_sizes ponemos 2 para tener 2 ocultas
    #Documentacion
    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
]

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
#Cargamos los 3 primeros datasets con pandas
#ruta_abs='D:/AA_Trabajos_Escuela/7mo sem/Machine learning/Practica 5/'
dataset1=pd.read_csv('dataset_classifiers1.csv',index_col=0)
dataset2=pd.read_csv('dataset_classifiers2.csv',index_col=0)
dataset3=pd.read_csv('dataset_classifiers3.csv')
#Creamos el dataset XOR
#Generamos los puntos de la clase 1
X_0, y0 = make_blobs(n_samples=[1000,1000], centers = [(0, 0),  (1, 1)],cluster_std=0.1)
x_data0=X_0[:,0]
y_data0=X_0[:,1]
y0=np.full(len(y0),0)

#Generamos los puntos de la clase 0
X_1, y1 = make_blobs(n_samples=[1000,1000], centers = [(0, 1),  (1, 0)],cluster_std=0.1)
x_data1=X_1[:,0]
y_data1=X_1[:,1]
y1=np.full(len(y1),1)
x_data=list(x_data0)+list(x_data1)
y_data=list(y_data0)+list(y_data1)
y=list(y0)+list(y1)
datos={'0':x_data,'1':y_data,'y_true':y}
dataset4=pd.DataFrame(datos)
#Guardamos en tuplas dentro de una lista los datos y las labels de los datasets
datasets = [
  
    (dataset1[['0','1']].to_numpy(),dataset1['y_true'].to_numpy()),
    (dataset2[['0','1']].to_numpy(),dataset2['y_true'].to_numpy()),
    (dataset3[['0','1']].to_numpy(),dataset3['y_true'].to_numpy()),
    (dataset4[['0','1']].to_numpy(),dataset4['y_true'].to_numpy())
]
datasets_names=['Lineal separable', 'Círculos concéntricos', 'Lunas' , 'XOR']
accuracies=[]#Aquí guardaremos la precisión de cada clasificador
figure = plt.figure(figsize=(27, 9))#inicializamos la figura
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,shuffle=(True)
    )#En test_size ponemos 0.2 para tener 80/20

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
        ax.set_ylabel(datasets_names[0])
    if ds_cnt == 1:#con esto ponemos los títulos de los datasets
        ax.set_ylabel(datasets_names[1])
    if ds_cnt == 2:
        ax.set_ylabel(datasets_names[2])
    if ds_cnt == 3:
        ax.set_ylabel(datasets_names[3])
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)#obtenemos la precisión del clasificador
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1
        accuracies.append(score)#Guardamos los score 

plt.tight_layout()
plt.savefig("fronteras.png")#guardamos la figura
plt.show()

#generamos el dataframe que contenga las precisiones
accuracies=[round(accuracie,2) for accuracie in accuracies]
accuracie=[accuracies[:4],accuracies[4:8],accuracies[8:12],accuracies[12:]]
accuracie=np.matrix(accuracie)
precision=pd.DataFrame(accuracie,index=datasets_names,columns=names)
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
t= ax.table(cellText=precision.values,  colLabels=precision.columns, rowLabels=precision.index,  loc='center')
t.auto_set_font_size(False) 
t.set_fontsize(8)
fig.tight_layout()
plt.savefig("accuracies.png")#guardamos la tabla
plt.show()

"""
En esta práctica, resolverá 4 datasets con los clasificadores analizados en clase (k-NN, SVC, Perceptron y MLP). Los 4 datasets contienen problemas de clasificación de dos clases de puntos en 2 dimensiones.

El dataset classifiers1 consta de 500 registros y es linealmente separable.

El dataset classifiers2 consta de 350 vectores de características distribuidos en dos circulos concéntricos (anillo)

El dataset classifiers3 consta de 10000 puntos distribuidos en forma de lunas

El cuarto dataset lo deberá generar usted siguiendo las notas y códigos analizados en clase. El dataset constará de 4000 puntos, 1000 para cada vértice de un XOR.

Se recomienda el estudio previo del script "classifier comparison" de sklearn para la elaboración de su gráfica comparativa. Puede apoyarse en los scripts proporcionados en la sección de materiales o en otras implementaciones disponibles en internet.

https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

1.  Resuelva los 4 problemas de clasificación (lineal separable, círculos concéntricos, lunas y XOR)  con los 4 clasificadores estudiados en clase. Utilice un valor de k =5 para el clasificador k-NN, los hiperparámetros C=10 y gamma = 0.1 para una SVC con kernel RBF y una tasa de aprendizaje de 0.3 para el Perceptron con 10 épocas y función de activación sigmoide. Para el MLP utilice los mismos hiperparámetros que el perceptron, en una arquitectura 2-2-1 (2 neuronas en capa de entrada, 2 en capa oculta y 1 en la capa de salida)

2. Genere una figura donde muestre las fronteras de decisión de los 4 clasificadores resolviendo los 4 datasets (puede apoyarse en el script anterior para la generación de la figura). Llame a esta figura fronteras.jpeg o png

3. Utilice una partición de dataset 80/20 para validar sus clasificadores (80% de entrenamiento y 20% de prueba) y reporte en una tabla los accuracies obtenidos por cada clasificador para cada uno de los 4 datasets. Guarde esta tabla como una imagen llamada accuracies.jpeg o .png

4. Coloque su script con comentarios describiendo los bloques principales de su implementación, la figura de las fronteras de decisión y la figura de accuracies en un archivo comprimido llamado ML_Apellidos_Practica5.zip o .rar y súbalo a esta plataforma

"""