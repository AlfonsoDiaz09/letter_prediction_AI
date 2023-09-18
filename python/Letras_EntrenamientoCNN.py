import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
#instalar tensorflowjs
# pip install tensorflowjs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':

    K.clear_session()
    
    data_entrenamiento = 'C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/Alphabet/train/'
    data_validacion = 'C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/Alphabet/validation/'

    """ Parametros """

    epocas = int(20)#20 Numero de veces que itera sobre nuestro set de datos
    longitud, altura = int(28), int(28) #tamaño de imagenes
    pasos = None #numero de veces que se va a procesar el set de datos
    validation_steps = int(2) #saber como va el algoritmo
    batch_size = int(32) #numero de imagenes que vamos a madar a procesar en cada paso
    clases = 26
    lr = 0.0005#ajustes para acercarse a una solucion óptima
    filtrosConv1 = int(32) #numero de filtros que se van a aplicar en la primera capa de convolucion
    filtrosConv2 = int(64) #numero de filtros que se van a aplicar en la segunda capa de convolucion
    tamano_filtro1 = (2,2) #tamaño de la ventana de convolucion
    tamano_filtro2 = (3,3) #tamaño de la ventana de convolucion
    tamano_pool = (2,2) #tamaño de la ventana de maxpooling

    #Preparar las imagenes
    #Aumento de datos
    #Variables para controlar las transformaciones que se haran en el aumento de datos
    #utilizando ImageDataGenerator de keras

    entrenamiento_datagen = ImageDataGenerator( #generador de imagenes de entrenamiento para aumentar el set de datos de entrenamiento con imagenes rotadas, zoom, etc para que el modelo no se sobreajuste a las imagenes de entrenamiento y pueda generalizar mejor a imagenes nuevas que no ha visto antes
        rescale=1. / 255, #normalizar los pixeles en rango [0,1]
        rotation_range=20, #grados de rotacion de la imagen 
        zoom_range=0.2, #rango de zoom de la imagen
        horizontal_flip=True) #invierte la imagen horizontalmente

    test_datagen = ImageDataGenerator(rescale=1./ 255)

    entrenamiento_generador = entrenamiento_datagen.flow_from_directory( #generador de imagenes de validacion para aumentar el set de datos de validacion con imagenes rotadas, zoom, etc para que el modelo no se sobreajuste a las imagenes de entrenamiento y pueda generalizar mejor a imagenes nuevas que no ha visto antes
        data_entrenamiento, #ruta de las imagenes de entrenamiento
        target_size=(altura, longitud), #tamaño de las imagenes
        batch_size=batch_size, #numero de imagenes que se van a procesar en cada paso
        class_mode='categorical',  #tipo de clasificacion que se va a hacer
        color_mode='grayscale') #color de las imagenes

    validacion_generador = test_datagen.flow_from_directory(
        data_validacion, #ruta de las imagenes de validacion
        target_size=(altura, longitud), #tamaño de las imagenes
        batch_size=batch_size, #numero de imagenes que se van a procesar en cada paso
        class_mode='categorical', #tipo de clasificacion que se va a hacer
        color_mode='grayscale') #color de las imagenes

    cnn = Sequential() #modelo secuencial de keras para crear la red neuronal convolucional
    cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 1), activation='relu')) #capa de entrada con 32 filtros de 3x3 y activacion relu para la funcion de activacion de la neurona para que no sea lineal sino que sea no lineal y pueda aprender mas cosas de la imagen y no solo lineas rectas y pixeles de la imagen


    cnn.add(Convolution2D(filtrosConv1, tamano_filtro2, padding ="same", activation='relu')) #capa de convolucion con 32 filtros de 3x3 y activacion relu para la funcion de activacion de la neurona
    cnn.add(MaxPooling2D(pool_size=tamano_pool)) #capa de maxpooling para reducir la cantidad de datos y que el algoritmo no se demore tanto en entrenar y que no se sobreajuste a los datos de entrenamiento y pueda generalizar mejor a los datos de validacion

    cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu')) #capa de convolucion con 64 filtros de 3x3 y activacion relu para la funcion de activacion de la neurona
    cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu')) 
    cnn.add(MaxPooling2D(pool_size=tamano_pool)) #capa de maxpooling para reducir la cantidad de datos

    cnn.add(Flatten()) #capa de aplanamiento para convertir la imagen en un vector de 1 dimension para poder ingresarla a la red neuronal convolucional y que pueda aprender de los datos de la imagen y no solo de los pixeles de la imagen y que pueda aprender de las lineas y de los bordes de la imagen
    cnn.add(Dense(256, activation='relu'))  #capa densa con 256 neuronas y activacion relu para la funcion de activacion de la neurona
    cnn.add(Dropout(0.5)) #capa de dropout con valor 
    cnn.add(Dense(clases, activation='softmax')) #capa de salida regular con softmax para clasificar las entradas de manera apropiada, esta funcion nos da la probabilidad de que una imagen pertenezca a una clase en particular en un valor entre 0 y 1

    cnn.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy']) #compilar el modelo con la funcion de perdida categorical_crossentropy para clasificacion de mas de dos clases, optimizador adam para ajustar los pesos de la red neuronal y metrica accuracy para saber que tan bien esta aprendiendo el modelo

    checkpoint = ModelCheckpoint("C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/Pesos/modelo_checkpoint.h5", monitor='loss', verbose=1,
    save_weights_only=False,save_best_only=False, mode='auto', save_freq=184) #guardar el modelo cada 184 pasos para no perder el modelo en caso de que se corte la ejecucion del programa        

    checkpoint2 = ModelCheckpoint("C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/Pesos/pesos_checkpoint.h5", monitor='loss', verbose=1,
    save_weights_only=True,save_best_only=False, mode='auto', save_freq=184) #guardar los pesos cada 184 pasos para no perder los pesos en caso de que se corte la ejecucion del programa

    print("Entrenando modelo...");

    historial = cnn.fit( #entrenar el modelo con el generador de imagenes de entrenamiento y validacion
        entrenamiento_generador, #generador de imagenes de entrenamiento para aumentar el set de datos de entrenamiento con imagenes rotadas, zoom, etc para que el modelo no se sobreajuste a las imagenes de entrenamiento y pueda generalizar mejor a imagenes nuevas que no ha visto antes
        steps_per_epoch=pasos, #numero de pasos por epoca para entrenar el modelo con todas las imagenes de entrenamiento y que no se quede corto de imagenes para entrenar
        epochs=epocas, #numero de epocas para entrenar el modelo con todas las imagenes de entrenamiento 
        validation_data=validacion_generador, #generador de imagenes de validacion para validar el modelo con imagenes que no ha visto antes
        validation_steps=validation_steps, #numero de pasos por epoca para validar el modelo con todas las imagenes de validacion y que no se quede corto de imagenes para validar
        callbacks=[checkpoint,checkpoint2] #callbacks para guardar el modelo y los pesos cada 184 pasos
    )

    print("Modelo entrenado!");

    target_dir = 'C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/Pesos/'

    if not os.path.exists(target_dir):
        print("se creo el directorio")
        os.mkdir(target_dir)
    cnn.save(target_dir+'/modelo1.h5')
    cnn.save_weights(target_dir+'/pesos1.h5')
    print("Guardadado")

    #Exportar el modelo con tensorflowjs
    export_dir = 'C:/Users/alfon/Desktop/Proyecto_Sistemas_Inteligentes/python/carpeta_salida/'

    if not os.path.exists(export_dir):
        print("se creo el directorio")
        os.mkdir(export_dir)

    '''
    En consola ejecutar el siguiente comando:
    tensorflowjs_converter --input_format keras modelo1.h5 carpeta_salida
    '''
    

    acc = historial.history['accuracy'] #obtener la precision del modelo
    val_acc = historial.history['val_accuracy'] #obtener la precision del modelo

    loss = historial.history['loss'] #obtener la perdida del modelo
    val_loss = historial.history['val_loss'] #obtener la perdida del modelo

    rango_epocas = range(20) #numero de epocas

    plt.figure(figsize=(10,10)) #tamaño de la grafica
    plt.subplot(1,2,1) #grafica de 1 fila, 2 columnas, posicion 1
    plt.plot(rango_epocas, acc, label='Precisión Entrenamiento') #graficar la precision del modelo
    plt.plot(rango_epocas, val_acc, label='Precisión Pruebas') #graficar la precision del modelo
    plt.legend(loc='lower right') #ubicacion de la leyenda
    plt.title('Precisión de entrenamiento y pruebas') #titulo de la grafica

    plt.subplot(1,2,2) #grafica de 1 fila, 2 columnas, posicion 2
    plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento') #graficar la perdida del modelo
    plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas') #graficar la perdida del modelo
    plt.legend(loc='upper right') #ubicacion de la leyenda
    plt.title('Pérdida de entrenamiento y pruebas') #titulo de la grafica
    plt.show() #mostrar la grafica


