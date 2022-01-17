# ObjectRecognition
Object Detection and Recognition in ROS

Projecto the Trabajo de Fin de Master - Master Ingenieria Industrial UC3M 2017/2018.

Titulo del TFM: Multi-Object Recognition based on Deep Learning applied to Mobile Robots
Autor: Jesus Saez Alegre



Parte 1: 
Creación de CCN

En la carpeta “Created_CNN”, hay dos scripts principales:
•	train.py
•	predict.py // Predict_3classes.py
Train.py sirve para crear la red y entrenarla en función de los elementos incluidos en la carpeta “training_data”. Este archivo genera tres archivos que servirán para cargar la red y los pesos en el siguiente script, además de un archivo Excel donde se puede ver el progreso del entrenamiento.
Para entrenar con más clases basta con añadir una carpeta nueva con el título de la clase e imágenes en su interior. Para entrenar con más imágenes solo hay que añadir más imágenes en su respectiva carpeta.
En este script se pueden cambiar el tipo de filtros y el número de los mismos, el learning rate, el número de iteraciones para el entrenamiento y el número de capas, siendo el output de una capa el input de la siguiente, y la última flatten layer ha de tener tantas salidas como clases entrenadas.
Pedict.py sirve para predecir imágenes de la carpeta “testing_data” y crear un Excel con los resultados en forma de matriz de confusión (este scrip está preparado para ordenar los resultados de dos clases y el otro de 3). La carpeta “testing_data” debe contener en su interior la misma estructura de carpetas que “training_data” pero sus imágenes deben ser distintas para dar validez a la prueba.
 Es importante que los meta-datos que se cargan en esta red correspondan a los archivos generados por entrenamiento que queremos usar. El checkpoint que genera “train.py” también debe estar en la carpeta junto a los demás archivos, y debe ser correspondiente al entrenamiento que se quiere usar.
Si se quieren usar los archivos en otro lugar es necesario cambiar los directorios a los que apuntan los distintos scrips, siendo estos los de las carpetas “training_data” y “testing_data”, así como el lugar donde se van a generar los datos del entrenamiento, y de donde se deben cargar éstos datos en el módulo predict.py
Por último, se crean scrips preparados para funcionar con ROS. Las funciones a las que llama este último script es “3classes.py”. Este caso esta implementado para ambos métodos de multi detección y para detectar 3 clases. En estos scrips es posible modificar el valor de threshold.
Además, se ha incluido una carpeta “google-images-download-master”, que contiene un script para descargar de forma automática imágenes de google. Cómo usarlo se puede ver en el siguiente enlace:
•	https://github.com/hardikvasa/google-images-download


 



Parte 2: 
Descarga de modelos y multi-detección


Los archivos se encuentran en la carpeta “Developed algorithms”, y hacen referencia a los tres sistemas empleados, InceptionV3, YOLOV2, y SSD. Dentro de la carpeta se encuentran los scrips relativos a InceptionV3, y dos carpetas que contienen los archivos de los otros dos sistemas.


1.	InceptionV3
Repositorio descargado desde: https://github.com/tensorflow/models
Video tutorial de ayuda: https://www.youtube.com/watch?v=COlbP62-B-U&vl=en
•	Archivo interés del repositorio: models/tutorials/image/imagenet/classify_object.py (no está el repositorio entero en la carpeta)
Ese es el documento de interés, que he modificado un poco y he dejado en la carpeta, “Developed algorithms”, y del cual leen los archivos ROS_ColorFiltering.py y ROS_Segmentation_Inception.py
Para que los sistemas basados en el módulo de reconocimiento de imagen Inception funcionen, en la carpeta temporal del ordenador ha de encontrarse los archivos relativos a la configuración de la red, los cuales  deben estar dentro de una carpeta llamada “imagenet”.
Los scripts “segmentation” y “ColorFiltering” contienen las funciones de interés del programa, de los cuales los archivos preparados para funcionar en ROS se sirven. Estos archivos permiten cambiar el threshold de las predicciones fácilmente, así como el número de predicciones que se quiere por imagen.
En ambos casos la función clasificación sirve para analizar imágenes y obtener un resultado que incluye nombre de la clase, probabilidad, y localización.
La función “charge” sirve para segmentar la imagen en 8 cuadrantes.
La función “contornos” sirve para detectar 8 contornos en la imagen.
En todos los scripts se incluyen comentarios de cómo funciona el código.



2.	YOLO
Librería darkflow descargada desde: https://github.com/thtrieu/darkflow
Video tutorial ayuda e instalación:
 https://www.youtube.com/watch?v=PyjBd7IDYZs&t=387s
Una vez instalado el paquete, es necesario cargar pesos y configuración de red del mismo tipo a la hora de crear la red. La configuración de la red debe encontrarse dentro de la carpeta “cfg”, y los pesos dentro de la carpeta “bin”. Y los paths deben apuntar a estas carpetas.
El script  “ROS_YOLO.py” incluye el modelo de reconocimiento de imágenes, su función de clasificación se encuentra en “YOLO_func.py”, y es posible cambiar el valor de Threshold.



3.	SSD
El repositorio descargado para hacer funcionar Inception, sirve también para SSD.
•	La ruta al archivo de interés:
/models/research/object_detection/object_detection_tutorial.py
Basado en ese archivo se desarrollan los demás scripts. Dentro del código hay que seleccionar el modelo que se quiere usar y las labels correspondientes, cambiando el path a la carpeta que los contienen.
Este path NO se ha cambiado, pero debe apuntar a la carpeta “ssd_mobilenet_v1_coco_2017_11_17”
Se puede ver el funcionamiento del sistema con el archivo ROS_SSD.py, que hace uso de las funciones de SSD_func.py. También se puede variar el threshold value de las predicciones
