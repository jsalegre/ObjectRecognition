import cv2
from darkflow.net.build import TFNet
import numpy as np





def classify(net, image):

	tfnet=net
	frame=image

	#lista de colores para las cajas de los objetos
	#255 * tres columnas aleatorias, durante 5 filas
	colors=[tuple(255 * np.random.rand(3)) for _ in range(10)]

	results= tfnet.return_predict(frame)
	texto=[None]*length(results)
	i=0

	for color, result in zip(colors, results):
		tl=(result['topleft']['x'],result['topleft']['y'])
		br=(result['bottomright']['x'],result['bottomright']['y'])
		label=result['label']
		confidence=round(result['confidence'],3)
		texto[i]= (label+"("+tl+","+br+")" + ":" + str(confidence))
		i=i+1

	return texto
		


		