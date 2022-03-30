
El análisis de sentimientos en redes sociales permite identificar cómo se sienten las personas sobre una temática particular en el mundo digital. A diferencia de técnicas como hacer un simple recuento de menciones o comentarios, cuando trabajamos con técnicas de Machine Learning o Deep Learning el análisis engloba emociones y opiniones. 
Este proceso implica la recopilación y análisis de información de publicaciones de usuarios en redes sociales con la finalidad de calcular la probabilidad de sentimiento positivo hacia una determinada temática.

El objetivo de este proyecto es desarrollar una aplicación web que ofrezca a los usuarios registrados una métrica exclusiva llamada Sentimiento. La propuesta de valor es ofrecer una KPI en Twitter que muestra si los tweets analizados relacionados con una temática escogida por los usuarios han sido positivos o negativos. Por ejemplo, si el usuario es un trabajador del departamento de marketing de una empresa que está ejecutando una campaña que obtuvo 1000 tweets, podrá conocer si los usuarios que participan en ella les gusta o no. Por otro lado, es también muy útil para saber si las menciones en Twitter a una marca son positivas o negativas

Se ha creado una aplicación web con Python con la funcionalidad explicada anteriormente. Los requisitos que debe cumplir esta aplicación son:
1.	Disponer de una base de datos que almacene:
    - Los usuarios que se registran para utilizar el servicio.

    - Los tweets que se han analizado en cada temática indicada por todos los usuarios.

    - Los modelos entrenados y productivizados que permiten analizar el sentimiento del texto de los tweets.

2.	 Gestionar de forma adecuada el registro y acceso a la aplicación.

3.	Una página que permita al usuario introducir una temática para buscar tweets relacionados y analizar su sentimiento.

4.	Modelos basados en redes neuronales entrenados que analicen el sentimiento de los tweets de una cierta temática.

5.	Comunicación con la API de twitter para descargar y analizar los tweets.

6.	Mostrar los resultados de las búsquedas que ha realizado el usuario para mostrarlas en la página del servicio.

Enlace a la imagen docker publicada en Docker Hub:
- https://hub.docker.com/repository/docker/sergiomasavi/fayspy-apps-ml-tf
