# Resources

En la carpeta *weights* se pueden encontrar los pesos para el modelo entrenado bajo diferentes configuraciones. Los archivos "categorical_crossentropy",
"jaccard" y "focal_loss" poseen los pesos para el modelo sin usar dropout pero con las funciones de costo del nombre. Luego, los
archivos "jaccard_dropout_xx" poseen los pesos del modelo entrenado usando jaccard distance como función de costo y diferentes
dropouts. 
