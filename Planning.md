Plan de trabajo con fechas

Supuesto de arranque: mañana es la primera reunión del equipo.
Objetivo organizativo: 6 personas, trabajo independiente en ML, tuning y testing, con modelos distintos por persona para comparar resultados.

24/04 - Primera reunión
- Revisar el enunciado, el dataset y las reglas de entrega.
- Confirmar el reparto de trabajo entre 6 personas.
- Acordar que cada persona pruebe al menos un modelo distinto.
- Definir una plantilla común para registrar experimentos, métricas y conclusiones.

25/04 - Revisión de datos y preprocesado inicial
- Cada persona revisa las columnas, tipos de variables y valores faltantes.
- Cada persona prueba el preprocesado básico de las columnas disponibles.
- Unificar criterios mínimos: limpieza de texto, tratamiento de nulos y codificación de categorías.
- Dejar anotados problemas detectados: ruido, duplicados, etiquetas ambiguas y desbalance.

26/04 - Línea base compartida
- Construir una baseline común para tener una referencia inicial.
- Validar con el mismo esquema de evaluación para todos.
- Registrar métricas y errores frecuentes.

27/04 a 30/04 - Modelos independientes 1
- Persona 1: probar un modelo lineal sobre texto.
- Persona 2: probar un modelo basado en árboles con variables tabulares.
- Persona 3: probar un enfoque con TF-IDF y metadata combinada.
- Persona 4: probar otro clasificador distinto y comparar contra la baseline.
- Persona 5: probar una variante con ingeniería de variables.
- Persona 6: probar un modelo alternativo y documentar resultados.
- Cada persona debe guardar resultados, parámetros y observaciones.

01/05 a 04/05 - Tuning y mejora
- Afinar hiperparámetros de los mejores candidatos.
- Probar variantes de preprocesado y selección de features.
- Comparar qué mejoras son reales y cuáles no superan la validación.
- Mantener un registro de experimentos descartados.

05/05 a 08/05 - Modelos independientes 2
- Cada persona prueba una segunda familia de modelos o una mejora clara del primer intento.
- Comparar si el cambio de representación o de features aporta valor.
- Revisar errores por clase y casos difíciles.

09/05 a 12/05 - Análisis cruzado
- Poner en común los resultados de los 6 modelos.
- Elegir los 2 o 3 enfoques más sólidos.
- Decidir si conviene combinar modelos, ajustar umbrales o simplificar el pipeline.

13/05 a 15/05 - Pipeline final
- Reentrenar el mejor modelo o ensamblado con la configuración final.
- Comprobar reproducibilidad de extremo a extremo.
- Generar un CSV de prueba y verificar que respeta el formato de Kaggle.

16/05 a 19/05 - Cierre y entrega
- Hacer las últimas submissions en Kaggle.
- Congelar el código final y preparar el ZIP de entrega.
- Redactar la memoria de grupo y revisar la presentación.
- Confirmar que todos los miembros han participado en el análisis y en la presentación.