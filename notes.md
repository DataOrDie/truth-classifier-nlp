La competición se desarrolla a través de Kaggle, como se hizo en la actividad anterior. En enlace a la competición de kaggle es éste.

La estructura del ejercicio es la siguiente:

    Los alumnos se organizarán en los mismos equipos que en la actividad anterior. Los equipos no deben ser más pequeños.
    Cada equipo debe tener un capitán, que será el que se comunique con el profesorado, y el que realice la entrega del ejercicio.
    El capitán de cada equipo debe enviar un correo comunicando los miembros su equipo.
    La competición comienza el 10 de Abril. A partir de este momento se podrá realizar envíos a la plataforma Kaggle.
    El día 19 de Mayo a las 23:50 se cierra la competición en Kaggle.
    El día 20 de Mayo a las 23:59 se cierra la entrega de moodle, con el contenido descrito abajo.
    El día 21 de Mayo se realiza la presentación en clase.

Respecto a la presentación y la documentación a entregar:

    Se debe entregar una presentación en formato PDF, en la que se debe explicar lo que se hecho en la competición, con especial énfasis en las técnicas empleadas para mejorar los modelos. No se debe hacer una introducción al problema ni hablar de generalidades: la exposición debe tener un interés alto. Como entrega, se debe subir a Moodle:
        Notebook(s) o código con el que se ha generado la submission final
        PDF con transparencias de la presentación
    El día de la presentación, todos los miembros del grupo deben participar en la presentación, que en total durará no más de 10 minutos.
    En la entrega de Moodle, se debe subir el fichero de la presentación en PDF y el código empleado en un archivo ZIP.

El desarrollo del ejercicio será en Kaggle. Para poder participar, cada alumno debe tener cuenta en esta plataforma. Además, cada equipo deberá tener un nombre único que lo identifique dentro de la competición. Para conocer más detalles acerca de Kaggle, ver las FAQs. Hay que tener en cuenta que para realizar un envío a la plataforma Kaggle hay que generar un fichero CSV con las predicciones generadas, y subirlo. Los detalles del formato de este fichero se encuentran en la página del competición.
 

Durante la realización de las prácticas pueden utilizarse cualquiera de los conceptos aprendidos en el curso (p.ej. algoritmos genéticos para optimizar, ensembles, para mejorar, NER, ...). 

Aunque se puede colaborar y se fomenta hacerlo con grupos de estudio, el trabajo entregado en la parte individual debe ser personal  y original, respetando el código de ética de la UPM. El uso de cualquier trabajo externo debe ser citado, considerándose plagio en caso contrario.

Cada grupo debe:

    Elegir un capitán/portavoz
    Distribuir tareas comunes, preprocesado, u otras tareas que decida (captura, visualización, ....). También pueden construirse diferentes clasificadores por diferentes miembros del grupo que proporcionen características nuevas (p.ej. uso del lenguaje, orientación política, una emoción o sentimiento, ...)
    Discutir y analizar los resultados individuales
    Redactar una memoria de grupo (máximo 10 páginas) de los resultados de la práctica. Se sugiere como esquema: Introducción, Descripción del problema, Descripción Conjunto de Datos, Descripción Tareas Comunes, Evaluación, Análisis y Conclusiones conjuntas, Bibliografía. La bibliografía debe estar correctamente referenciada, se recomienda usar latex u otro gestor de la bibliografía.
    Entregar notebooks conjuntos, bien documentados. Incluir una tabla con tantas columnas como número de miembros del equipo más uno. La primera columna son los nombres de los notebooks entregados, y el resto de columnas son los miembros del equipo. Se debe poner una X en los autores de cada notebook.

Cada participante del grupo debe:

    Aplicar técnicas de procesado de lenguaje natural
    Aplicar técnicas de aprendizaje automático y construir un clasificador
    Redactar una memoria, aparte de la de grupo, máximo 10 páginas. Se sugiere el siguiente esquema: Introducción, Objetivos, Tareas Realizadas, Evaluación, Análisis y conclusiones.
    Enviar un fichero comprimido (zip) con los notebooks realizados y la memoria individual. El capitán del grupo incluirá también la memoria y notebooks comunes del grupo en dicho fichero comprimido.
    Incluir una tabla con tantas filas como miembros del grupo, él mismo incluido, y dos columnas: Valoración (calificación asignada sobre 10),  Repetir (con valor sí, si desea volver a trabajar con este miembro o no en caso contrario).

Durante la realización de las prácticas pueden utilizarse cualquiera de los conceptos aprendidos en el curso (p.ej. algoritmos genéticos para optimizar, ensembles, para mejorar, NER, ...). 

Aunque se puede colaborar y se fomenta hacerlo con grupos de estudio, el trabajo entregado en la parte individual debe ser personal  y original, respetando el código de ética de la UPM. El uso de cualquier trabajo externo debe ser citado, considerándose plagio en caso contrario.




-----------------------------------------------------------------
https://www.kaggle.com/competitions/2025-26-false-political-claim-detection



Description
This challenge consists in a Natural Language Processing (NLP) and Machine Learning (ML) competition. The propose task is to predict pieces of information that are intentionally and can be verifies as false. That is, **fake news**. You will need to use the techniques learned in class, applying them to a competition with the aim of obtaining the best model. In this challenge, nothing is banned, you can try any technique you want, including **Deep Learning** models. To this end, a dataset of fake claims is provided. Your mission is to build a classifier model that determines whether a certain claim is **false** or **true**.


This dataset is related to false political claim detection, containing labeled statements (or claims) along with metadata about the speaker, subject, and other contextual information. Below is a detailed breakdown of the dataset's structure and key characteristics:

    id (unique identifier)
    label (truthfulness score). The classification task is binary, meaning that a claim can be either true or false.
    statement (text of the claim)
    subject (topic of the claim)
    speaker (person making the claim)
    speaker_job (occupation of the speaker)
    state_info (geographic context)
    party_affiliation (political party of the speaker)


Modelos que podemos usar y probar

1. Baselines simples
    - DummyClassifier
    - Regresión Logística
    - Naive Bayes

2. Modelos clásicos de ML
    - SVM lineal
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost

3. Modelos con texto y vectores
    - Regresión Logística con TF-IDF
    - SVM con TF-IDF
    - Naive Bayes con TF-IDF
    - Combinar texto + variables tabulares

4. Modelos de deep learning
    - MLP sencillo sobre embeddings
    - CNN para texto
    - LSTM
    - BiLSTM
    - GRU

5. Transformers
    - BERT
    - RoBERTa
    - DistilBERT
    - DeBERTa
    - XLM-RoBERTa

6. Enfoques extra para comparar
    - Ensemble de varios modelos
    - Voting classifier
    - Stacking
    - Ajuste de umbral de decisión


