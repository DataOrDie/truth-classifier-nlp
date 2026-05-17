# Informe del Proyecto: Truth Classifier NLP

**Kaggle — Competencia del Curso UPM**
**Tarea:** Clasificación binaria de afirmaciones políticas como verdaderas (0) o falsas (1).
**Dataset:** 8,950 muestras de entrenamiento y 3,860 de prueba sin etiqueta.
**Métrica objetivo:** Macro F1 Score.

---

## 1. Planificación

El proyecto arrancó con una reunión de equipo de seis personas el 24 de abril. La idea desde el principio era que cada quien trabajara de forma independiente, probando modelos distintos, para luego comparar resultados y tomar decisiones colectivas. No era un proyecto donde alguien hacía todo y los demás esperaban — era un experimento distribuido.

La primera semana se dedicó a entender el dataset y ponerse de acuerdo en los criterios mínimos: cómo limpiar el texto, qué hacer con los valores faltantes, cómo codificar las variables categóricas y cómo registrar los experimentos de forma que fueran comparables. Todo eso quedó anotado antes de que alguien entrenara un solo modelo.

La semana siguiente fue de exploración activa. Cada persona corrió sus primeros modelos — uno con un clasificador lineal sobre texto, otro con árboles y variables tabulares, otro combinando TF-IDF con metadata — y se registraron todas las métricas y los errores más frecuentes. El objetivo era tener una línea base compartida antes de empezar a optimizar.

De ahí en adelante el calendario fue: tuning entre el 1 y el 4 de mayo, una segunda ronda de modelos del 5 al 8, análisis cruzado de resultados del 9 al 12, pipeline final del 13 al 15, y cierre con entrega del 16 al 19. Era un calendario ajustado pero realista.

En paralelo, se trazó una hoja de ruta técnica con cuatro niveles de modelo ordenados por retorno esperado. Primero los métodos de gradient boosting — LightGBM y CatBoost — porque con un dataset de texto más metadata tabulares son exactamente el tipo de problema en el que estos modelos brillan, y se entrenan en minutos. Segundo, el stacking: una vez con tres o cuatro modelos entrenados, apilar sus predicciones fuera de muestra con un meta-clasificador suele dar uno o dos puntos de F1 sin esfuerzo adicional. Tercero, transformers fine-tuned — DeBERTa o DistilBERT — que tienen el mayor techo de rendimiento pero requieren GPU. Cuarto, como mejora inmediata sin cambiar el modelo, reemplazar el embedding pequeño por `all-mpnet-base-v2` de 768 dimensiones.

---

## 2. Flujo de Trabajo / Etapas / Fases

El proyecto se organizó en siete fases experimentales que se fueron construyendo una sobre otra. Cada fase siguió el mismo ciclo: primero se documentaba la hipótesis y el razonamiento detrás de cada decisión de diseño, luego se entrenaba con validación cruzada estratificada de cinco folds más un holdout fijo del 20%, se analizaban los resultados contra lo esperado, y se definía el siguiente paso con base en lo aprendido — no en lo que se esperaba antes.

La primera fase fue Regresión Logística, que estableció la baseline lineal. Se fue iterando a través de umbral de decisión, bigrams, features de tasa de veracidad y búsqueda anidada de hiperparámetros. La segunda fue Random Forest, donde se descubrieron dos bugs de preprocesado importantes y se construyó la feature matrix completa con embeddings y NER. La tercera fue LightGBM, que reveló el problema de dominancia de features. La cuarta fue CatBoost, que atacó ese problema con su ordered boosting. La quinta fue Stacking, que combinó los cuatro modelos anteriores. La sexta fue fine-tuning de transformers con DeBERTa. La séptima y última fase fue la fusión híbrida: transformer text-only formateado con metadata en el input, k-fold de transformers, y late fusion con el stacking ensemble.

El flujo nunca fue lineal en el sentido de que cada modelo mejorara al anterior de forma ordenada. Hubo resultados que sorprendieron, features que resultaron contraproducentes, y decisiones que en retrospectiva fueron las más importantes del proyecto.

---

## 3. Herramientas y Tecnologías

El proyecto está construido completamente en Python. Para el ML clásico se usó scikit-learn en casi todo: regresión logística, random forest, escalado, codificación ordinal, validación cruzada estratificada y búsqueda de hiperparámetros con `GridSearchCV` y `RandomizedSearchCV`. LightGBM y CatBoost se instalaron como dependencias separadas pero se integran con la interfaz de scikit-learn sin problemas.

Para el texto, el pipeline depende de sentence-transformers para generar embeddings semánticos — primero con `all-MiniLM-L6-v2` de 384 dimensiones y después con `all-mpnet-base-v2` de 768. NLTK manejó el stemming con Porter y Snowball y la lematización con WordNet. spaCy con el modelo `en_core_web_sm` se encargó del reconocimiento de entidades nombradas.

Los transformers finos se corrieron con HuggingFace `transformers` y PyTorch directamente, sin ningún wrapper de alto nivel. Para los experimentos de LoRA se usaron `peft` y `bitsandbytes` para cuantización de 4 bits.

Todo el tracking de experimentos — métricas, curvas, importancia de features, artefactos — fue a través de Weights & Biases. Los modelos, opciones de preprocesado, vectorizadores y thresholds se serializaron con `joblib`. El hardware local es una RTX 5070 de 12 GB VRAM con Ryzen 7 7700X y 32 GB de RAM, más que suficiente para DeBERTa-v3-base con batch size de 16.

---

## 4. Análisis Inicial del Dataset

El dataset contiene afirmaciones políticas tomadas de PolitiFact, una plataforma de fact-checking de declaraciones de figuras públicas en Estados Unidos. Cada fila tiene ocho columnas: el identificador, la etiqueta de verdad, el texto de la afirmación, el tema o temas, el nombre del hablante, su cargo, el estado geográfico y la afiliación partidaria.

Lo primero que se nota al explorar el dataset es el desbalance de clases: el 64.75% de las afirmaciones son falsas y el 35.25% son verdaderas. No es un desbalance extremo, pero sí suficiente para que un modelo sin corrección tienda a predecir falso en casi todo. Los pesos de clase calculados a partir de ese ratio — 1.42 para verdadero y 0.77 para falso — se usaron de forma consistente en todos los experimentos.

La columna `statement` es la más predictiva del dataset, y conceptualmente también la más obvia: si vamos a detectar si una afirmación es falsa, el texto de la afirmación es donde empieza todo. Las afirmaciones son cortas — la mediana es de alrededor de 20 tokens — pero tienen patrones lingüísticos característicos: lenguaje absolutista, números específicos, negaciones, palabras de cobertura (hedge words) que suelen aparecer más en afirmaciones falsas.

El tema de la afirmación también importa mucho. Hay 115 temas únicos distribuidos según una ley de potencia: los 20 más frecuentes concentran casi el 80% de los registros. Un análisis de chi-cuadrado (estadístico: 95.87, p < 0.0001) confirma que la relación entre tema y veracidad es estadísticamente significativa. Los temas más confiables son energía (64.4% de afirmaciones verdaderas), presupuesto federal (62.2%) y elecciones (61.5%). Los más propensos a la falsedad son legal-issues y climate-change (0% verdadero), ética (4.8%) y derechos civiles (6.1%). Alrededor del 32% de las filas tiene más de un tema.

El hablante es una variable particularmente interesante y delicada. El dataset está dominado por Barack Obama, Donald Trump y Hillary Clinton, pero también incluye entidades no-persona como `chain-email` o `facebook-posts`. La distribución sigue la misma ley de potencia: pocos hablantes con muchas afirmaciones y una larga cola de hablantes con muy pocas. El patrón de mentiras por hablante — cuántas veces un político específico ha dicho cosas falsas — es la señal más predictiva del dataset, pero también la más peligrosa: si se usa incorrectamente puede introducir leakage desde el set de validación al de entrenamiento.

El cargo del hablante tiene un 27.7% de valores faltantes (2,482 filas), y la distribución es extremadamente larga: 1,018 títulos únicos de los cuales el 84.6% aparece menos de cinco veces. Los cargos con mayor tasa de mentiras son `unknown` (69.7%), `president-elect` (83.4%) y `social media posting` (81.3%). La afiliación partidaria, en cambio, no tiene valores faltantes y tiene solo 22 valores únicos. Demócratas y Republicanos cubren casi el 78% del dataset; las organizaciones no partidarias tienen la tasa de mentiras más alta (~85%), seguidas por tea-party-members (~75%) y talk-show-hosts (~70%).

---

## 5. Problemas Detectados

El primer problema, y el más persistente, fue el desbalance de clases. No importa qué modelo se corriera, la clase 0 (afirmaciones verdaderas) era consistentemente mal predicha. La precision de clase 0 arrancó en 0.47 en la baseline y el recall en 0.24 en Random Forest — el modelo simplemente predecía falso para casi todo. Esto obligó a usar pesos de clase en todos los modelos, búsqueda de umbral de decisión sobre predicciones fuera de muestra, y a monitorear Macro F1 y recall por clase en lugar de accuracy.

El segundo problema fue descubierto durante el experimento de Random Forest y resultó ser un bug real: la columna `statement` cruda — el texto sin procesar — estaba pasando por el `OrdinalEncoder` junto con las demás columnas categóricas. Cada texto es prácticamente único en el dataset, así que el encoder le asignaba un entero casi-único a cada fila. El árbol podía usar ese entero para "memorizar" la posición de cada muestra durante el entrenamiento. El modelo entrenaba perfecto pero no generalizaba nada. Otro bug relacionado era el vocabulario TF-IDF: con `max_df=0.9`, palabras como "the", "in", "of" sobrevivían al filtro porque aparecen en el 50–80% de los documentos, no en más del 90%, y terminaban siendo de los features con mayor peso. Ruido puro.

El tercer problema fue el de dominancia de features, que apareció con LightGBM y continuó con CatBoost. `fe_speaker_true_rate` — la tasa de veracidad histórica del hablante — tenía entre 6.7 y 8.76 veces la ganancia del siguiente feature en importancia. El modelo gastaba casi toda su capacidad refinando splits sobre ese único feature y prácticamente ignoraba los embeddings semánticos, los features de interacción, el NER y el resto. Era tan dominante que el modelo aprendía a identificar quién habló, no qué dijo ni cómo lo dijo. En el holdout funcionaba, pero el problema era que hablantes poco frecuentes o no vistos en entrenamiento recibían un valor de fallback de 0.5 — sin señal real.

El cuarto problema fue el leakage de target. Las tasas de veracidad por hablante, tema, partido y cargo son features extremadamente predictivas, pero si se calculan sobre todo el dataset antes de dividir en folds, el modelo ve la información del set de validación durante el entrenamiento. El remedio fue computar estas tasas únicamente sobre el split de entrenamiento de cada fold y luego aplicarlas al split de validación — lo que se llama out-of-fold encoding.

Otros problemas menores pero que se resolvieron: el threshold de decisión por defecto de 0.5 systemáticamente favorecía la clase mayoritaria y había que calibrarlo mediante búsqueda sobre las predicciones OOF; el early stopping de LightGBM calibrado contra log-loss se detuvo demasiado pronto (apenas 35 árboles en algunos folds) porque log-loss converge antes que Macro F1; y en el transformer híbrido con rama MLP paralela, el gradiente del MLP interfería con el aprendizaje del DeBERTa hasta el punto de que el modelo no aprendía nada en ninguna de las tres épocas — siempre el mismo val_macro_f1=0.5836.

---

## 6. Técnicas de Limpieza de Datos

Para el texto de las afirmaciones, el proceso fue convertir todo a minúsculas, eliminar HTML, URLs y espacios múltiples, y normalizar la puntuación. En los modelos lineales se aplicó stemming con Porter Stemmer para reducir variantes morfológicas; en los modelos de árbol y en los transformers se desactivó el stemming porque comprimir el vocabulario puede colapsar distinciones que esos modelos saben aprovechar.

Para las variables categóricas el principio fue preservar información sin crear ruido. Los valores faltantes se convirtieron explícitamente en la categoría `unknown` — no se eliminaron las filas ni se imputaron valores inventados. Todos los textos categóricos se normalizaron a minúsculas con trim de espacios. Los separadores inconsistentes en la columna de temas (pipes, slashes, puntos y comas) se estandarizaron a comas antes de hacer el split. Para el hablante, el cargo, el tema y el partido, los valores que aparecían menos de un umbral mínimo de veces (5 para speaker y speaker_job, 10 para subject) se agruparon bajo la categoría `other` — una forma de regularización que reduce la cardinalidad sin descartar la información de que el valor es infrecuente. El estado geográfico se normalizó hacia los nombres de estado de EE.UU. y se agrupó por región (Northeast, South, Midwest, West).

El identificador `id` se mantuvo únicamente para trazabilidad y se excluyó explícitamente de todas las matrices de features. En ningún momento se incluyó como predictor.

---

## 7. Técnicas contra el Sobreajuste

La validación cruzada estratificada de cinco folds fue la base de todo. Al ser estratificada, cada fold mantiene el mismo ratio de clases que el dataset completo, lo que hace las estimaciones de métricas más estables. Encima de eso, el 20% del dataset se apartó como holdout fijo desde el principio y no se tocó hasta la evaluación final del mejor modelo.

Todas las tasas de veracidad (por hablante, tema, partido y cargo) se computaron dentro del fold de entrenamiento y se aplicaron al fold de validación — nunca se vieron las etiquetas del fold de validación al momento de calcular esos promedios. Lo mismo se hizo para el set de test en la submisión.

La búsqueda de hiperparámetros se hizo con nested CV: dentro de cada fold exterior se corría un `RandomizedSearchCV` con tres folds internos y 20 iteraciones. Así, los hiperparámetros seleccionados no tienen acceso a las etiquetas del fold de validación exterior. Sin eso, la selección de hiperparámetros introduce un sesgo optimista que infla las métricas en cross-validation.

Para la Regresión Logística, la regularización fue C (que controla qué tan fuerte es la penalización) y el tipo de penalización (L1 vs L2). L1 hace feature selection automática sobre el TF-IDF esparcido, zeroeando los pesos de palabras irrelevantes. Para Random Forest, `min_samples_leaf=6` fue el resultado del HP search — exigir mínimo seis muestras por hoja evita que los árboles memoricen ruido en celdas muy pequeñas. `max_features=0.5` (50% de features por split) aumenta la diversidad entre árboles. Para LightGBM, `num_leaves=31` y `min_child_samples=48` fueron los valores que el search seleccionó de forma casi unánime en todos los folds — el modelo prefería configuraciones conservadoras porque la feature `fe_speaker_true_rate` ya le daba mucha señal desde los primeros splits. Para CatBoost, el `depth=4` (16 patrones de hoja por árbol) y `l2_leaf_reg=5` fueron también elegidos por unanimidad. El ordered boosting de CatBoost es en sí mismo una forma de regularización implícita: al calcular las estadísticas de target con solo las muestras previas en una permutación aleatoria, el modelo no puede sobreajustar a las identidades exactas del training set.

El threshold de decisión se buscó en un grid de 0.20 a 0.76 en pasos de 0.02, evaluando Macro F1 sobre las predicciones OOF. Nunca se usó el holdout para calibrar el threshold — eso habría introducido un sesgo optimista en la evaluación final.

Para los transformers, se usó Layer-wise Learning Rate Decay (LLRD): las capas más profundas del DeBERTa aprenden con la tasa de aprendizaje completa, y las capas más bajas con tasas progresivamente menores. Esto evita destruir las representaciones aprendidas durante el preentrenamiento. Se añadió dropout de 0.3 en la cabeza de clasificación. El mejor checkpoint de cada corrida se seleccionó por val_macro_f1, no por val_loss.

En el stacking, el meta-clasificador usa `C=0.1` — regularización suave que evita que la meta-LR asigne pesos extremos a alguno de los modelos base basándose en patrones de los folds.

---

## 8. Feature Engineering

El pipeline de features se diseñó para capturar señales desde tres ángulos distintos: el texto de la afirmación, las características del hablante y el contexto político, y las interacciones entre ambos.

Del texto de la afirmación se extrajeron dos tipos de representaciones. La primera es vectorial-semántica: embeddings de oraciones con `all-MiniLM-L6-v2` (384 dimensiones) o `all-mpnet-base-v2` (768 dimensiones), que comprimen el significado semántico completo de la afirmación en un vector denso. La segunda son features léxicas y estructurales: longitud en caracteres y palabras, ratio de mayúsculas, densidad de dígitos, frecuencia media de tokens, conteo de errores ortográficos, conteo de entidades nombradas (personas, organizaciones, lugares, fechas, números con spaCy), y features lingüísticas de detección de engaño — conteo de negaciones, palabras de cobertura como "podría" o "dicen que", lenguaje absolutista ("siempre", "el más", "nunca"), números específicos y estadísticas, score de legibilidad Flesch-Kincaid, y polaridad y subjetividad de sentimiento con TextBlob.

De las variables sobre el hablante y el contexto se extrajeron features de frecuencia (qué tan común es este hablante en el dataset), flags de rareza (si es un hablante infrecuente), longitud del nombre, si tiene prefijo de título (senator, governor, doctor), y las tasas de veracidad históricas computadas out-of-fold para hablante, tema, partido y cargo. Esta última familia de features — `fe_speaker_true_rate`, `fe_subject_true_rate`, `fe_party_true_rate` y `fe_speaker_job_true_rate` — resultó ser la más predictiva del dataset. La lógica es directa: si un político ha mentido el 80% de las veces en el training fold, es probable que vuelva a mentir.

Las features de interacción se generaron como claves compuestas de pares de columnas, codificadas con `OrdinalEncoder` para los árboles. Las más informativas fueron speaker × subject (este hablante hablando de este tema), speaker × party, subject × party, speaker_job × subject, state × party y speaker × bucket de longitud del statement. Los modelos lineales no pueden aprender estas interacciones directamente, pero los árboles las explotan con facilidad: una sola condición de split en la clave compuesta es equivalente a dos splits secuenciales.

Además de los features de interacción, se construyeron agregados no leaky por grupo: longitud media de statements por hablante, ratio de puntuación promedio por hablante, ratio de números promedio por hablante. Se llaman "no leaky" porque no usan las etiquetas — son estadísticas del texto, no del label.

---

## 9. Recorrido de Experimentos

### Experimento 1 — Regresión Logística

El primer experimento buscaba establecer una baseline lineal sólida. No la primera cosa que funcionara, sino la mejor configuración lineal posible, partiendo de sweeps individuales por módulo. El preprocesado seleccionado usó TF-IDF de unigrams con 5,000 features, stemming Porter, features léxicas de negación, cobertura, lenguaje absolutista y numerales, más readability y sentimiento. Los pesos de clase se fijaron a los valores del dataset.

La baseline arrancó en Macro F1 = 0.604 con ROC-AUC = 0.654. El problema inmediato era visible en el desglose por clase: precision de 0.47 en clase 0, lo que significa que casi la mitad de todo lo que el modelo llamaba "verdadero" era en realidad falso. El modelo tenía un sesgo claro hacia predecir falso.

El primer intento de corrección fue tuning de umbral: buscar en el grid [0.20, 0.76] el punto de corte que maximizara Macro F1 sobre las predicciones OOF. El tuner encontró 0.46 — apenas por debajo del default — y el F1 bajó de 0.604 a 0.596. El problema era que las probabilidades del modelo no estaban bien separadas, y ningún umbral puede rescatar distribuciones de probabilidad que se traslapan.

Cambiar a bigrams (bigramas de hasta 10,000 features) sí ayudó: Macro F1 subió a 0.607, la clase 0 mejoró de 0.48 a 0.51 de F1. Las frases políticas características como "never raised taxes" o "lowest unemployment ever" llevan más señal como unidad que sus palabras individuales.

El paso más esperado fue agregar las true-rate features dentro de los folds, combinado con un cambio de TF-IDF a embeddings (`all-MiniLM-L6-v2`). El resultado fue mixto: el ROC-AUC mejoró al mejor valor del experimento (0.665), pero el Macro F1 bajó a 0.603. Los embeddings comprimen información semántica rica pero pierden la especificidad de palabra que el TF-IDF capturaba barato.

El nested CV fue la mejora más limpia: Macro F1 = 0.611, el mejor valor del experimento, con la menor brecha entre CV y holdout — señal de que el modelo generalizaba bien. La calibración isotónica fue el peor resultado: Macro F1 = 0.555, con recall de clase 0 colapsando a 0.24. La calibración interactuó mal con el tuner de umbral en datos pequeños e imbalanceados. La lección fue clara: isotonic calibration en datasets pequeños con class weights es impredecible.

El techo de la regresión logística es la linearidad misma. No puede aprender que un hablante republicano hablando de economía durante una elección tiene más probabilidad de mentir que cualquiera de esas features predicha por separado.

---

### Experimento 2 — Random Forest

Random Forest tiene la ventaja de entrenar en un solo paso, sin tasa de aprendizaje que calibrar, y siendo más difícil de sobreajustar accidentalmente que el gradient boosting. Se usó como estimador de techo para la feature matrix completa antes de comprometerse con los modelos más lentos.

La primera corrida reveló dos bugs críticos. El más serio: la columna `statement` (el texto crudo) estaba siendo codificada como entero por el `OrdinalEncoder`, funcionando como pseudo row-ID. Con 8,950 textos casi-únicos, el árbol podía hacer un split por cada texto y aprenderse el dataset de entrenamiento de memoria. Quitarlo eliminó ese feature de la lista de importancias y redistribuyó su peso a features reales. El segundo bug eran los stopwords en el TF-IDF: bajando `max_df` de 0.9 a 0.7 y activando filtrado de stopwords antes de construir el vocabulario se limpiaron de los top-30 features `vec_the`, `vec_in` y `vec_of`.

Con los bugs corregidos, el trabajo real empezó. Tuning de umbral OOF encontró 0.58 — el umbral subió porque `class_weight={0: 1.42, 1: 0.77}` hace que el modelo internamente penalice más los errores de clase 1 durante el entrenamiento, empujando las probabilidades crudas hacia arriba. El recall de clase 0 saltó de 0.26 a 0.46.

El HP search encontró `min_samples_leaf=6` como el valor óptimo de regularización. Con el default de 1, los árboles crecían hojas de una sola muestra y memorizaban ruido. Con 6, forzaban splits generalizables. El threshold volvió a 0.50 — las probabilidades se re-centraron con mayor regularización.

El salto más grande fue cambiar los 500 features TF-IDF por embeddings de 384 dimensiones con `all-MiniLM-L6-v2`. Macro F1 subió de 0.5942 a 0.6080, y `max_features` óptimo cambió de `sqrt` a 0.5 porque con 870 columnas totales (384 embedding + metadata), el modelo necesitaba ver más features por split para combinar dimensiones de embedding de forma efectiva.

El último paso fue agregar conteos NER de spaCy y `fe_speaker_job_true_rate`. La tasa de veracidad por cargo debutó en el puesto 2 de importancia con 0.0139 — la mayor entrada de debut de cualquier feature en el proyecto. El recall de clase 0 llegó a 0.60, y el Macro F1 final fue 0.6209 con ROC-AUC = 0.6674.

---

### Experimento 3 — LightGBM

LightGBM es el paso natural después de Random Forest: gradient boosting con crecimiento leaf-wise (el nodo con mayor ganancia en todo el árbol en lugar de nivel a nivel) es más eficiente en el uso de cada split y generalmente supera a RF en datos tabulares mixtos.

La predicción antes de correr era Macro F1 entre 0.67 y 0.72. El resultado real fue 0.5934 en CV y 0.6062 en holdout — comparable o inferior al RFC en varios indicadores. Algo estaba mal.

La causa la reveló la importancia de features: `fe_speaker_true_rate` tenía 6.7 veces la ganancia del siguiente feature. El HP search convergió de forma casi unánime hacia la configuración más conservadora disponible (num_leaves=31 — el mínimo del grid, min_child_samples=48 — muy alto). El mensaje era claro: el modelo estaba sobreajustando al feature dominante y tratando de compensar con regularización máxima, en lugar de aprender de la feature matrix completa.

Se intentaron tres correcciones. La opción A fue early stopping calibrado contra log-loss: la idea era dejar que el modelo use tantos árboles como sean útiles. El resultado fue peor — el early stopping disparó después de apenas 35 árboles en promedio porque log-loss converge antes que Macro F1. El feature dominante aprende rápido y después los árboles adicionales mejoran F1 sin mejorar log-loss.

La opción B fue eliminar `fe_speaker_true_rate` directamente. El CV apenas se movió (+0.001), pero el holdout mejoró 0.012 puntos de F1 y 0.011 de ROC-AUC. Eso es el patrón clásico de un feature que memoriza identidades del training set: funciona en CV porque los hablantes del fold de validación sí aparecen en el fold de entrenamiento, pero en el holdout hay hablantes con menos historial y el fallback de 0.5 no da señal real.

La opción C fue agregar regularización L1/L2 (`reg_alpha`, `reg_lambda`) al grid de búsqueda. Tuvo un bug en la primera corrida — los parámetros se buscaban en el CV interno pero no llegaban al modelo final. Corregido el bug, el resultado fue peor que la opción B: la L1/L2 no logró reducir la dominancia del feature de la misma manera que simplemente quitarlo.

La conclusión fue que el mejor LGBM era la opción B, con Macro F1 = 0.6179 y ROC-AUC = 0.6790 — este último el récord del proyecto hasta ese momento.

---

### Experimento 4 — CatBoost

CatBoost ataca el problema de dominancia de features desde la raíz, con dos mecanismos que LightGBM no tiene. El ordered boosting calcula las estadísticas de target para cada muestra usando solo las muestras que vinieron antes en una permutación aleatoria — nunca la muestra misma. Eso hace que las estimaciones de true-rate sean más ruidosas por muestra durante el entrenamiento, lo que reduce la ventaja injusta que `fe_speaker_true_rate` tenía. Los árboles simétricos (cada nodo a igual profundidad usa la misma condición de split) son un regularizador implícito fuerte que limita cuánto puede profundizar cualquier feature individual en un solo árbol.

La primera corrida de CatBoost, con todos los features incluyendo `fe_speaker_true_rate`, dio Macro F1 = 0.6184 y ROC-AUC = 0.6653. El dato más interesante fue el threshold óptimo: 0.48 — la primera vez en el proyecto que caía por debajo de 0.5. El modelo estaba asignando probabilidades más bien calibradas, sin el sesgo hacia clase 1 que LGBM producía con su greedy leaf-wise optimization.

El ratio de dominancia era todavía 8.76× (peor que LGBM en números absolutos, aunque la métrica de importancia no es directamente comparable). Pero CatBoost nativo de categorical handling sí empezó a mostrar `speaker_grouped` en el tercer puesto de importancia — evidencia de que las estadísticas de target internas del modelo estaban extrayendo señal de las columnas categóricas que LGBM trataba solo como enteros ordinales.

Combinar CatBoost con la opción B (eliminar `fe_speaker_true_rate`) produjo el mejor modelo individual del proyecto: Macro F1 = 0.6294, ROC-AUC = 0.6740, threshold = 0.46. El ratio de dominancia bajó de 8.76× a 3.49×, y `fe_speaker_job_true_rate` saltó al primer puesto de importancia con un peso 3.49× mayor que el siguiente feature. `speaker_grouped` y `party_affiliation_grouped` — features categóricos nativos — aparecieron en el top-5. El modelo por fin estaba aprendiendo de toda la feature matrix.

---

### Experimento 5 — Stacking

Con cuatro familias de modelos entrenadas — LR, RFC, LGBM y CatBoost — el stacking era el siguiente paso lógico. La hipótesis era que cada modelo comete errores distintos: LR dibuja una frontera lineal en el espacio de embeddings, RFC construye árboles independientes con bootstrap, LGBM hace gradient boosting leaf-wise, CatBoost usa árboles simétricos con estadísticas ordenadas. Sus predicciones fuera de muestra son complementarias — y un meta-clasificador puede aprender a mezclarlas mejor que cualquier peso fijo.

El diseño fue directo: todos los modelos reciben la misma feature matrix (configuración CatBoost+OptB, con `drop_speaker_true_rate=True`), cada uno genera sus probabilidades OOF en el loop de CV, esas cuatro columnas se apilan en una matriz de meta-features, y un `LogisticRegression` con `C=0.1` aprende los pesos óptimos. La meta-LR no tiene class_weight — los modelos base ya aplican el suyo y no tiene sentido duplicar la corrección.

La primera corrida, con LGBM usando `num_leaves=31`, dio Macro F1 = 0.6303 y ROC-AUC = 0.6830. El nuevo récord en ROC-AUC (superando el 0.6790 de LGBM-OptB), pero solo una mejora marginal en Macro F1 sobre CatBoost+OptB (+0.0009). El threshold óptimo fue 0.62 — el más alto del proyecto. La razón: los cuatro modelos base aplican class_weight que empuja las probabilidades de clase 1 hacia arriba, y la meta-LR sin class_weight hereda ese sesgo y necesita un umbral alto para compensar.

Los coeficientes del meta-clasificador revelaron algo inesperado: RFC tenía el coeficiente más alto (1.644), a pesar de no ser el modelo con mejor ROC-AUC individual. LGBM tenía el coeficiente más bajo (0.570) a pesar de tener el mejor ROC-AUC individual. La explicación más plausible es que RFC, por su bootstrap aggregation y subsampling de features, produce errores decorrelacionados de los dos modelos boosted. Cuando LGBM y CatBoost se equivocan en la misma muestra, RFC frecuentemente la acierta — y el meta-clasificador aprendió a valorar esa señal correctiva.

La segunda corrida, subiendo LGBM a `num_leaves=63`, confirmó la hipótesis: el coeficiente de LGBM subió de 0.570 a 0.695, el de RFC bajó de 1.644 a 1.506 (menor dominancia), y el Macro F1 holdout mejoró de 0.6303 a 0.6323 con ROC-AUC = 0.6835. Nuevos récords en ambas métricas.

Antes de pasar a transformers, se upgradeó el embedding de `all-MiniLM-L6-v2` (384-dim) a `all-mpnet-base-v2` (768-dim) en todo el pipeline de stacking. Ese cambio solo — sin modificar ningún modelo — llevó el Macro F1 de 0.6323 a **0.6428**, el nuevo techo que los transformers necesitarían superar.

---

### Experimento 6 — Fine-Tuning de Transformers

La motivación para los transformers es que los embeddings de `all-mpnet-base-v2`, por buenos que sean, son representaciones estáticas: el modelo no fue entrenado con afirmaciones políticas de PolitiFact, y no puede ajustar sus representaciones a las señales específicas de este dataset. Un transformer fine-tuned lee el texto token por token y aprende, dentro del proceso de entrenamiento, qué palabras y frases predicen falsedad en este corpus específico.

Se eligió DeBERTa-v3-small como primer candidato: 86M de parámetros, rendimiento superior a DistilBERT en benchmarks de NLI y fact-checking, y encaja cómodo en los 12 GB de VRAM de la RTX 5070. La arquitectura fue la más simple posible: tokens de la afirmación → encoder DeBERTa → representación [CLS] → dropout → Linear(768→2). Cross-entropy con pesos de clase.

La primera corrida text-only alcanzó Macro F1 = 0.6128 en el mejor checkpoint (época 1 de 3). En épocas subsecuentes el val loss subía — overfitting clásico en un dataset pequeño. A través de siete runs refinando freeze strategy, LLRD, learning rate y dropout, el mejor resultado fue Macro F1 = 0.6205 con ROC-AUC = 0.6700. Bueno, pero todavía por debajo del stacking con mpnet (0.6428).

El transformer text-only tiene una limitación estructural: no tiene acceso a las true-rate features de hablante, tema y partido — la señal más predictiva del dataset. El texto por sí solo no es suficiente para alcanzar el techo del stacking.

---

### Experimento 7 — Hybrid Transformers

Esta fase buscó tres cosas: mejor fusión de texto con metadata, más cobertura de datos con k-fold, y modelos más grandes.

**Run 1 — MLP híbrido paralelo.** La primera arquitectura combinó el [CLS] de DeBERTa (768-dim) con una rama MLP que procesaba cuatro features de metadata (true-rates de hablante, tema, partido y `is_major_party`), concatenando las dos salidas antes de la capa de clasificación. El resultado fue un desastre: val macro_f1 = 0.5836 en todas las épocas sin ninguna mejora. El gradiente del MLP destabilizaba el aprendizaje del DeBERTa. Con un dataset de 8,950 muestras y dos rutas de gradiente compitiendo, el modelo no convergía.

**Experimento 1 — Text Formatting.** La solución fue más elegante: en lugar de un MLP paralelo, inyectar la metadata directamente como tokens de texto al inicio del input. El formato fue `"speaker: john edwards | party: democrat | subject: health-care | {statement}"`. Así el propio mecanismo de atención de DeBERTa puede cruzar información entre el texto de la afirmación y la identidad del hablante — sin ningún cambio de arquitectura. El resultado fue Macro F1 = 0.6254 y ROC-AUC = 0.6885. El recall de clase 0 (afirmaciones verdaderas) mejoró de 0.55 a 0.65 — el mejor resultado de recall de clase 0 hasta ese momento.

El modelo todavía estaba aprendiendo al terminar la época 3, así que se corrió de nuevo con `FREEZE_EPOCHS=0` y 5 épocas. El mejor checkpoint fue época 3 con Macro F1 = 0.6392. Las épocas 4 y 5 mostraron overfitting.

**Experimento 2 — K-Fold.** Entrenar en 5 folds en lugar de un solo split da a cada muestra un turno en el training set. El resultado fue Macro F1 = 0.6393 — prácticamente igual al mejor single-split. El hallazgo fue que la época 2 era siempre la mejor en todos los folds; usar `EPOCHS=2` en el futuro ahorraría tiempo.

**Experimento 3 — Late Fusion.** En lugar de hacer el transformer standalone, se usaron sus predicciones OOF como quinto modelo base en el stacking. El meta-LR re-aprendió los pesos con las cinco columnas. Resultado: Macro F1 = 0.6435, ROC-AUC = 0.7089. **Primer ROC-AUC por encima de 0.70 en todo el proyecto.** El transformer recibió coeficiente 1.68 — el meta-clasificador lo valoró mucho por sus errores decorrelacionados de los árboles.

**Experimento 4a — DeBERTa-v3-base standalone.** Swappear el modelo a la versión base (184M parámetros) con k-fold dio Macro F1 = 0.6397 y ROC-AUC = 0.7004 — apenas mejor que small.

**Experimento 4b — DeBERTa-v3-base + Late Fusion.** La combinación del modelo más grande con el late fusion en el stacking dio el **mejor resultado del proyecto**: Macro F1 = **0.6472**, ROC-AUC = **0.7109**. El coeficiente del transformer subió de 1.68 a 1.97 — evidencia de que el modelo base captura señal de texto que los árboles definitivamente no tienen.

---

## 10. Resumen de Resultados

El proyecto partió de una Regresión Logística con Macro F1 = 0.604 y terminó con DeBERTa-v3-base en late fusion con Macro F1 = 0.647 y ROC-AUC = 0.711. Los saltos más grandes fueron: el upgrade de TF-IDF a embeddings en Random Forest (+0.014 F1), la eliminación del feature dominante `fe_speaker_true_rate` en LGBM y CatBoost (confirmando que memorizaba identidades en lugar de generalizar), el upgrade de embeddings de 384 a 768 dimensiones en el stacking (+0.01 F1), y la incorporación del transformer como quinto modelo base mediante late fusion (+0.005 F1 y +0.009 ROC-AUC sobre el stacking puro).

El camino no fue lineal. Hubo experimentos que empeoraron las cosas — la calibración isotónica en LR, el early stopping de LightGBM, el MLP híbrido paralelo — y cada uno enseñó algo sobre los límites del enfoque. La feature más predictiva del dataset resultó también ser la que más sobreajustaba. Y el modelo que parecía más simple al final — inyectar metadata como texto en el input del transformer — fue más efectivo que la arquitectura híbrida explícita.

---

## 11. Lecciones Aprendidas

La primera lección es que la señal de texto es necesaria pero no suficiente por sí sola. El feature más predictivo en todos los modelos de árbol fue siempre la tasa de veracidad histórica del hablante, no los embeddings semánticos — pero esa misma feature era la que más sobreajustaba cuando se calculaba mal.

La segunda lección es que eliminar el feature dominante mejoró la generalización. Contra la intuición, quitar `fe_speaker_true_rate` mejoró el holdout en LGBM (+0.012 F1) y también fue aditivo con CatBoost. El feature estaba memorizando identidades del training set, no capturando un patrón real sobre el lenguaje de falsedad.

La tercera lección es que el stacking explota complementariedad real. Random Forest, pese a ser el modelo más simple de los cuatro, recibió el coeficiente más alto en el meta-clasificador porque sus errores son los más decorrelacionados de los modelos boosted. La diversidad de mecanismos de aprendizaje — bagging vs leaf-wise boosting vs árboles simétricos con ordered statistics — se traduce en errores estructuralmente distintos.

La cuarta es que el texto formateado supera al MLP híbrido. Inyectar metadata como tokens de texto es más efectivo que una rama paralela de red neuronal en datasets pequeños, porque evita la interferencia de gradientes y le da al transformer su representación nativa para razonar sobre speaker, party y subject.

La quinta es que los transformers ganan más en late fusion que como modelos standalone. DeBERTa-base solo alcanzó 0.640; en late fusion con el stacking llegó a 0.647. El transformer aporta señal de texto puro que los árboles no tienen, y el stacking sabe aprovecharlo.

La sexta es que la calibración del threshold de decisión es crítica con datos imbalanceados. El umbral óptimo nunca fue exactamente 0.5 — varió entre 0.46 (CatBoost, bien calibrado) y 0.62 (stacking con meta-LR sin class_weight). Los modelos con mejor calibración probabilística, como CatBoost, requieren menos corrección post-hoc.

La séptima lección es la más general: los mejores resultados no vinieron de modelos más complicados, sino de entender bien las fallas. El bug del pseudo row-ID en Random Forest, el problema de dominancia en LGBM, la interferencia de gradientes en el MLP híbrido — resolverlos fue más valioso que agregar capas o features adicionales.
