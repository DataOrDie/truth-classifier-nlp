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



------

Next Steps — Recommended Order                                                                                                                                       
                                                                                                                                                                       
  Tier 1: Gradient Boosting (highest ROI, run on PC today)                                                                                                             
                                                                                                                                                                       
  Your feature matrix — 384-dim embeddings + rich metadata — is exactly what gradient boosting excels at. These typically beat both LR and RFC on tabular-NLP blends.

  XGBoost or LightGBM — drop-in for your existing rfc.py pattern, trains in minutes on CPU, supports class_weight, has built-in HP search. LightGBM is faster; XGBoost
  has slightly better defaults out of the box.

  CatBoost — handles categorical columns (speaker, party, subject) natively without OrdinalEncoder. Particularly strong when you have high-cardinality string features.
   Very little tuning needed.

  Why these first: Literature on PolitiFact-style datasets consistently puts gradient boosting at or above RF. Adding true-rate features (which you already have) gives
   XGB/CatBoost a big edge.

  ---
  Tier 2: Ensemble / Stacking (quick win once you have 3+ models)

  You already collect OOF probabilities in your CV loop. Once you have LR + RFC + XGB probas, stack them with a simple LR or ridge meta-learner. This is often worth
  +1–2 F1 points with zero new features.

  ---
  Tier 3: Fine-tuned Transformers (best ceiling, needs Kaggle GPU)

  A fine-tuned DistilBERT or DeBERTa-v3-small on the raw statement column will outperform all of the above on the text signal — but metadata features (speaker, party,
  true-rates) won't be directly used. The best approach is a hybrid: transformer text head + metadata MLP head → combined classifier.

  - On Kaggle: free T4 GPU, fits DistilBERT or deberta-v3-small fine-tuning in ~20 min
  - On PC without GPU: too slow for full fine-tuning, but you could use the transformer just for embeddings (larger model than all-MiniLM-L6-v2, e.g.
  all-mpnet-base-v2)

  ---
  Tier 4: Bigger Embeddings (easy upgrade, no new model)

  Swap all-MiniLM-L6-v2 (384-dim) for all-mpnet-base-v2 (768-dim) in your current LR/RFC/XGB runs. Often +1–2 points with zero code change beyond the model name.
  Slower preprocessing but same training code.

  ---
  Concrete recommended sequence

  ┌──────┬───────────────────────────────────────────┬────────┬────────────────────────────────────┐
  │ Step │                   Model                   │ Where  │          New dependencies          │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 1    │ LightGBM                                  │ PC     │ pip install lightgbm               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 2    │ CatBoost                                  │ PC     │ pip install catboost               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 3    │ Stacking ensemble                         │ PC     │ already have sklearn               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 4    │ Bigger embeddings (all-mpnet-base-v2)     │ PC     │ already have sentence-transformers │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 5    │ Fine-tuned DistilBERT or DeBERTa          │ Kaggle │ transformers, torch                │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  Slower preprocessing but same training code.

  ---
  Concrete recommended sequence

  ┌──────┬───────────────────────────────────────────┬────────┬────────────────────────────────────┐
  │ Step │                   Model                   │ Where  │          New dependencies          │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 1    │ LightGBM                                  │ PC     │ pip install lightgbm               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 2    │ CatBoost                                  │ PC     │ pip install catboost               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 3    │ Stacking ensemble                         │ PC     │ already have sklearn               │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 4    │ Bigger embeddings (all-mpnet-base-v2)     │ PC     │ already have sentence-transformers │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 5    │ Fine-tuned DistilBERT or DeBERTa          │ Kaggle │ transformers, torch                │
  ├──────┼───────────────────────────────────────────┼────────┼────────────────────────────────────┤
  │ 6    │ Hybrid: fine-tuned transformer + metadata │ Kaggle │ same                               │
  └──────┴───────────────────────────────────────────┴────────┴────────────────────────────────────┘
