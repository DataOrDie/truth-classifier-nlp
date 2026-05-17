# Informe del Proyecto: Truth Classifier NLP

**Kaggle — UPM Course Competition**
**Tarea:** Clasificación binaria de afirmaciones políticas como verdaderas (0) o falsas (1).
**Dataset:** 8,950 muestras de entrenamiento + 3,860 de test sin etiqueta.
**Métrica objetivo:** Macro F1 Score.

---

## 1. Planificación

El proyecto fue planificado con un equipo de 6 personas trabajando de forma independiente, cada una probando modelos distintos. El calendario de trabajo fue:

| Fechas | Actividad |
|--------|-----------|
| 24/04 | Primera reunión — revisión del enunciado, dataset y reglas de entrega. Reparto de trabajo. |
| 25/04 | Revisión de datos y preprocesado inicial. Criterios comunes: limpieza de texto, nulos, categorías. Detección de ruido, duplicados, desbalance. |
| 26/04 | Línea base compartida: baseline común, misma métrica de evaluación, registro de errores. |
| 27/04 – 30/04 | Modelos independientes (fase 1): modelo lineal sobre texto, árbol con variables tabulares, TF-IDF + metadata, etc. |
| 01/05 – 04/05 | Tuning y mejora: hiperparámetros, variantes de preprocesado, selección de features. |
| 05/05 – 08/05 | Modelos independientes (fase 2): segunda familia o mejora clara del primer intento. |
| 09/05 – 12/05 | Análisis cruzado: comparación de los 6 modelos, elección de 2–3 enfoques, decisión sobre ensamblado. |
| 13/05 – 15/05 | Pipeline final: reentrenamiento, reproducibilidad, generación del CSV de test. |
| 16/05 – 19/05 | Cierre y entrega: últimas submissions, código final, memoria de grupo y presentación. |

La hoja de ruta técnica preveía 4 niveles de modelos por orden de ROI:

1. **Tier 1 — Gradient Boosting** (LightGBM / CatBoost): datos mixtos texto+tabular, entrena en minutos.
2. **Tier 2 — Stacking**: apilar OOF probabilities de múltiples modelos con un meta-clasificador.
3. **Tier 3 — Transformers fine-tuned**: DeBERTa / DistilBERT en GPU (Kaggle T4 o RTX 5070).
4. **Tier 4 — Embeddings más grandes**: `all-mpnet-base-v2` (768-dim) como upgrade inmediato.

---

## 2. Flujo de Trabajo / Etapas / Fases

El proyecto se organizó en siete fases experimentales claramente delimitadas:

```
Phase 1 — Logistic Regression (baseline lineal)
  └── TF-IDF → umbral, bigrams, true-rate features, nested CV

Phase 2 — Random Forest (baseline de árbol)
  └── Embeddings → NER, speaker_job_true_rate

Phase 3 — LightGBM (gradient boosting)
  └── Opciones A/B/C: early stopping, drop featur, L1/L2

Phase 4 — CatBoost (boosting con ordered statistics)
  └── Variante +OptB (drop speaker_true_rate)

Phase 5 — Stacking (ensemble de 4 modelos)
  └── Meta-LR sobre OOF probabilities → Run 2 (num_leaves=63)

Phase 6 — Transformers (fine-tuning DeBERTa)
  └── Text-only → Formatted input → K-Fold → Late Fusion

Phase 7 — Hybrid Transformers (DeBERTa-v3-base + stacking)
  └── Exp 1–4b: text format, k-fold, late fusion, upgrade a base model
  └── Exp 5: LoRA fine-tuning (7B LLM)
```

Cada fase siguió el mismo ciclo:

1. Decisión de diseño documentada (hipótesis + razonamiento).
2. Ejecución con 5-fold estratificado + holdout 20%.
3. Análisis de resultados vs. expectativas.
4. Plan de siguiente paso.

---

## 3. Herramientas y Tecnologías

| Categoría | Herramienta | Uso |
|-----------|-------------|-----|
| **Lenguaje** | Python 3.x | Todo el pipeline |
| **ML clásico** | scikit-learn | LR, RFC, StandardScaler, OrdinalEncoder, StratifiedKFold, GridSearchCV, RandomizedSearchCV |
| **Gradient Boosting** | LightGBM | Fase 3 |
| **Gradient Boosting** | CatBoost | Fase 4 — con soporte nativo de categóricas y ordered boosting |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) | Vectorización semántica de statements |
| **NLP** | NLTK (Porter/Snowball stemmer, WordNet lemmatizer) | Limpieza y normalización de texto |
| **NER** | spaCy `en_core_web_sm` | Conteo de entidades nombradas (PERSON, ORG, GPE, DATE...) |
| **Transformers** | HuggingFace `transformers` + PyTorch | Fine-tuning DeBERTa-v3-small/base |
| **Fine-tuning eficiente** | `peft` + `bitsandbytes` | LoRA sobre modelos 7B |
| **Tracking** | Weights & Biases (W&B) | Logging de métricas, curvas, feature importance, artefactos |
| **Serialización** | `joblib` | Guardado de modelos, opciones, vectorizadores y threshold |
| **Hardware** | RTX 5070 (12 GB VRAM), Ryzen 7 7700X, 32 GB RAM | Entrenamiento local de transformers |
| **GPU cloud** | Kaggle (T4) | Alternativa a GPU local |

---

## 4. Análisis Inicial del Dataset (de `aboutData.md`)

### Descripción general

El dataset consiste en afirmaciones políticas de la base de datos PolitiFact, con 8 columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | identificador | No predictivo; solo para trazabilidad |
| `label` | target binario | 0 = verdadero, 1 = falso |
| `statement` | texto | El texto de la afirmación política |
| `subject` | categórica | Tema(s) de la afirmación (hasta 4 temas por fila) |
| `speaker` | categórica | Persona que hace la afirmación |
| `speaker_job` | categórica | Cargo/profesión del hablante |
| `state_info` | categórica | Estado geográfico (EE.UU.) |
| `party_affiliation` | categórica | Partido político del hablante |

### Distribución de clases

El dataset presenta **desbalance moderado**:
- Clase 0 (verdadero): **35.25%** (3,155 muestras)
- Clase 1 (falso): **64.75%** (5,795 muestras)

Los pesos de clase calculados son: `{0: 1.42, 1: 0.77}`, usados de forma consistente en todos los modelos.

### Hallazgos por columna

#### `subject` — Tema de la afirmación
- 115 temas únicos; distribución en ley de potencia.
- El top 20 cubre el **79.5%** de los registros.
- **Relación estadísticamente significativa con el label** (chi-cuadrado: 95.87, p < 0.0001).
- Temas más verídicos: energía (64.4%), presupuesto federal (62.2%), elecciones (61.5%).
- Temas más falsos: legal-issues (0%), climate-change (0%), ethics (4.8%).
- El 32% de las filas tiene múltiples temas.

#### `speaker` — Hablante
- Alta cardinalidad con cola larga.
- Dominado por `barack-obama`, `donald-trump`, `hillary-clinton`.
- Incluye entidades no-persona: `chain-email`, `facebook-posts`.
- El patrón de mentiras por hablante es un **predictor clave** del dataset (riesgo de leakage si se usa mal).

#### `speaker_job` — Cargo del hablante
- 1,018 títulos únicos; 27.7% missing (2,482 filas).
- Alta cardinalidad: 84.6% de los títulos únicos aparecen menos de 5 veces.
- Cargos con mayor tasa de mentiras: `unknown` (69.7%), `president-elect` (83.4%), `social media posting` (81.3%).

#### `party_affiliation` — Partido político
- 22 valores únicos; sin valores faltantes.
- El top 2 (Democrat + Republican) cubre el **77.8%**.
- Tasa de mentiras mayor en: organizations (~85%), tea-party-member (~75%), talk-show-host (~70%), Republican (~64%).
- Tasa menor en: business-leader (~40%), state-official (~45%).

#### `state_info` — Estado geográfico
- Variabilidad geográfica; muchos valores raros.
- Útil principalmente como frecuencia y región.

#### `statement` — El texto de la afirmación
- La columna más predictiva.
- Mediana de ~20 tokens; máximo histórico 467 tokens.
- Contiene señales lingüísticas características de fake news: lenguaje absolutista, negaciones, números específicos.

---

## 5. Problemas Detectados

| Problema | Descripción | Impacto |
|----------|-------------|---------|
| **Desbalance de clases** | 35% vs 65%; clase 0 (verdadero) constantemente sub-predicha | Alto — precision de clase 0 arranca en 0.47 |
| **Alta cardinalidad** | `speaker` (2,634), `speaker_job` (1,018), claves de interacción | Riesgo de sobreajuste con one-hot encoding |
| **Feature dominance** | `fe_speaker_true_rate` monopolizaba el 6.7–8.76× la ganancia del siguiente feature en LGBM/CatBoost | El modelo ignoraba otros predictores válidos |
| **Leakage de target** | Calcular tasas de veracidad por hablante en todo el dataset introduce información del set de validación | Invalida la estimación de rendimiento |
| **Bug: columna `statement` en OrdinalEncoder** | La columna cruda de texto era encodificada como integer casi-único, funcionando como pseudo row-ID | El modelo "memorizaba" las filas de entrenamiento |
| **Stopwords en TF-IDF** | `vec_the`, `vec_in`, `vec_of` en top features por `max_df=0.9` demasiado alto | El vocabulario TF-IDF era ruido más que señal |
| **Threshold subóptimo** | Con desequilibrio de clases, el threshold 0.5 favorece sistemáticamente la clase mayoritaria | Clase 0 recall tan bajo como 0.24 |
| **Overfitting del transformer** | 8,950 muestras es pequeño para fine-tuning completo; val loss sube tras epoch 1–2 | Solo el checkpoint de mejor epoch es útil |
| **MLP branch gradient interference** | En el hybrid transformer, el gradiente del MLP competía con el del DeBERTa | Todos los epochs valían exactamente 0.5836 (sin mejora) |
| **Early stopping calibrado a log-loss** | Log-loss converge antes que Macro F1; modelo se detuvo en 35 trees en LGBM | Subrendimiento del modelo boosted |

---

## 6. Técnicas de Limpieza de Datos

### Texto (`statement`)
- Lowercase de todo el texto.
- Eliminación de HTML, URLs y espacios múltiples.
- Preservación de la puntuación significativa.
- Stemming con Porter Stemmer (para TF-IDF en modelos lineales; desactivado para embeddings).
- Tokenización preservando contracciones y entidades.

### Variables categóricas
- **Normalización**: lowercase, trim de espacios, colapso de separadores (pipes, slashes, semicolons → comas).
- **Imputación de nulos**: `unknown` como categoría explícita (no drop, para conservar trazabilidad).
- **Agrupación de valores raros**: umbral configurable (ej. < 5 ocurrencias para `speaker`, < 10 para `subject`) → categoría `other`.
- **Speaker**: normalización del nombre, extracción de prefijos de título (`dr`, `sen`, `gov`, `rep`, `mr`).
- **Party**: normalización a formas canónicas (`democrat`, `republican`, `none`, etc.).
- **State**: normalización de nombres de estado, abreviaciones a forma completa, agrupación por región de EE.UU.
- **Subject**: separación de temas múltiples, extracción del tema primario.

### `id`
- Mantenido solo para trazabilidad; excluido de todas las features del modelo.
- Conversión a bucket hash MD5 (`id_bucket`) para splits reproducibles.

---

## 7. Técnicas contra el Sobreajuste

| Técnica | Descripción | Aplicada en |
|---------|-------------|-------------|
| **Stratified K-Fold (5 folds)** | División estratificada respetando el ratio de clases | Todos los modelos |
| **Holdout 20% fijo** | Evaluación final solo al terminar; nunca tocado durante desarrollo | Todos los modelos |
| **True-rate features computadas dentro del fold** | Las tasas de veracidad por speaker/subject/party se computan solo sobre el split de entrenamiento de cada fold | LR, RFC, LGBM, CatBoost, Stacking |
| **Nested CV (RandomizedSearchCV)** | HP search interna en 3 folds dentro de cada fold exterior; evita leakage de parámetros | LR, RFC, LGBM, CatBoost |
| **Regularización L2 (LR)** | C=0.1–1.0; poda pesos irrelevantes del TF-IDF | LR |
| **L1 Penalty (LR)** | Feature selection automática en TF-IDF esparcido | LR (sweeps) |
| **`min_samples_leaf` (RFC)** | Valor óptimo = 6; evita hojas con un solo ejemplo | RFC |
| **`max_features=0.5` (RFC)** | Subsampling de columnas por split; aumenta diversidad | RFC |
| **`num_leaves=31` (LGBM)** | Controla la complejidad de los árboles leaf-wise | LGBM |
| **`min_child_samples=48` (LGBM)** | Requiere mínimo 48 muestras por hoja | LGBM |
| **`depth=4` (CatBoost)** | Árboles simétricos de profundidad 4 = 16 patrones de hoja máximo | CatBoost |
| **`l2_leaf_reg=5` (CatBoost)** | Penalización L2 sobre pesos de hojas | CatBoost |
| **Ordered Boosting (CatBoost)** | Target statistics computadas con permutación ordenada, eliminando leakage intra-fold | CatBoost |
| **Drop `fe_speaker_true_rate`** | Elimina la feature dominante que memorizaba identidades de entrenamiento | LGBM Opt B, CatBoost Opt B |
| **Meta-LR C=0.1** | Regularización del meta-clasificador en stacking | Stacking |
| **Checkpoint selection (transformers)** | Solo se guarda el checkpoint de mejor `val_macro_f1` por época | Transformers |
| **LLRD (Layer-wise LR Decay)** | Tasa de aprendizaje decrece en las capas inferiores del transformer | Transformers |
| **`CLS dropout = 0.3`** | Dropout en la cabeza de clasificación del transformer | Transformers |
| **`class_weight` / `auto_class_weights='Balanced'`** | Upweighting de la clase minoritaria (verdadero: ×1.42) | Todos los modelos |
| **OOF Threshold Tuning** | Grid search [0.20, 0.76] sobre predicciones OOF para maximizar Macro F1 | Todos los modelos desde LR |

---

## 8. Feature Engineering

### Features de texto (`statement_ds.py`)

| Feature | Descripción |
|---------|-------------|
| TF-IDF (unigrams/bigrams) | Vectorización con `max_features` 5,000–15,000 |
| Sentence embeddings | `all-MiniLM-L6-v2` (384-dim) o `all-mpnet-base-v2` (768-dim) |
| `statement_original_char_len` | Longitud en caracteres |
| `statement_original_word_count` | Conteo de palabras |
| `statement_upper_ratio` | Ratio de mayúsculas |
| `statement_clean_digit_ratio` | Densidad de dígitos |
| `statement_clean_avg_token_freq` | Frecuencia media de tokens |
| `statement_clean_spelling_err_count` | Conteo de errores ortográficos |
| NER counts | PERSON, ORG, GPE, DATE, NUM, OTHER (spaCy) |
| `fe_negation_count` | Conteo de palabras de negación |
| `fe_hedge_count` | Palabras de evasión/duda |
| `fe_absolutist_count` | Lenguaje absolutista (siempre, nunca, el mayor...) |
| `fe_numeral_count` | Números y estadísticas específicas |
| `fe_readability` | Score de legibilidad (Flesch–Kincaid) |
| `fe_sentiment_polarity` | Polaridad de sentimiento (TextBlob) |
| `fe_sentiment_subjectivity` | Subjetividad de sentimiento |

### Features de subject (`subject.py`)

| Feature | Descripción |
|---------|-------------|
| `subject_primary` | Primer tema extraído |
| `subject_primary_grouped` | Con agrupación de raros → `other` |
| `subject_frequency` | Frecuencia en training set |
| `subject_topic_count` | Número de temas en la fila |
| `subject_has_multiple_topics` | Flag binario |
| `subject_is_rare` | Flag de tema poco frecuente |
| `subject_primary_true_rate` | Tasa de veracidad por tema (computada OOF) |

### Features de speaker (`speaker.py`)

| Feature | Descripción |
|---------|-------------|
| `speaker_grouped` | Con agrupación de raros |
| `speaker_frequency` / `speaker_frequency_pct` | Frecuencia absoluta y relativa |
| `speaker_char_len` | Longitud del nombre |
| `speaker_is_rare` | Flag de hablante infrecuente |
| `speaker_has_title` | Flag de prefijo de título |
| `speaker_primary_true_rate` | Tasa de veracidad por hablante (OOF) |

### Features de speaker_job, party, state

Análogamente se generan frecuencias, flags de rareza, longitudes y tasas de veracidad OOF para `speaker_job`, `party_affiliation` y `state_info`. Flags adicionales: `is_major_party`, `is_institutional`, `is_us_state`, `us_region`.

### Features de interacción (`feature_engineering.py`)

| Feature | Descripción |
|---------|-------------|
| `fe_speaker_subject` | Clave compuesta speaker × subject |
| `fe_speaker_party` | Clave compuesta speaker × party |
| `fe_subject_party` | Clave compuesta subject × party |
| `fe_speaker_job_subject` | Clave compuesta speaker_job × subject |
| `fe_state_party` | Clave compuesta state × party |
| `fe_speaker_len_bucket` | Speaker × bucket de longitud del statement |
| `fe_speaker_avg_statement_len` | Longitud media de statements por speaker (no-leaky) |
| `fe_speaker_avg_punctuation` | Puntuación media por speaker |
| `fe_speaker_avg_number_ratio` | Ratio de números medio por speaker |
| `fe_subject_avg_statement_len` | Longitud media por tema |
| `fe_speaker_true_rate` | Tasa de veracidad histórica del hablante (OOF) |
| `fe_subject_true_rate` | Tasa de veracidad histórica por tema (OOF) |
| `fe_party_true_rate` | Tasa de veracidad histórica por partido (OOF) |
| `fe_speaker_job_true_rate` | Tasa de veracidad histórica por cargo (OOF) |

Las true-rate features son las más predictivas del dataset (mayor ganancia en árboles) y deben computarse **estrictamente dentro de cada fold** para evitar leakage.

---

## 9. Recorrido de Experimentos

### Experimento 1 — Logistic Regression (`lr.py`)

**Objetivo:** Establecer una baseline lineal sólida usando el mejor preprocesado encontrado por sweeps individuales.

**Configuración:**
- TF-IDF unigrams (5,000 features, min_df=2, max_df=0.9), Porter stemmer.
- Features léxicas: negation, hedge, absolutist, numeral, readability, sentiment.
- `class_weight={0: 1.42, 1: 0.77}`, C=1.0, L2, liblinear.
- 5-fold stratificado + holdout 20%.

**Iteraciones realizadas:**

| Iteración | Cambio | Macro F1 holdout | ROC-AUC |
|-----------|--------|-----------------|---------|
| Baseline | TF-IDF + LR | 0.604 | 0.654 |
| + Threshold tuning OOF | Umbral 0.46 | 0.596 | 0.653 |
| + Bigrams | max_features=10,000 | 0.607 | 0.653 |
| + Embeddings + True-rate | MiniLM + speaker/subject/party OOF rates | 0.603 | **0.665** |
| + Nested CV | RandomizedSearchCV por fold | **0.611** | 0.660 |
| + Isotonic calibration | CalibratedClassifierCV | 0.555 ↓ | 0.660 |

**Conclusiones clave:**
- El modelo lineal tiene un techo real (~0.611): no puede aprender interacciones entre features.
- La calibración isotónica empeoró significativamente (clase 0 recall colapsó a 0.24) por interacción con el threshold tuner.
- El ROC-AUC más alto (0.665) fue con embeddings + true-rates; el mejor Macro F1 fue con nested CV.

---

### Experimento 2 — Random Forest (`rfc.py`)

**Objetivo:** Establecer baseline de árbol con feature matrix completa; detectar bugs de preprocesado.

**Configuración inicial (detectó 2 bugs críticos):**
- Bug A: columna `statement` cruda pasaba por OrdinalEncoder → pseudo row-ID.
- Bug B: stopwords (`the`, `in`, `of`) dominando el vocabulario TF-IDF por `max_df=0.9` demasiado alto.

**Iteraciones (5 steps):**

| Step | Cambio | Macro F1 | ROC-AUC | Class 0 recall |
|------|--------|----------|---------|----------------|
| Initial (con bugs) | — | 0.5516 | 0.6598 | 0.24 |
| Step 1 | Fix bugs (drop `statement`, stopwords max_df=0.7) | 0.5606 | 0.6388 | 0.26 |
| Step 2 | OOF threshold tuning → threshold=0.58 | 0.5952 | 0.6388 | 0.46 |
| Step 3 | HP tuning: `min_samples_leaf=6`, `max_features=sqrt` | 0.5942 | 0.6481 | 0.50 |
| Step 4 | TF-IDF → embeddings (MiniLM 384-dim), `max_features=0.5` | 0.6080 | 0.6606 | 0.52 |
| **Step 5** | **NER features + `fe_speaker_job_true_rate`** | **0.6209** | **0.6674** | **0.60** |

**Conclusiones clave:**
- `fe_speaker_true_rate` es el feature de mayor ganancia (0.0827 en importancia); `fe_speaker_job_true_rate` debutó en el puesto 2.
- La sustitución de TF-IDF → embeddings fue el salto individual más grande (+0.014 F1).
- `min_samples_leaf=6` resultó ser la regularización óptima.

---

### Experimento 3 — LightGBM (`lgbm.py`)

**Objetivo:** Probar gradient boosting leaf-wise; esperaba superar RFC con datos mixtos texto+tabular.

**Configuración base:** misma feature matrix que RFC Step 5 (embeddings + NER + true-rates). Nested CV con 20 iteraciones.

**Resultado inesperado:** Macro F1 CV = 0.5934, holdout = 0.6062 — comparable o inferior al RFC en algunos aspectos.

**Causa raíz descubierta:** `fe_speaker_true_rate` monopolizaba el 6.7× la ganancia del siguiente feature. El HP search convergió siempre a configuraciones conservadoras (num_leaves=31, min_child_samples=48).

**Tres opciones experimentadas:**

| Opción | Cambio | Holdout Macro F1 | ROC-AUC |
|--------|--------|-----------------|---------|
| Initial | Baseline LGBM | 0.6062 | 0.6681 |
| A | Early stopping (log-loss) | 0.5985 ↓ | 0.6579 ↓ |
| **B** | **Drop `fe_speaker_true_rate`** | **0.6179 ↑** | **0.6790 ↑** |
| C | L1/L2 regularization (reg_alpha=0.5) | 0.6052 ↓ | 0.6624 ↓ |

**Conclusión clave:** Eliminar el feature dominante mejoró la generalización (+0.012 holdout F1). El CV se mantuvo igual (+0.001), confirmando que `fe_speaker_true_rate` memorizaba identidades del training set, no generalizaba. Early stopping calibrado a log-loss paró demasiado pronto (solo 35 árboles en algún fold).

---

### Experimento 4 — CatBoost (`cat.py`)

**Objetivo:** Aprovechar el ordered boosting de CatBoost para reducir la dominancia de un solo feature sin eliminar la señal.

**Ventajas de CatBoost vs LGBM:**
- **Ordered boosting**: las estadísticas de target se computan solo sobre muestras anteriores en una permutación aleatoria → sin leakage interno.
- **Árboles simétricos**: every nodo a igual profundidad usa la misma condición de split → regularización implícita fuerte.
- **Categorical handling nativo**: ordinal encoding + target statistics internas por categoría.

**Configuración:** HP search con 6 parámetros (iterations, learning_rate, depth, l2_leaf_reg, border_count, bagging_temperature). Unanimidad en 3 HPs en CatBoost+OptB (iterations=300, depth=4, border_count=32).

| Variante | Holdout Macro F1 | ROC-AUC | Threshold |
|----------|-----------------|---------|-----------|
| Initial | 0.6184 | 0.6653 | **0.48** (primero bajo 0.5) |
| **+ Option B (drop speaker_true_rate)** | **0.6294** | 0.6740 | **0.46** |

**Conclusiones:**
- CatBoost+OptB es el mejor modelo individual: ratio de dominancia bajó de 8.76× a 3.49×.
- El threshold 0.48 es la primera vez en el proyecto que cae bajo 0.5, evidenciando mejor calibración probabilística.
- `fe_speaker_job_true_rate` saltó al puesto 1 con 14.86 (3.49× por encima del segundo: `fe_subject_true_rate`).

---

### Experimento 5 — Stacking (`stacking.py`)

**Objetivo:** Combinar las OOF probabilities de 4 modelos base con un meta-LogisticRegression.

**Diseño:**
- 4 modelos base: LR (lbfgs, C=1.0), RFC (300 trees, max_features=0.3), LGBM-OptB (num_leaves=31→63), CatBoost-OptB.
- Todos con `drop_speaker_true_rate=True` y misma feature matrix.
- Meta-LR: `C=0.1`, sin class_weight.
- OOF sin HP search por base model (costosa combinación).

**Run 1 vs Run 2 (LGBM num_leaves=31 → 63):**

| Run | Holdout Macro F1 | ROC-AUC | Threshold |
|-----|-----------------|---------|-----------|
| Stacking Run 1 (nl=31) | 0.6303 | 0.6830 | 0.62 |
| **Stacking Run 2 (nl=63)** | **0.6323** | **0.6835** | 0.62 |

**Hallazgos clave:**
- **ROC-AUC = 0.6835**: nuevo récord del proyecto, superando LGBM-OptB (0.6790).
- **Threshold = 0.62**: el más alto del proyecto. Los 4 base models con class_weight empujan las probabilidades hacia clase 1; el meta-LR hereda ese sesgo sin corregirlo.
- **RFC tiene el coeficiente más alto** (1.64 → 1.51 en Run 2), a pesar de un ROC-AUC individual menor. Sus errores son los más decorrelacionados de los boosted models.
- LGBM subió de coeficiente (0.57 → 0.70) al aumentar num_leaves, confirmando la hipótesis.

---

### Experimento 6 — Transformers Fine-Tuning (`transformer.py`)

**Objetivo:** Superar el techo del stacking con representaciones semánticas profundas de DeBERTa.

**Modelos evaluados:**
- DeBERTa-v3-small (86M params): ~3–4 GB VRAM, fit en RTX 5070.
- DeBERTa-v3-base (184M params): ~7–8 GB VRAM.

**Hallazgo intermedio:** Antes del fine-tuning de transformers, el upgrade de embeddings a `all-mpnet-base-v2` (768-dim) mejoró el stacking de 0.6323 a **0.6428** — el upgrade de embeddings fue el mayor salto individual de toda la fase de stacking.

**Runs de fine-tuning (text-only, DeBERTa-v3-small):**

| Run | Config | Holdout Macro F1 | ROC-AUC |
|-----|--------|-----------------|---------|
| Run 1 | 3 epochs, no freeze | 0.6128 | 0.6700 |
| Mejor (Run 7) | freeze epoch 1 + LLRD + cls_dropout=0.3 | 0.6205 | 0.6700 |

**Análisis:** El transformer text-only no supera el stacking con mpnet (0.6428). Causas:
1. Single split: solo ve ~72% de los datos en training.
2. El modelo aprende señal de texto pero no tiene acceso a true-rate features.

---

### Experimento 7 — Hybrid Transformers (`7.HybridTransformers-Journey.md`)

**Objetivo:** Combinar DeBERTa con metadata para superar el ceiling del stacking (0.6428).

#### Sub-experimentos realizados:

**Run 1 — Hybrid MLP (paralelo):**
- DeBERTa CLS (768-dim) + MetaMLP(4→32→16) → Linear(784→2).
- Resultado: **0.6133** — peor que text-only.
- Causa: plateau en val macro_f1=0.5836 en todos los epochs. El gradiente del MLP interfería con el text path.

**Exp 1 — Text Formatting (prepend metadata):**
- Input: `"speaker: X | party: Y | subject: Z | {statement}"`
- Resultado: **0.6254** F1, **0.6885** ROC-AUC.
- Class 0 recall mejoró de 0.55 → 0.65.
- Limitación: freeze epoch 1 "desperdiciado"; 3 epochs insuficientes.

**Exp 1b — Text Format, no freeze, 5 epochs:**
- Resultado: **0.6392** F1, 0.6975 ROC-AUC.
- Mejor epoch = 3; epochs 4–5 muestran overfitting.

**Exp 2 — K-Fold × 5:**
- Resultado: **0.6393** F1, 0.6954 ROC-AUC.
- Tied con 1b; siempre mejor en epoch 2 (usar EPOCHS=2 en el futuro).

**Exp 3 — Late Fusion (transformer OOF como 5º base model en stacking):**
- Resultado: **0.6435** F1, **0.7089** ROC-AUC.
- **Primer ROC-AUC > 0.70 del proyecto.**
- Transformer coeficiente = 1.68 (muy valorado por el meta-LR).

**Exp 4a — DeBERTa-v3-base k-fold (standalone):**
- Resultado: **0.6397** F1, 0.7004 ROC-AUC.
- Marginal gain sobre small.

**Exp 4b — DeBERTa-v3-base + Late Fusion:**
- Resultado: **0.6472** F1, **0.7109** ROC-AUC.
- **NUEVO RÉCORD DEL PROYECTO.**
- Transformer coeficiente = 1.97 (up from 1.68).

**Exp 5 — LoRA Fine-Tuning (7B LLM):** planificado con `mistralai/Mistral-7B-v0.1` y cuantización 4-bit NF4 + LoRA (r=8, alpha=16). Viabilidad confirmada en RTX 5070 (~8–9 GB VRAM).

---

## 10. Resumen de Resultados — Todos los Modelos

| Modelo | Holdout Macro F1 | Holdout ROC-AUC | Threshold |
|--------|-----------------|----------------|-----------|
| LR baseline (TF-IDF) | 0.604 | 0.654 | 0.50 |
| LR + nested CV + embeddings | 0.611 | 0.660 | — |
| RFC Step 5 (embeddings + NER + job_true_rate) | 0.621 | 0.667 | 0.58 |
| LGBM Option B (drop speaker_true_rate) | 0.618 | **0.679** | 0.52 |
| CatBoost + Option B | 0.629 | 0.674 | 0.46 |
| Stacking Run 2 (mpnet 768-dim) | 0.643 | 0.700 | 0.62 |
| Transformer Exp 3 — Late Fusion DeBERTa-small | 0.644 | 0.709 | — |
| **Transformer Exp 4b — DeBERTa-base + Late Fusion** | **0.647** | **0.711** | — |

---

## 11. Lecciones Aprendidas

1. **La señal de texto es necesaria pero no suficiente.** El feature más predictivo en todos los modelos fue la tasa de veracidad histórica del hablante (`fe_speaker_true_rate`), no los embeddings — pero esa feature también era la que más sobreajustaba.

2. **Eliminar el feature dominante mejoró la generalización.** Contra la intuición, quitar `fe_speaker_true_rate` mejoró el holdout en LGBM (+0.012) y fue aditivo con CatBoost. El feature memorizaba identidades del training set.

3. **Los embeddings densos superan al TF-IDF disperso** para los modelos de árbol. La sustitución de TF-IDF → MiniLM fue la mayor mejora individual en RFC (+0.014 F1). El upgrade a mpnet (768-dim) fue el mayor salto en stacking.

4. **El stacking explota complementariedad real.** RFC (bagging, bootstrap) aporta errores decorrelacionados de los boosted models — coeficiente más alto (1.51–1.64) a pesar de ROC-AUC individual menor que LGBM.

5. **El texto formateado supera al MLP híbrido.** Inyectar metadata como tokens de texto (speaker, party, subject como prefijo) es más efectivo que un MLP paralelo, que genera interferencia de gradientes.

6. **Los transformers ganan más en late fusion que standalone.** DeBERTa-base solo: 0.640; en late fusion con el stacker: 0.647. El transformer aporta señal de texto puro que los árboles no tienen.

7. **La calibración del threshold es crítica.** Con desbalance de clases y class_weight, el threshold óptimo raramente es 0.5. Los modelos con mejor calibración (CatBoost, 0.46) o bien calibrados externamente (threshold tuning sobre OOF) producen mejores Macro F1.
