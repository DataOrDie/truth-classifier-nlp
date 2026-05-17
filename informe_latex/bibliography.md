# Bibliography

Academic and official references for the algorithms, models, and techniques used in this project. Organised by training script, followed by a preprocessing section.

---

## `logisticRegression.py`

**Logistic Regression**
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- scikit-learn. Logistic Regression — official documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

**Regularisation (L1/L2/ElasticNet)**
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x

**Hyperparameter Search — `RandomizedSearchCV`**
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305. https://jmlr.org/papers/v13/bergstra12a.html
- scikit-learn. RandomizedSearchCV — official documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

**Stratified K-Fold Cross-Validation**
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, 2, 1137–1143.
- scikit-learn. Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html

**Evaluation — Macro F1, MCC, ROC-AUC**
- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010 

**scikit-learn (shared library)**
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

---

## `randomForestClassifier.py`

**Random Forest**
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

**Mean Decrease Impurity (MDI) Feature Importance**
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
<!-- - Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). Understanding variable importances in forests of randomized trees. *Advances in Neural Information Processing Systems*, 26. https://papers.nips.cc/paper_files/paper/2013/hash/e3796ae838835da0b6f6ea37bcf8bcb5-Abstract.html NOT-FOUND -->

**Class Weighting for Imbalanced Data**
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
- scikit-learn. class_weight parameter. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

*(See also: RandomizedSearchCV, Stratified K-Fold, Macro F1/MCC, scikit-learn — cited in `logisticRegression.py`)*

---

## `LGBMClassifier.py`

**LightGBM**
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- LightGBM. Official documentation. https://lightgbm.readthedocs.io/en/stable/

**Gradient Boosted Decision Trees (foundational)**
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

**Gain-Based Feature Importance**
- LightGBM. Feature importances. https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance

**Leaf-wise Tree Growth**
- LightGBM. Features — Leaf-wise (best-first) tree growth. https://lightgbm.readthedocs.io/en/stable/Features.html#leaf-wise-best-first-tree-growth

**L1/L2 Regularisation in GBDT**
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://arxiv.org/abs/1603.06212

*(See also: RandomizedSearchCV, Stratified K-Fold, Macro F1/MCC, scikit-learn — cited in `logisticRegression.py`)*

---

## `catboost.py`

**CatBoost**
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31. https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac1bef54c0f6d53b-Abstract.html
- Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: Gradient boosting with categorical features support. arXiv:1810.11363. https://arxiv.org/abs/1810.11363
- CatBoost. Official documentation. https://catboost.ai/docs/

**Ordered Boosting (target leakage prevention)**
- Prokhorenkova et al. (2018) — same paper as above; see Section 2 on ordered boosting.

**Ordinal Encoding for Categorical Features**
- scikit-learn. OrdinalEncoder — official documentation. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html

*(See also: Gradient Boosted Decision Trees (Friedman 2001), Stratified K-Fold, Macro F1/MCC, scikit-learn — cited in earlier sections)*

---

## `stacking.py`

**Stacked Generalisation**
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
- Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49–64. https://doi.org/10.1007/BF00117832

**XGBoost (base model in stacking ensemble)**
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794. https://arxiv.org/abs/1603.06212
- XGBoost. Official documentation. https://xgboost.readthedocs.io/en/stable/

**XGBoost `sample_weight` for class imbalance**
- XGBoost. Tips on dealing with imbalanced datasets. https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html

**Out-of-Fold (OOF) Predictions**
- Wolpert (1992) — ibid.
- Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys*, 4, 40–79. https://doi.org/10.1214/09-SS054

**Meta-Learner (Logistic Regression with L2)**
- See `logisticRegression.py` section above.

**Late Fusion / Ensemble Combination**
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.

*(See also: LightGBM, CatBoost, Random Forest — cited in earlier sections)*

---

## `transformer.py`

**DeBERTa-v3-small**
- He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. *International Conference on Learning Representations (ICLR) 2021*. https://arxiv.org/abs/2006.03654
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*. https://arxiv.org/abs/2111.09543

**BERT-style Pre-training (foundational)**
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*. https://arxiv.org/abs/1810.04805

**Transformer Architecture**
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

**AdamW Optimiser**
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. https://arxiv.org/abs/1711.05101

**Linear Warmup + Linear Decay Scheduler**
- Devlin et al. (2019) — ibid, Section 3.3.
- Hugging Face. `get_linear_schedule_with_warmup`. https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup

**Dropout Regularisation**
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

**Class-Weighted Cross-Entropy Loss**
- King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163. https://doi.org/10.1093/pan/9.2.137

**HuggingFace Transformers Library**
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., … Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020*. https://arxiv.org/abs/1910.03771

---

## `transformer_hybrid.py`

**Hybrid Text + Metadata Architecture**
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? *China National Conference on Chinese Computational Linguistics (CCL) 2019*. https://arxiv.org/abs/1905.05583

**Batch Normalisation**
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*. https://arxiv.org/abs/1502.03167

**GELU Activation Function**
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415. https://arxiv.org/abs/1606.08415

*(See also: DeBERTa-v3, AdamW, Linear Warmup, HuggingFace Transformers — cited in `transformer.py`)*

---

## `transformer_textformat.py`

**Metadata as Text Tokens (prompt / input formatting)**
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., … Amodei, D. (2020). Language models are few-shot learners. *NeurIPS 2020*. https://arxiv.org/abs/2005.14165
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. *EMNLP 2021*. https://arxiv.org/abs/2104.08691

**Layer-wise Learning Rate Decay (LLRD)**
- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL 2018*. https://arxiv.org/abs/1801.06146
- Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? https://arxiv.org/abs/1905.05583

**Early Stopping / Best Checkpoint Selection**
- Prechelt, L. (1998). Early stopping — but when? In *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, Vol. 1524 (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3

*(See also: DeBERTa-v3, AdamW, Dropout, HuggingFace Transformers — cited in `transformer.py`)*

---

## `transformer_kfold.py`

**Stratified K-Fold for Sequence Models**
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI 1995*, 2, 1137–1143.

**Ensemble Averaging**
- Dietterich, T. G. (2000). Ensemble methods in machine learning. *International Workshop on Multiple Classifier Systems*, 1857, 1–15. https://doi.org/10.1007/3-540-45014-9_1
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*. https://arxiv.org/abs/1612.01474

**Out-of-Fold Probabilities for Late Fusion**
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1

*(See also: DeBERTa-v3, LLRD, text format, AdamW — cited in earlier sections)*

---

## `transformer_kfold_base.py`

**DeBERTa-v3-base**
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*. https://arxiv.org/abs/2111.09543
- Microsoft. DeBERTa-v3-base on Hugging Face. https://huggingface.co/microsoft/deberta-v3-base

**ELECTRA-style Pre-training**
- Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. *ICLR 2020*. https://arxiv.org/abs/2003.10555

*(See also: K-Fold ensemble, OOF, LLRD, AdamW — cited in earlier sections)*

---

## `transformer_lora_kfold.py`

**LoRA — Low-Rank Adaptation**
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., … Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

**QLoRA — 4-bit NF4 Quantisation**
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*. https://arxiv.org/abs/2305.14314

**bitsandbytes Library**
- Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS 2022*. https://arxiv.org/abs/2208.07339
- bitsandbytes. Official documentation. https://huggingface.co/docs/bitsandbytes/index

**PEFT Library**
- Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). PEFT: State-of-the-art parameter-efficient fine-tuning methods. Hugging Face. https://github.com/huggingface/peft

**Mistral-7B**
- Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., … El Sayed, W. (2023). Mistral 7B. arXiv:2310.06825. https://arxiv.org/abs/2310.06825

**Grouped-Query Attention (Mistral architecture)**
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. *EMNLP 2023*. https://arxiv.org/abs/2305.13245

**Gradient Checkpointing**
- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training deep nets with sublinear memory cost. arXiv:1604.06174. https://arxiv.org/abs/1604.06174

**Gradient Accumulation**
- Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., … Auli, M. (2019). fairseq: A fast, extensible toolkit for sequence modeling. *NAACL-HLT 2019 (Demo)*. https://arxiv.org/abs/1904.01038

**Near-zero Initialisation of Classification Head**
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*. https://proceedings.mlr.press/v9/glorot10a.html

**Left-Padding for Causal Decoder Classification**
- Touvron, H., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288. https://arxiv.org/abs/2307.09288 *(Section 3.1 discusses decoder classification setup)*

*(See also: K-Fold ensemble, OOF, AdamW, Linear Warmup, HuggingFace Transformers — cited in earlier sections)*

---

## `transformer_threshold.py`

**Threshold Optimisation for Imbalanced Classification**
- Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. *Machine Learning*, 42(3), 203–231. https://doi.org/10.1023/A:1007601015854
- Sheng, V. S., & Ling, C. X. (2006). Thresholding for making classifiers cost-sensitive. *AAAI 2006*. https://ojs.aaai.org/index.php/AAAI/article/view/7693

**Precision-Recall Tradeoff**
- Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *ICML 2006*. https://doi.org/10.1145/1143844.1143874

*(See also: Macro F1, MCC, ROC-AUC — cited in `logisticRegression.py`)*

---

## Preprocessing — `statement_ds.py`

### Text Cleaning

**Unicode Normalisation**
- Unicode Consortium. (2023). *The Unicode Standard*, Version 15.0. https://unicode.org/versions/Unicode15.0.0/

**HTML and URL removal**
- Standard regular expression techniques; see NLTK documentation: https://www.nltk.org/

---

### Vectorisation

**TF-IDF (Term Frequency — Inverse Document Frequency)**
- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11–21. https://doi.org/10.1108/eb026526
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
- scikit-learn. TfidfVectorizer. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

**N-gram Models (bigram vectorisation)**
- Cavnar, W. B., & Trenkle, J. M. (1994). N-gram-based text categorization. *SDAIR-94*, 22, 161–175.
- scikit-learn. CountVectorizer with `ngram_range`. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

**Binary Bag-of-Words**
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/

---

### Sentence Embeddings

**Sentence-BERT / Sentence Transformers**
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *EMNLP-IJCNLP 2019*. https://arxiv.org/abs/1908.10084
- sentence-transformers. Official documentation. https://www.sbert.net/

**`all-MiniLM-L6-v2` model**
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS 2020*. https://arxiv.org/abs/2002.10957
- Hugging Face model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**`all-mpnet-base-v2` model**
- Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*. https://arxiv.org/abs/2004.09297
- Hugging Face model card: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

---

### Stemming & Lemmatisation

**Porter Stemmer**
- Porter, M. F. (1980). An algorithm for suffix stripping. *Program: Electronic Library and Information Systems*, 14(3), 130–137. https://doi.org/10.1108/eb046814

**Snowball Stemmer**
- Porter, M. F. (2001). Snowball: A language for stemming algorithms. https://snowballstem.org/

**WordNet Lemmatiser**
- Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39–41. https://doi.org/10.1145/219717.219748
- Fellbaum, C. (Ed.). (1998). *WordNet: An Electronic Lexical Database*. MIT Press.

---

### Stop Word Removal

**NLTK Stop Words**
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. https://www.nltk.org/book/
- NLTK. stopwords corpus. https://www.nltk.org/api/nltk.corpus.html

---

### Named Entity Recognition (NER)

**spaCy**
- Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength natural language processing in Python. Zenodo. https://doi.org/10.5281/zenodo.1212303
- spaCy. `en_core_web_sm` model documentation. https://spacy.io/models/en

**NER as a feature for fake news detection**
- Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). Automatic detection of fake news. *COLING 2018*. https://arxiv.org/abs/1708.07104

---

### Lexical Features

**Spelling Error Detection**
- NLTK. words corpus (English vocabulary reference). https://www.nltk.org/api/nltk.corpus.html

**Lexical Richness / Type-Token Ratio**
- Malvern, D., Richards, B., Chipere, N., & Durán, P. (2004). *Lexical Diversity and Language Development: Quantification and Assessment*. Palgrave Macmillan.

---

### Feature Scaling

**Z-score Standardisation / Min-Max Normalisation**
- scikit-learn. Preprocessing data. https://scikit-learn.org/stable/modules/preprocessing.html
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

---

## Datasets

**LIAR Dataset (basis of this competition)**
- Wang, W. Y. (2017). "Liar, liar pants on fire": A new benchmark dataset for fake news detection. *ACL 2017*. https://arxiv.org/abs/1705.00648

---

## General Evaluation References

**Imbalanced Classification**
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

**Fake News Detection Survey**
- Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys*, 53(5), 1–40. https://doi.org/10.1145/3395046

**Natural Language Processing (general reference)**
- Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed., draft). https://web.stanford.edu/~jurafsky/slp3/
