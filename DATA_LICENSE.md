# Data licenses and terms

Checked 2026-11-09. Raw data is never committed to this repository.

## C-NMC 2019 (training, validation, internal test)

- Source: The Cancer Imaging Archive (TCIA), collection C-NMC 2019. 118 subjects (69 ALL, 49 normal), 15,135 images.
- License: **CC BY 3.0**. Redistribution and derived works (including trained weights and example images) are allowed with attribution.
- Copies used here: Hugging Face mirrors `dwb2023/cnmc-leukemia-2019` (10,661 labeled training images) and
  `dwb2023/cnmc-leukemia-2019-test` (1,867 labeled preliminary-test images). The final challenge test set is not used.
- Required data citation:
  Mourya, S., Kant, S., Kumar, P., Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 (C-NMC 2019)
  (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r
- Recommended publication: Gupta, R., Gehlot, S., & Gupta, A. (2022). C-NMC: B-lineage acute lymphoblastic
  leukaemia: A blood cancer dataset. Medical Engineering & Physics, 103. https://doi.org/10.1016/j.medengphy.2022.103793

## ALL-IDB2 (external evaluation only)

- Source: University of Milan, https://scotti.di.unimi.it/all/ (access by signed application form).
- Terms (form ver.2): no redistribution, publication or dissemination in whole or in part without the maintainer's
  authorization; internal non-commercial research only; must be cited; not for diagnostic use.
- Consequences for this project: ALL-IDB images never appear in the repo, web app, Hugging Face, figures or the paper.
  Only aggregate metrics are reported. Never used for training or threshold selection.
- Citation: Labati, R. D., Piuri, V., & Scotti, F. (2011). ALL-IDB: The acute lymphoblastic leukemia image database
  for image processing. 18th IEEE International Conference on Image Processing (ICIP), 2045-2048.

## Model weights

Trained only on C-NMC (CC BY 3.0), so weights can be published on Hugging Face with the citation above in the model card.
