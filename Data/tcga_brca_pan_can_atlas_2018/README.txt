TCGA-BRCA data for external validation (same source as tcga_external_validation.py).

Study: brca_tcga_pan_can_atlas_2018 (TCGA Pan-Cancer Atlas 2018) via cBioPortal REST API.
Endpoint (patient clinical, JSON): https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018/clinical-data?clinicalDataType=PATIENT&projection=DETAILED

Files:
- tcga_brca_clinical_raw.csv — wide table of all patient-level clinical attributes returned by the API (one row per patient).
- tcga_brca_harmonised.csv — subset mapped to METABRIC RSF features (age_at_diagnosis, lymph_nodes_examined_positive, subtype) plus os_months and event used in the pilot.


