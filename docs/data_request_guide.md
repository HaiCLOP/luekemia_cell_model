# Getting clinical data from a haematology lab

The models in this repo are limited by data from one lab (C-NMC) and by the lack of paired patient records (blood count
plus smear plus confirmed diagnosis). A partnership with a hospital haematology or haematopathology department is the
way past both. This guide covers who to approach, what to ask for, and what they will need from you.

## Who to approach

Start close to home and with places that see many leukemia cases. Find the head of **Pathology** or
**Haematopathology / Haematology** on each institution's official website (faculty pages list emails) rather than
third-party sites.

- **Assam Medical College and Hospital (AMCH), Dibrugarh**: Department of Pathology. Closest to you, a large teaching
  hospital.
- **Dr. B. Borooah Cancer Institute (BBCI), Guwahati**: a regional cancer centre (a Tata Memorial Centre unit); sees
  leukemia patients routinely.
- **AIIMS Guwahati** and **Gauhati Medical College and Hospital**: departments of Pathology and Haematology.
- Further afield, large haematopathology groups (Tata Memorial Centre, Mumbai; AIIMS New Delhi; PGIMER Chandigarh)
  publish on leukemia AI and may welcome a well-prepared collaborator.

A faculty member who agrees becomes the **principal investigator** on their side; as an independent researcher you work
under their project, which is how hospitals handle outside collaborators.

## What they will need from you

- **A one-page proposal**: the question (does a calibrated, refer-to-expert model generalise across labs?), what you
  have already built (link this repo and the model card, including the honest negative results), and what you need.
- **Ethics approval**: they will submit it to their **Institutional Ethics Committee (IEC)**. Offer to draft the
  protocol. Retrospective, fully de-identified data usually qualifies for an expedited review or a waiver of consent.
- **A data use agreement**: data stays de-identified, is used only for the stated research, is never redistributed,
  and is deleted or returned on request.
- **Credit**: offer co-authorship on any paper. This matters more to them than anything else.

## What to ask for (be specific)

Per patient, with a pseudonymous ID that links the items but cannot identify the person:

1. **Smear images**: single-cell crops or whole fields at 100x oil, with the microscope, camera and stain recorded.
2. **Final diagnosis** confirmed by flow cytometry and genetics (B-ALL, T-ALL, AML subtype, CLL, CML, or normal).
3. **Blood count** at the time of the smear (WBC, differential, haemoglobin, platelets).
4. **Flow cytometry** summary (markers) and key genetic results (for example BCR::ABL1).
5. **Hard negatives**: reactive lymphocytosis, hematogones, infections, so the model learns what is *not* leukemia.

Even 50 to 100 patients from one new hospital would give this project its first true external validation.

## Email template

> **Subject:** Research collaboration request: AI triage of leukemia cells (de-identified data)
>
> Dear Dr. [Name],
>
> I am Arnav Srivastava, an independent researcher in Dibrugarh (HaiCLOP Labs). I have built an open-source research
> model that classifies blood cells as acute lymphoblastic leukemia or normal, with calibrated confidence and a
> refer-to-expert option. On 26 held-out subjects it reached PR-AUC 0.93 while missing 2.8% of leukemic cells, but tests
> on smears from other hospitals show it does not yet generalise beyond the lab it was trained on. The code, model card
> and results are public: https://github.com/HaiCLOP/luekemia_cell_model
>
> I would like to ask whether your department would consider a research collaboration using de-identified, retrospective
> data: smear images with confirmed diagnoses and the matching blood counts. I would draft the protocol for your
> Institutional Ethics Committee, sign a data use agreement, and share all results and credit, including co-authorship.
> The work is for research only and would not be used for patient care.
>
> Could we schedule a short call at your convenience? I am happy to present the work in person.
>
> With regards,
> Arnav Srivastava
> HaiCLOP Labs, Dibrugarh, Assam
> [phone] · [email]

## Public datasets worth requesting in parallel

- **ALL-IDB** (University of Milan): requested; follow up once with a short, polite reminder.
- **Munich AML Cytomorphology (Matek et al. 2019)**: public on TCIA, CC BY 3.0; being used for the ALL vs AML work.
- **Bone Marrow Cytomorphology (MLL / Helmholtz, Matek et al. 2021)**: public on TCIA; bone marrow cells across many
  diagnoses.
