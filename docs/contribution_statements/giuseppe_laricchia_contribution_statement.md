# Individual contribution statement

**Name:** Giuseppe Laricchia  
**Team role:** Data Workflow

I owned the data pipeline epic (SCRUM-5): the EMNIST audit (SCRUM-10), reproducible preprocessing and splits (SCRUM-11), and the hand-filled Appendix A forms (SCRUM-33). I set up the EMNIST download with SHA-256 checksums, built the loader and audit that verify the label mapping and fix EMNIST's sideways storage, and created the first hand-labeled ground truth for the team forms.

I filtered EMNIST ByClass to the 36 characters the form allows (0–9, A–Z), which is why the model never outputs lowercase. I took a stratified 10% validation set with a fixed seed (6642) and kept the official test split untouched as the only source of characters for synthetic test logs. The main challenge was imbalance: digits are 64.6% of training images and the largest class has 15.6× the images of the rarest, while letters on the form are close to uniform. I addressed it with inverse-frequency class weights and a per-field digit/letter restriction. A leakage check also found two test images duplicated in training, which are excluded.

Claude assisted with code scaffolding and draft documentation, as disclosed in `docs/ai_assistance.md`. My follow-up work included checking the orientation and class-count evidence and correcting values in the team-form ground truth.
