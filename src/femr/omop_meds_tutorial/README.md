> [!CAUTION]
> This tutorial is still a work in progress, use at your own risk!

FEMR Columbia MEDS Tutorial
======================

This tutorial illustrates how to use the MOTOR foundation model with the FEMR EHR modeling library on MEDS formatted Columbia data.

This involves multiple steps, including training baselines, training a MOTOR foundation model and applying that MOTOR foundation model.

Step 1. Converting into meds_reader
------------------------

The FEMR library uses the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. This requires a second preprocessing step

```bash
pip install meds_reader==0.0.6
```

```bash
export OMOP_MEDS = ""
export OMOP_MEDS_READER = ""
export PRETRAINING_DATA = ""
export ATHENA_DATA = ""

```

```bash
meds_reader_convert $OMOP_MEDS $OMOP_MEDS_READER --num_threads 16
```

Step 2. Downloading Athena
-------------------------

FEMR uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.

You can create an account for free.

Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

Step 3. Installing FEMR
------------------------

Finally, you need to install FEMR.

This code currently uses a branch of FEMR, mimic_tutorial, in order to function. However, soon these changes will be merged upstream into femr 2.4

```bash
git clone https://github.com/ChaoPang/femr.git
cd femr
git checkout omop_meds_v3_tutorial
pip install -e .
pip install xformers
```

Make sure to also install the correct gpu enabled version of PyTorch


Step 4. Preparing For Pretraining
------------------------
We have a single script, prepare_motor that generates these splits and then training things like tokenizers to prepare for pretraining

```bash
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER
```

Step 5. Generate Labels
------------------------
We use FEMR's built-in labeling tools to define two prediction tasks: long length of stay (7 days or more) and inpatient mortality. Both predictions are made 48 hours after admission.

```bash
python -u -m femr.omop_meds_tutorial.generate_labels \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```


Step 6. Generate Tabular Features
------------------------
We use FEMR's built-in labeling tools to define two prediction tasks: long length of stay (7 days or more) and inpatient mortality. Both predictions are made 48 hours after admission.

```bash
python -u -m femr.omop_meds_tutorial.generate_tabular_features \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```

Step 7. Train Tabular Baselines
------------------------

We can then train baselines on those labels and tabular features. We train two baselines in particular, LightGBM and logistic regresison.
```bash
python -u -m femr.omop_meds_tutorial.train_baseline \
  --pretraining_data $PRETRAINING_DATA \ 
  --meds_reader $OMOP_MEDS_READER
```

Step 8. Pretrain MOTOR
------------------------
You could probably also train on smaller GPUs, even 16GB but that might require some hyperparameter tweaks.

```bash
python -u -m femr.omop_meds_tutorial.pretrain_motor \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```

When the model appears to have converged (after roughly 70,000 steps seems good enough for my experiments), copy the checkpoint to the `motor_model` directory.

```bash
cp -r tmp_trainer_1e-4/checkpoint-68000 $PRETRAINING_DATA/motor_model
```


Step 9. Generate MOTOR Embedddings
------------------------

We can then use MOTOR as an embedding model to obtain patient representations

```bash
python -u -m femr.omop_meds_tutorial.generate_motor_features \ 
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```

Step 10. Train Logistic Regression On MOTOR Embeddings
------------------------

Finally we can use MOTOR by training a linear head (aka a logistic regression model) on top of the frozen MOTOR embeddings to solve our prediction tasks.

```bash
python -u -m femr.omop_meds_tutorial.finetune_motor \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```