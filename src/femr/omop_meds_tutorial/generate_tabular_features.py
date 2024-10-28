"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""

import shutil
import meds_reader
import pandas as pd
import femr.featurizers
import pickle
from pathlib import Path
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser

def main():

    args = create_omop_meds_tutorial_arg_parser().parse_args()

    features_path = Path(args.pretraining_data) / "features"
    if features_path.exists():
        shutil.rmtree(str(features_path))
    features_path.mkdir(exist_ok=False)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=32) as database:
        for label_name in LABEL_NAMES:
            labels = pd.read_parquet(
                features_path.parent / "labels"  / (label_name + '.parquet')
            )
            featurizer = femr.featurizers.FeaturizerList([
                femr.featurizers.AgeFeaturizer(is_normalize=True),
                femr.featurizers.CountFeaturizer(),
            ])

            print("Preprocessing")

            featurizer.preprocess_featurizers(database, labels)

            print("Done preprossing, about to featurize")

            with open(features_path / (label_name + '_featurizer.pkl'), 'wb') as f:
                pickle.dump(featurizer, f)

            features = featurizer.featurize(database, labels)

            print("Done featurizing")

            with open(features_path / (label_name + '.pkl'), 'wb') as f:
                pickle.dump(features, f)

if __name__ == "__main__":
    main()