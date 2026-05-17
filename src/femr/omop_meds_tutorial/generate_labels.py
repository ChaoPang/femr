"""
One of FEMR's main features is utilities for helping write labeling functions.

The following are two simple labelers for inpatient mortality and long admission for MIMIC-IV.
"""

import femr.labelers
import meds_reader
import meds
import datetime
import shutil

from typing import List, Mapping
from pathlib import Path


LABEL_NAMES = [
    "death",
    "long_los",
]
# Truveta encodes encounters as paired (start, end) events with code
# "ENCOUNTER//<type>" and "ENCOUNTER_END//<type>", and event.end is always None.
ADMISSION_START_CODES = ["ENCOUNTER//Inpatient encounter"]
ADMISSION_END_CODES = ["ENCOUNTER_END//Inpatient encounter"]


def _admission_ranges(subject: meds_reader.Subject) -> List:
    """Pair admission starts with the next ENCOUNTER_END of the same kind via a sorted two-pointer scan."""
    starts = sorted(e.time for e in subject.events if e.code in ADMISSION_START_CODES and e.time is not None)
    ends = sorted(e.time for e in subject.events if e.code in ADMISSION_END_CODES and e.time is not None)
    ranges = []
    i = j = 0
    while i < len(starts) and j < len(ends):
        if ends[j] >= starts[i]:
            ranges.append((starts[i], ends[j]))
            i += 1
            j += 1
        else:
            j += 1
    return ranges


class OmopInpatientMortalityLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta):
        self.time_after_admission = time_after_admission

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        death_times = {e.time for e in subject.events if e.code == meds.death_code}

        if len(death_times) not in [0, 1]:
            print(f"Warning: found {len(death_times)} death events in subject: {subject.subject_id}")

        if len(death_times) == 1:
            death_time = list(death_times)[0]
        else:
            death_time = datetime.datetime(9999, 1, 1)  # Very far in the future

        labels = []
        for (admission_start, admission_end) in _admission_ranges(subject):
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue
            if prediction_time >= death_time:
                continue
            is_death = death_time < admission_end
            labels.append(
                meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=is_death))
        return labels


class OmopLongAdmissionLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta, admission_length: datetime.timedelta):
        self.time_after_admission = time_after_admission
        self.admission_length = admission_length

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        labels = []
        for (admission_start, admission_end) in _admission_ranges(subject):
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue
            is_long_admission = (admission_end - admission_start) > self.admission_length
            labels.append(meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time,
                                     boolean_value=is_long_admission))
        return labels


labelers: Mapping[str, femr.labelers.Labeler] = {
    'death': OmopInpatientMortalityLabeler(time_after_admission=datetime.timedelta(hours=48)),
    'long_los': OmopLongAdmissionLabeler(time_after_admission=datetime.timedelta(hours=48),
                                         admission_length=datetime.timedelta(days=7)),
}


def create_omop_meds_tutorial_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
    return parser


def main():
    args = create_omop_meds_tutorial_arg_parser().parse_args()
    
    labels_path = Path(args.pretraining_data) / "labels"
    labels_path.mkdir(parents=True, exist_ok=True)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in LABEL_NAMES:
            labeler = labelers[label_name]
            labels = labeler.apply(database)
            labels.to_parquet(str(labels_path / (label_name + '.parquet')))


if __name__ == "__main__":
    main()
