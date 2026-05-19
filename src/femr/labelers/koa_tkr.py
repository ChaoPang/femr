"""Labeler: predict the first total knee replacement (TKR) since first knee
osteoarthritis (KOA) diagnosis.

* Prediction time: time of the subject's first KOA event.
* Boolean label: True iff the subject has a TKR event strictly after the first
  KOA event, within `time_horizon` of the prediction time.
* Subjects with no KOA event: no label emitted.
* Subjects with KOA but whose timeline ends before
  `first_koa_time + time_horizon` without a TKR: censored (no label emitted) so
  we don't pretend "False" for someone we simply stopped observing.

KOA is defined via SNOMED + ICD10CM (M17.x). TKR is defined via SNOMED + CPT4,
matching the constraint that TKR must be a procedural code.
"""
from __future__ import annotations

import datetime
from typing import List, Optional, Set

import meds
import meds_reader

from .core import Labeler


KOA_SNOMED_CODES: frozenset[str] = frozenset(
    {
        "SNOMED/239873007",         # Osteoarthritis of knee
        "SNOMED/323301000119109",   # Osteoarthritis of left knee joint
        "SNOMED/323321000119100",   # Osteoarthritis of right knee joint
        "SNOMED/112981000119107",   # Bilateral osteoarthritis of knees
        "SNOMED/1074341000119106",  # Secondary osteoarthritis of bilateral knees
        "SNOMED/1074351000119108",  # Secondary osteoarthritis of left knee
        "SNOMED/1074361000119105",  # Secondary osteoarthritis of right knee
        "SNOMED/1287058006",        # Secondary osteoarthritis of knee joint
        "SNOMED/789001000",         # Osteoarthritis of knee due to and following trauma
    }
)

KOA_ICD10CM_CODES: frozenset[str] = frozenset(
    {
        "ICD10CM/M17.0", "ICD10CM/M17.1", "ICD10CM/M17.10",
        "ICD10CM/M17.11", "ICD10CM/M17.12", "ICD10CM/M17.2",
        "ICD10CM/M17.3", "ICD10CM/M17.30", "ICD10CM/M17.31",
        "ICD10CM/M17.32", "ICD10CM/M17.4", "ICD10CM/M17.5",
        "ICD10CM/M17.9",
    }
)

KOA_CODES: frozenset[str] = KOA_SNOMED_CODES | KOA_ICD10CM_CODES


TKR_CPT4_CODES: frozenset[str] = frozenset(
    {
        "CPT4/27445",  # Arthroplasty, knee, hinge prosthesis
        "CPT4/27447",  # Total knee arthroplasty (primary)
        "CPT4/27486",  # Revision TKA, 1 component
        "CPT4/27487",  # Revision TKA, femoral + entire tibial component
        "CPT4/27488",  # Removal of knee prosthesis
    }
)

TKR_SNOMED_CODES: frozenset[str] = frozenset(
    {
        "SNOMED/109228008", "SNOMED/1201749003", "SNOMED/1222567002",
        "SNOMED/1222568007", "SNOMED/1231396005", "SNOMED/1240411007",
        "SNOMED/1240412000", "SNOMED/16117008", "SNOMED/179344006",
        "SNOMED/179346008", "SNOMED/179351002", "SNOMED/19063003",
        "SNOMED/265170009", "SNOMED/265172001", "SNOMED/280462001",
        "SNOMED/307819006", "SNOMED/309852007", "SNOMED/313062001",
        "SNOMED/392237008", "SNOMED/392238003", "SNOMED/430698003",
        "SNOMED/444886001", "SNOMED/450814005", "SNOMED/609588000",
        "SNOMED/735262008", "SNOMED/911000119102",
    }
)

TKR_CODES: frozenset[str] = TKR_CPT4_CODES | TKR_SNOMED_CODES


class TKRSinceKOALabeler(Labeler):
    """First-TKR-after-first-KOA, within a finite time horizon.

    One label per subject (or none, if the subject has no KOA event or is censored).
    """

    def __init__(
        self,
        koa_codes: Optional[Set[str]] = None,
        tkr_codes: Optional[Set[str]] = None,
        time_horizon: datetime.timedelta = datetime.timedelta(days=365 * 5),
        prediction_time_offset: datetime.timedelta = datetime.timedelta(0),
    ):
        self.koa_codes: Set[str] = set(koa_codes) if koa_codes is not None else set(KOA_CODES)
        self.tkr_codes: Set[str] = set(tkr_codes) if tkr_codes is not None else set(TKR_CODES)
        self.time_horizon = time_horizon
        self.prediction_time_offset = prediction_time_offset

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        if not subject.events:
            return []

        first_koa_time: Optional[datetime.datetime] = None
        first_tkr_after_koa_time: Optional[datetime.datetime] = None

        for event in subject.events:
            if event.time is None:
                continue
            if first_koa_time is None:
                if event.code in self.koa_codes:
                    first_koa_time = event.time
                continue
            # first_koa_time is set; look for the first TKR strictly after it
            if event.code in self.tkr_codes and event.time > first_koa_time:
                first_tkr_after_koa_time = event.time
                break

        if first_koa_time is None:
            return []

        prediction_time = first_koa_time + self.prediction_time_offset
        end_of_window = prediction_time + self.time_horizon
        last_event_time = subject.events[-1].time

        if first_tkr_after_koa_time is not None and first_tkr_after_koa_time <= end_of_window:
            return [
                meds.Label(
                    subject_id=subject.subject_id,
                    prediction_time=prediction_time,
                    boolean_value=True,
                )
            ]

        # Censor subjects whose observation ends before we'd be able to call it "False"
        if last_event_time is None or last_event_time < end_of_window:
            return []

        return [
            meds.Label(
                subject_id=subject.subject_id,
                prediction_time=prediction_time,
                boolean_value=False,
            )
        ]
