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


# "History of" knee arthroplasty SNOMED codes. These indicate that a TKR
# *happened at some point* but the code itself is retrospective documentation,
# so the procedure may have occurred years before the code is recorded.
# Therefore these are used for COHORT EXCLUSION (any subject with one of these
# before first KOA is dropped) but NOT for outcome detection — we don't want
# a retrospective "history of TKR" coded after first KOA to count as a
# positive outcome when the actual procedure could predate first KOA entirely.
HISTORY_OF_TKR_CODES: frozenset[str] = frozenset(
    {
        "SNOMED/1078631000119102",  # History of right total knee replacement
        "SNOMED/1078661000119105",  # History of left total knee replacement
        "SNOMED/1078691000119103",  # History of bilateral total knee replacement
        "SNOMED/1078641000119106",  # History of arthroplasty of right knee
        "SNOMED/1078671000119104",  # History of arthroplasty of left knee
        "SNOMED/1078701000119103",  # History of bilateral knee arthroplasty
        "SNOMED/1211000119105",     # History of total knee arthroplasty
        "SNOMED/674591000119103",   # History of revision of left total knee arthroplasty
        "SNOMED/674601000119105",   # History of revision of right total knee arthroplasty
        "SNOMED/16087931000119100", # History of revision of bilateral total knee joints
        "SNOMED/10997641000119102", # History of right prosthetic knee joint removal due to infection
        "SNOMED/10997811000119100", # History of left prosthetic knee joint removal due to infection
    }
)


# Partial / unicompartmental knee replacement (PKR / UKA).
# Patients with any of these codes at or before their first KOA event are
# excluded from the cohort, since their knee disease history is fundamentally
# different from a treatment-naive KOA patient.
PKR_CPT4_CODES: frozenset[str] = frozenset(
    {
        "CPT4/27446",  # Arthroplasty, knee, condyle and plateau; medial OR lateral compartment (UKA)
    }
)

PKR_SNOMED_CODES: frozenset[str] = frozenset(
    {
        "SNOMED/313064000",          # Unicompartmental knee joint prosthesis
        "SNOMED/1231398006",         # Unicompartmental knee arthroplasty using robotic assistance
        "SNOMED/1078651000119108",   # History of prosthetic unicompartmental arthroplasty of right knee
        "SNOMED/1078681000119101",   # History of prosthetic unicompartmental arthroplasty of left knee
        "SNOMED/1078711000119100",   # History of bilateral knee prosthetic unicompartmental arthroplasty
    }
)

PKR_CODES: frozenset[str] = PKR_CPT4_CODES | PKR_SNOMED_CODES


# Pre-KOA surgical-workup codes. Patients with any of these in their record
# strictly before the first KOA event are excluded as not treatment-naive —
# they entered the EHR already in a pre-arthroplasty workup, and their first
# KOA diagnosis is a late catch-up code rather than a true index event.
PRE_KOA_EXCLUSION_CODES: frozenset[str] = frozenset(
    {
        "CPT4/77073",  # Bone length studies (orthoroentgenogram / scanogram) —
                       # standard pre-arthroplasty alignment / leg-length study;
                       # 4× relative TKR risk vs cohort baseline.
    }
)


class TKRSinceKOALabeler(Labeler):
    """First-TKR-after-first-KOA, within a finite time horizon.

    One label per subject (or none, if the subject is excluded or censored).

    Cohort rules:
      * Subject must have a first KOA event.
      * Subject must NOT have any PKR, TKR, or "history of TKR" code strictly
        before the first KOA event (treats any prior knee arthroplasty as an
        exclusion criterion — most of these are contralateral-knee surgeries,
        same-encounter coding artifacts, or retrospective documentation of
        TKRs that happened outside the observation window).
      * Subject must NOT have any `pre_koa_exclusion_codes` event strictly
        before the first KOA event (e.g., bone-length studies — a pre-
        arthroplasty workup marker that creates "workup precedes diagnosis"
        leakage).
      * The first TKR considered as an outcome must be a real TKR procedure
        code in `tkr_codes` (NOT a `history_of_tkr_codes` entry, because
        history-of codes are retrospective and could refer to surgeries
        outside the prediction window) and strictly more than `tkr_washout`
        after the first KOA event.
      * Subject is censored (no label emitted) if their timeline ends before
        `first_koa_time + prediction_time_offset + time_horizon` without a
        qualifying TKR.
    """

    def __init__(
        self,
        koa_codes: Optional[Set[str]] = None,
        tkr_codes: Optional[Set[str]] = None,
        pkr_codes: Optional[Set[str]] = None,
        history_of_tkr_codes: Optional[Set[str]] = None,
        pre_koa_exclusion_codes: Optional[Set[str]] = None,
        time_horizon: datetime.timedelta = datetime.timedelta(days=365 * 5),
        prediction_time_offset: datetime.timedelta = datetime.timedelta(0),
        tkr_washout: datetime.timedelta = datetime.timedelta(days=60),
    ):
        self.koa_codes: Set[str] = set(koa_codes) if koa_codes is not None else set(KOA_CODES)
        self.tkr_codes: Set[str] = set(tkr_codes) if tkr_codes is not None else set(TKR_CODES)
        self.pkr_codes: Set[str] = set(pkr_codes) if pkr_codes is not None else set(PKR_CODES)
        self.history_of_tkr_codes: Set[str] = (
            set(history_of_tkr_codes)
            if history_of_tkr_codes is not None
            else set(HISTORY_OF_TKR_CODES)
        )
        self.pre_koa_exclusion_codes: Set[str] = (
            set(pre_koa_exclusion_codes)
            if pre_koa_exclusion_codes is not None
            else set(PRE_KOA_EXCLUSION_CODES)
        )
        self.time_horizon = time_horizon
        self.prediction_time_offset = prediction_time_offset
        self.tkr_washout = tkr_washout

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        if not subject.events:
            return []

        first_koa_time: Optional[datetime.datetime] = None
        first_tkr_after_koa_time: Optional[datetime.datetime] = None

        for event in subject.events:
            if event.time is None:
                continue
            if first_koa_time is None:
                # Pre-KOA window: any prior knee-arthroplasty code or surgical
                # workup marker disqualifies the subject. History-of-TKR
                # codes are included here (used only for exclusion, not for
                # outcome detection) because they indicate a prior TKR even
                # if the actual procedure event isn't in this dataset.
                if (
                    event.code in self.pkr_codes
                    or event.code in self.tkr_codes
                    or event.code in self.history_of_tkr_codes
                    or event.code in self.pre_koa_exclusion_codes
                ):
                    return []
                if event.code in self.koa_codes:
                    first_koa_time = event.time
                continue
            # first_koa_time is set; look for the first TKR strictly after washout.
            if event.code in self.tkr_codes and event.time > first_koa_time + self.tkr_washout:
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
