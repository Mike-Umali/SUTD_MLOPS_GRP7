"""
Evaluation test set — 20 hand-crafted queries covering all 7 expert domains.

Each TestCase contains:
  - expected_domains: ground truth for routing eval (must match EXPERT_PROFILES keys)
  - domain_for_retrieval: which ChromaDB collection to test retrieval against
  - relevant_subtopics: exact subtopic strings from dataset.csv metadata (hit = any match in top-5)
  - expected_keywords: keywords that should appear in the final advisory
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TestCase:
    id: str
    query: str
    expected_domains: List[str]
    domain_for_retrieval: str
    relevant_subtopics: List[str]
    expected_keywords: List[str]
    notes: str = ""


TEST_CASES: List[TestCase] = [

    # ── DRUG OFFENCES ─────────────────────────────────────────────────────────

    TestCase(
        id="TC-01",
        query="My client is charged with drug trafficking 500g of heroin. Is the mandatory death penalty applicable and what is the s 33B defence?",
        expected_domains=["drug_offences", "sentencing"],
        domain_for_retrieval="drug_offences",
        relevant_subtopics=["Misuse of Drugs Act", "Drug offences"],
        expected_keywords=["death penalty", "s 33B", "substantive assistance", "courier"],
    ),

    TestCase(
        id="TC-02",
        query="What must the prosecution prove for a drug possession charge under MDA s 8(a) and what presumptions apply?",
        expected_domains=["drug_offences"],
        domain_for_retrieval="drug_offences",
        relevant_subtopics=["Misuse of Drugs Act", "Drug offences"],
        expected_keywords=["possession", "knowledge", "presumption"],
    ),

    TestCase(
        id="TC-03",
        query="Accused pleaded guilty to trafficking 14.99g of diamorphine. What sentencing range applies?",
        expected_domains=["drug_offences", "sentencing"],
        domain_for_retrieval="drug_offences",
        relevant_subtopics=["Misuse of Drugs Act", "Sentencing principles", "Drug offences"],
        expected_keywords=["trafficking", "diamorphine", "mandatory minimum", "life imprisonment"],
    ),

    # ── SEXUAL OFFENCES ───────────────────────────────────────────────────────

    TestCase(
        id="TC-04",
        query="What sentencing framework applies to a rape conviction under Penal Code s 375?",
        expected_domains=["sexual_offences", "sentencing"],
        domain_for_retrieval="sexual_offences",
        relevant_subtopics=["Rape", "General sexual offences", "Aggravated rape"],
        expected_keywords=["s 375", "Band", "rape", "imprisonment"],
    ),

    TestCase(
        id="TC-05",
        query="Client accused of outrage of modesty on a commuter train. What are the elements of the offence and what sentence is typical?",
        expected_domains=["sexual_offences", "sentencing"],
        domain_for_retrieval="sexual_offences",
        relevant_subtopics=["Outrage of modesty"],
        expected_keywords=["s 354", "outrage of modesty", "modesty", "imprisonment"],
    ),

    TestCase(
        id="TC-06",
        query="My client says his girlfriend gave nonverbal consent and initiated the encounter but has filed a rape complaint. How does consent law apply?",
        expected_domains=["sexual_offences", "criminal_procedure"],
        domain_for_retrieval="sexual_offences",
        relevant_subtopics=["Rape", "Consent", "General sexual offences"],
        expected_keywords=["consent", "s 375", "reasonable grounds", "belief"],
    ),

    # ── VIOLENT CRIMES ────────────────────────────────────────────────────────

    TestCase(
        id="TC-07",
        query="Client charged with murder. What is the legal distinction between murder under s 300 and culpable homicide under s 299 Penal Code?",
        expected_domains=["violent_crimes", "sentencing"],
        domain_for_retrieval="violent_crimes",
        relevant_subtopics=["Murder", "Culpable homicide"],
        expected_keywords=["s 300", "s 299", "intention", "culpable homicide"],
    ),

    TestCase(
        id="TC-08",
        query="Accused stabbed the victim claiming self-defence. What are the legal requirements to establish private defence under the Penal Code?",
        expected_domains=["violent_crimes"],
        domain_for_retrieval="violent_crimes",
        relevant_subtopics=["Murder", "Voluntarily causing grievous hurt", "Grievous hurt"],
        expected_keywords=["private defence", "reasonable apprehension", "force", "s 96"],
    ),

    TestCase(
        id="TC-09",
        query="Accused convicted of voluntarily causing grievous hurt. What mitigating factors can reduce the sentence?",
        expected_domains=["violent_crimes", "sentencing"],
        domain_for_retrieval="violent_crimes",
        relevant_subtopics=["Voluntarily causing grievous hurt", "Grievous hurt", "Mitigating factors"],
        expected_keywords=["s 322", "grievous hurt", "mitigating", "imprisonment"],
    ),

    # ── PROPERTY & FINANCIAL ─────────────────────────────────────────────────

    TestCase(
        id="TC-10",
        query="What are the elements of criminal breach of trust by a company director under Penal Code s 409?",
        expected_domains=["property_financial", "sentencing"],
        domain_for_retrieval="property_financial",
        relevant_subtopics=["Property offence", "Corruption", "CDSA offences"],
        expected_keywords=["criminal breach of trust", "s 409", "entrustment", "dishonest"],
    ),

    TestCase(
        id="TC-11",
        query="Client under investigation for money laundering. What does the prosecution need to prove under the CDSA?",
        expected_domains=["property_financial"],
        domain_for_retrieval="property_financial",
        relevant_subtopics=["CDSA offences", "Confiscation of benefits"],
        expected_keywords=["CDSA", "benefits", "criminal conduct", "money laundering"],
    ),

    # ── SENTENCING ────────────────────────────────────────────────────────────

    TestCase(
        id="TC-12",
        query="How does the totality principle apply when a court is imposing consecutive sentences for multiple charges?",
        expected_domains=["sentencing"],
        domain_for_retrieval="sentencing",
        relevant_subtopics=["Totality principle", "Sentencing principles"],
        expected_keywords=["totality", "consecutive", "aggregate", "proportionality"],
    ),

    TestCase(
        id="TC-13",
        query="19-year-old first-time offender pleaded guilty to drug possession. Is probation or reformative training more likely?",
        expected_domains=["sentencing", "drug_offences"],
        domain_for_retrieval="sentencing",
        relevant_subtopics=["Sentencing principles", "Appeals against sentence"],
        expected_keywords=["reformative training", "probation", "rehabilitation", "young offender"],
    ),

    # ── CRIMINAL PROCEDURE ────────────────────────────────────────────────────

    TestCase(
        id="TC-14",
        query="Client wants to apply for bail pending appeal after conviction and sentence. What factors will the court consider?",
        expected_domains=["criminal_procedure"],
        domain_for_retrieval="criminal_procedure",
        relevant_subtopics=["Bail applications", "Criminal appeal"],
        expected_keywords=["bail", "appeal", "pending", "CPC"],
    ),

    TestCase(
        id="TC-15",
        query="Defence counsel wants to challenge the admissibility of the accused's cautioned statement. On what grounds can it be excluded?",
        expected_domains=["criminal_procedure"],
        domain_for_retrieval="criminal_procedure",
        relevant_subtopics=["Voluntariness", "Admissibility", "Admissibility of statements"],
        expected_keywords=["cautioned statement", "voluntariness", "inducement", "CPC s 258"],
    ),

    TestCase(
        id="TC-16",
        query="What is a criminal reference to the Court of Appeal under CPC s 397 and when is it available?",
        expected_domains=["criminal_procedure"],
        domain_for_retrieval="criminal_procedure",
        relevant_subtopics=["Criminal reference"],
        expected_keywords=["s 397", "question of law", "public interest", "Court of Appeal"],
    ),

    TestCase(
        id="TC-17",
        query="What is the criminal review procedure under CPC s 394H and how does it differ from a regular appeal?",
        expected_domains=["criminal_procedure"],
        domain_for_retrieval="criminal_procedure",
        relevant_subtopics=["Criminal review", "Criminal appeal"],
        expected_keywords=["s 394H", "review", "appellate", "CPC"],
    ),

    # ── REGULATORY ────────────────────────────────────────────────────────────

    TestCase(
        id="TC-18",
        query="Client charged with drink driving under Road Traffic Act s 67. What is the sentencing framework and mandatory disqualification period?",
        expected_domains=["regulatory", "sentencing"],
        domain_for_retrieval="regulatory",
        relevant_subtopics=["Drink driving", "Road Traffic Act"],
        expected_keywords=["s 67", "Road Traffic Act", "disqualification", "drink driving"],
    ),

    TestCase(
        id="TC-19",
        query="Company director charged under the Workplace Safety and Health Act after a fatal worksite accident. What is the maximum penalty?",
        expected_domains=["regulatory"],
        domain_for_retrieval="regulatory",
        relevant_subtopics=["Workplace Safety and Health Act"],
        expected_keywords=["WSHA", "duty", "employer", "fine", "workplace"],
    ),

    TestCase(
        id="TC-20",
        query="First offender caught driving at 180km/h in a 90km/h zone. What charge applies under the Road Traffic Act and what sentence is likely?",
        expected_domains=["regulatory", "sentencing"],
        domain_for_retrieval="regulatory",
        relevant_subtopics=["Dangerous driving", "Reckless driving", "Road Traffic Act"],
        expected_keywords=["dangerous driving", "Road Traffic Act", "custodial", "speed"],
    ),
]
