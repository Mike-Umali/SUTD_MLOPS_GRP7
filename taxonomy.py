"""
Singapore Criminal Law Taxonomy
Comprehensive labeling taxonomy for Singapore criminal law cases scraped from eLitigation.
Each entry: catchword_prefix -> (area, topic, subtopic, primary_statute)
"""

import re

# ---------------------------------------------------------------------------
# PRIMARY TAXONOMY
# Keys are catchword prefixes (matched via startswith after normalization).
# Values: (area_of_law, topic, subtopic, primary_statute)
# ---------------------------------------------------------------------------

CRIMINAL_TAXONOMY = {

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — OFFENCES AGAINST THE PERSON
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Murder":
        ("Criminal Law", "Offences against person", "Murder", "Penal Code s 300"),
    "Criminal Law — Offences — Culpable homicide":
        ("Criminal Law", "Offences against person", "Culpable homicide", "Penal Code s 299"),
    "Criminal Law — Offences — Causing death by rash or negligent act":
        ("Criminal Law", "Offences against person", "Causing death by rash or negligent act", "Penal Code ss 304A"),
    "Criminal Law — Offences — Voluntarily causing hurt":
        ("Criminal Law", "Offences against person", "Voluntarily causing hurt", "Penal Code s 321"),
    "Criminal Law — Offences — Voluntarily causing grievous hurt":
        ("Criminal Law", "Offences against person", "Voluntarily causing grievous hurt", "Penal Code s 322"),
    "Criminal Law — Offences — Grievous hurt":
        ("Criminal Law", "Offences against person", "Grievous hurt", "Penal Code s 320"),
    "Criminal Law — Offences — Hurt":
        ("Criminal Law", "Offences against person", "Hurt", "Penal Code s 319"),
    "Criminal Law — Offences — Assault":
        ("Criminal Law", "Offences against person", "Assault", "Penal Code s 351"),
    "Criminal Law — Offences — Wrongful restraint":
        ("Criminal Law", "Offences against person", "Wrongful restraint", "Penal Code s 339"),
    "Criminal Law — Offences — Wrongful confinement":
        ("Criminal Law", "Offences against person", "Wrongful confinement", "Penal Code s 340"),
    "Criminal Law — Offences — Kidnapping":
        ("Criminal Law", "Offences against person", "Kidnapping", "Kidnapping Act / Penal Code s 359"),
    "Criminal Law — Offences — Abduction":
        ("Criminal Law", "Offences against person", "Abduction", "Penal Code s 362"),
    "Criminal Law — Offences — Criminal intimidation":
        ("Criminal Law", "Offences against person", "Criminal intimidation", "Penal Code s 503"),
    "Criminal Law — Offences — Harassment":
        ("Criminal Law", "Offences against person", "Harassment", "Protection from Harassment Act"),
    "Criminal Law — Offences — Stalking":
        ("Criminal Law", "Offences against person", "Unlawful stalking", "Protection from Harassment Act s 7"),
    "Criminal Law — Offences — Using criminal force":
        ("Criminal Law", "Offences against person", "Using criminal force", "Penal Code s 350"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — SEXUAL OFFENCES
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Sexual offences":
        ("Criminal Law", "Sexual offences", "General sexual offences", "Penal Code Pt XI"),
    "Criminal Law — Offences — Rape":
        ("Criminal Law", "Sexual offences", "Rape", "Penal Code s 375"),
    "Criminal Law — Offences — Sexual assault by penetration":
        ("Criminal Law", "Sexual offences", "Sexual assault by penetration", "Penal Code s 376"),
    "Criminal Law — Offences — Outrage of modesty":
        ("Criminal Law", "Sexual offences", "Outrage of modesty", "Penal Code s 354"),
    "Criminal Law — Offences — Aggravated outrage of modesty":
        ("Criminal Law", "Sexual offences", "Aggravated outrage of modesty", "Penal Code s 354A"),
    "Criminal Law — Offences — Sexual exploitation of minor":
        ("Criminal Law", "Sexual offences", "Sexual exploitation of minor", "Penal Code s 376AA / CYPA"),
    "Criminal Law — Offences — Possession of obscene films":
        ("Criminal Law", "Sexual offences", "Possession of obscene films", "Films Act"),
    "Criminal Law — Offences — Voyeurism":
        ("Criminal Law", "Sexual offences", "Voyeurism", "Penal Code s 377BB"),
    "Criminal Law — Offences — Distribution of intimate image":
        ("Criminal Law", "Sexual offences", "Distribution of intimate image", "Penal Code s 377BC"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — PROPERTY OFFENCES
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Theft":
        ("Criminal Law", "Property offences", "Theft", "Penal Code s 378"),
    "Criminal Law — Offences — Theft in dwelling":
        ("Criminal Law", "Property offences", "Theft in dwelling", "Penal Code s 380"),
    "Criminal Law — Offences — Theft by clerk or servant":
        ("Criminal Law", "Property offences", "Theft by clerk or servant", "Penal Code s 381"),
    "Criminal Law — Offences — Robbery":
        ("Criminal Law", "Property offences", "Robbery", "Penal Code s 390"),
    "Criminal Law — Offences — Robbery with hurt":
        ("Criminal Law", "Property offences", "Robbery with hurt", "Penal Code s 394"),
    "Criminal Law — Offences — Gang robbery":
        ("Criminal Law", "Property offences", "Gang robbery", "Penal Code s 395"),
    "Criminal Law — Offences — Extortion":
        ("Criminal Law", "Property offences", "Extortion", "Penal Code s 383"),
    "Criminal Law — Offences — Criminal breach of trust":
        ("Criminal Law", "Property offences", "Criminal breach of trust", "Penal Code s 405"),
    "Criminal Law — Offences — Criminal breach of trust by public servant":
        ("Criminal Law", "Property offences", "Criminal breach of trust by public servant", "Penal Code s 409"),
    "Criminal Law — Offences — Cheating":
        ("Criminal Law", "Property offences", "Cheating", "Penal Code s 415"),
    "Criminal Law — Offences — Cheating and dishonestly inducing delivery":
        ("Criminal Law", "Property offences", "Cheating and dishonestly inducing delivery", "Penal Code s 420"),
    "Criminal Law — Offences — Property":
        ("Criminal Law", "Property offences", "Property offence", "Penal Code"),
    "Criminal Law — Offences — Mischief":
        ("Criminal Law", "Property offences", "Mischief", "Penal Code s 425"),
    "Criminal Law — Offences — House-breaking":
        ("Criminal Law", "Property offences", "House-breaking", "Penal Code s 453"),
    "Criminal Law — Offences — Fraudulent disposal of property":
        ("Criminal Law", "Property offences", "Fraudulent disposal of property", "Penal Code s 424"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — FORGERY AND FRAUD
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Forgery":
        ("Criminal Law", "Forgery and fraud", "Forgery", "Penal Code s 463"),
    "Criminal Law — Offences — Forgery for purpose of cheating":
        ("Criminal Law", "Forgery and fraud", "Forgery for purpose of cheating", "Penal Code s 468"),
    "Criminal Law — Offences — Using forged document":
        ("Criminal Law", "Forgery and fraud", "Using forged document", "Penal Code s 471"),
    "Criminal Law — Offences — Possession of forged document":
        ("Criminal Law", "Forgery and fraud", "Possession of forged document", "Penal Code s 474"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — PUBLIC ORDER OFFENCES
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Rioting":
        ("Criminal Law", "Public order offences", "Rioting", "Penal Code s 146"),
    "Criminal Law — Offences — Unlawful assembly":
        ("Criminal Law", "Public order offences", "Unlawful assembly", "Penal Code s 141"),
    "Criminal Law — Offences — Affray":
        ("Criminal Law", "Public order offences", "Affray", "Penal Code s 267A"),
    "Criminal Law — Offences — Public nuisance":
        ("Criminal Law", "Public order offences", "Public nuisance", "Penal Code s 268"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — OFFENCES AGAINST PUBLIC SERVANTS / ADMINISTRATION
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Obstructing public servant":
        ("Criminal Law", "Offences against administration", "Obstructing public servant", "Penal Code s 204A"),
    "Criminal Law — Offences — Resisting arrest":
        ("Criminal Law", "Offences against administration", "Resisting arrest", "Penal Code s 225B"),
    "Criminal Law — Offences — False information to public servant":
        ("Criminal Law", "Offences against administration", "False information to public servant", "Penal Code s 182"),
    "Criminal Law — Offences — Giving false evidence":
        ("Criminal Law", "Offences against administration", "Giving false evidence", "Penal Code s 191"),
    "Criminal Law — Offences — Fabricating evidence":
        ("Criminal Law", "Offences against administration", "Fabricating evidence", "Penal Code s 192"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: MISUSE OF DRUGS ACT
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Misuse of Drugs Act":
        ("Criminal Law", "Statutory offences", "Misuse of Drugs Act", "Misuse of Drugs Act"),
    "Criminal Law — Statutory offences — Drug trafficking":
        ("Criminal Law", "Statutory offences", "Drug trafficking", "Misuse of Drugs Act s 5"),
    "Criminal Law — Statutory offences — Drug possession":
        ("Criminal Law", "Statutory offences", "Drug possession", "Misuse of Drugs Act s 8(a)"),
    "Criminal Law — Statutory offences — Drug consumption":
        ("Criminal Law", "Statutory offences", "Drug consumption", "Misuse of Drugs Act s 8(b)"),
    "Criminal Law — Statutory offences — Drug importation":
        ("Criminal Law", "Statutory offences", "Drug importation", "Misuse of Drugs Act s 7"),
    "Criminal Law — Statutory offences — Drug exportation":
        ("Criminal Law", "Statutory offences", "Drug exportation", "Misuse of Drugs Act s 7"),
    "Criminal Law — Statutory offences — Enhanced trafficking":
        ("Criminal Law", "Statutory offences", "Enhanced trafficking", "Misuse of Drugs Act s 33B"),
    "Criminal Law — Statutory offences — Possession of drug paraphernalia":
        ("Criminal Law", "Statutory offences", "Drug paraphernalia", "Misuse of Drugs Act s 9"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: ARMS & EXPLOSIVES
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Arms Offences Act":
        ("Criminal Law", "Statutory offences", "Arms Offences Act", "Arms Offences Act"),
    "Criminal Law — Statutory offences — Possession of arms":
        ("Criminal Law", "Statutory offences", "Possession of arms", "Arms Offences Act s 3"),
    "Criminal Law — Statutory offences — Possession of explosives":
        ("Criminal Law", "Statutory offences", "Possession of explosives", "Explosive Substances Act"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: CORRUPTION
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Prevention of Corruption Act":
        ("Criminal Law", "Statutory offences", "Prevention of Corruption Act", "Prevention of Corruption Act"),
    "Criminal Law — Statutory offences — Corruption":
        ("Criminal Law", "Statutory offences", "Corruption", "Prevention of Corruption Act s 6"),
    "Criminal Law — Statutory offences — Bribery":
        ("Criminal Law", "Statutory offences", "Bribery", "Prevention of Corruption Act"),
    "Criminal Law — Statutory offences — Money laundering":
        ("Criminal Law", "Statutory offences", "Money laundering", "CDSA s 47"),
    "Criminal Law — Statutory offences — Proceeds of crime":
        ("Criminal Law", "Statutory offences", "Proceeds of crime", "CDSA"),
    "Criminal Law — Statutory offences — Corruption, Drug Trafficking and Other Serious Crimes":
        ("Criminal Law", "Statutory offences", "CDSA offences", "CDSA"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: COMPUTER & CYBER
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Computer Misuse Act":
        ("Criminal Law", "Statutory offences", "Computer Misuse Act", "Computer Misuse Act"),
    "Criminal Law — Statutory offences — Unauthorised computer access":
        ("Criminal Law", "Statutory offences", "Unauthorised computer access", "Computer Misuse Act s 3"),
    "Criminal Law — Statutory offences — Cybercrime":
        ("Criminal Law", "Statutory offences", "Cybercrime", "Computer Misuse Act"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: IMMIGRATION
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Immigration Act":
        ("Criminal Law", "Statutory offences", "Immigration Act", "Immigration Act"),
    "Criminal Law — Statutory offences — Overstaying":
        ("Criminal Law", "Statutory offences", "Overstaying", "Immigration Act s 15"),
    "Criminal Law — Statutory offences — Illegal entry":
        ("Criminal Law", "Statutory offences", "Illegal entry", "Immigration Act s 6"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: ROAD TRAFFIC
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Road Traffic Act":
        ("Criminal Law", "Statutory offences", "Road Traffic Act", "Road Traffic Act"),
    "Criminal Law — Statutory offences — Dangerous driving":
        ("Criminal Law", "Statutory offences", "Dangerous driving", "Road Traffic Act s 64"),
    "Criminal Law — Statutory offences — Drink driving":
        ("Criminal Law", "Statutory offences", "Drink driving", "Road Traffic Act s 67"),
    "Criminal Law — Statutory offences — Driving without licence":
        ("Criminal Law", "Statutory offences", "Driving without licence", "Road Traffic Act s 35"),
    "Criminal Law — Statutory offences — Hit and run":
        ("Criminal Law", "Statutory offences", "Failing to render assistance", "Road Traffic Act s 84"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — STATUTORY OFFENCES: MISCELLANEOUS
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Penal Code":
        ("Criminal Law", "Statutory offences", "Penal Code", "Penal Code"),
    "Criminal Law — Statutory offences — Employment of Foreign Manpower Act":
        ("Criminal Law", "Statutory offences", "Employment of Foreign Manpower Act", "EFMA"),
    "Criminal Law — Statutory offences — Income Tax Act":
        ("Criminal Law", "Statutory offences", "Income Tax Act offences", "Income Tax Act"),
    "Criminal Law — Statutory offences — Securities and Futures Act":
        ("Criminal Law", "Statutory offences", "Securities and Futures Act", "Securities and Futures Act"),
    "Criminal Law — Statutory offences — Customs Act":
        ("Criminal Law", "Statutory offences", "Customs Act", "Customs Act"),
    "Criminal Law — Statutory offences — Wildlife Act":
        ("Criminal Law", "Statutory offences", "Wildlife Act", "Wildlife Act"),
    "Criminal Law — Statutory offences — Moneylenders Act":
        ("Criminal Law", "Statutory offences", "Moneylenders Act", "Moneylenders Act"),
    "Criminal Law — Statutory offences — Environmental":
        ("Criminal Law", "Statutory offences", "Environmental offences", "EPHA / NEA Act"),
    "Criminal Law — Statutory offences — Health Products Act":
        ("Criminal Law", "Statutory offences", "Health Products Act", "Health Products Act"),
    "Criminal Law — Statutory offences — Vandalism Act":
        ("Criminal Law", "Statutory offences", "Vandalism", "Vandalism Act"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — ELEMENTS OF CRIME
    # -----------------------------------------------------------------------
    "Criminal Law — Elements of crime — Mens rea":
        ("Criminal Law", "Elements of crime", "Mens rea", "Penal Code"),
    "Criminal Law — Elements of crime — Actus reus":
        ("Criminal Law", "Elements of crime", "Actus reus", "Penal Code"),
    "Criminal Law — Elements of crime — Intention":
        ("Criminal Law", "Elements of crime", "Intention", "Penal Code s 26A"),
    "Criminal Law — Elements of crime — Knowledge":
        ("Criminal Law", "Elements of crime", "Knowledge", "Penal Code s 26B"),
    "Criminal Law — Elements of crime — Recklessness":
        ("Criminal Law", "Elements of crime", "Recklessness", "Penal Code s 26C"),
    "Criminal Law — Elements of crime — Negligence":
        ("Criminal Law", "Elements of crime", "Negligence", "Penal Code s 26D"),
    "Criminal Law — Elements of crime — Causation":
        ("Criminal Law", "Elements of crime", "Causation", "Penal Code"),
    "Criminal Law — Elements of crime":
        ("Criminal Law", "Elements of crime", "General", "Penal Code"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — PARTICIPATION: COMMON INTENTION, ABETMENT, CONSPIRACY
    # -----------------------------------------------------------------------
    "Criminal Law — Common intention":
        ("Criminal Law", "Participation in crime", "Common intention", "Penal Code s 34"),
    "Criminal Law — Abetment":
        ("Criminal Law", "Participation in crime", "Abetment", "Penal Code ss 107-117"),
    "Criminal Law — Abetment — Conspiracy":
        ("Criminal Law", "Participation in crime", "Abetment by conspiracy", "Penal Code s 107(b)"),
    "Criminal Law — Abetment — Instigation":
        ("Criminal Law", "Participation in crime", "Abetment by instigation", "Penal Code s 107(a)"),
    "Criminal Law — Criminal conspiracy":
        ("Criminal Law", "Participation in crime", "Criminal conspiracy", "Penal Code s 120A"),
    "Criminal Law — Attempt":
        ("Criminal Law", "Participation in crime", "Attempt", "Penal Code s 511"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — DEFENCES (GENERAL EXCEPTIONS)
    # -----------------------------------------------------------------------
    "Criminal Law — General exceptions — Private defence":
        ("Criminal Law", "General exceptions / Defences", "Private defence", "Penal Code ss 96-106"),
    "Criminal Law — General exceptions — Intoxication":
        ("Criminal Law", "General exceptions / Defences", "Intoxication", "Penal Code ss 85-86"),
    "Criminal Law — General exceptions — Unsoundness of mind":
        ("Criminal Law", "General exceptions / Defences", "Unsoundness of mind", "Penal Code s 84"),
    "Criminal Law — General exceptions — Diminished responsibility":
        ("Criminal Law", "General exceptions / Defences", "Diminished responsibility", "Penal Code s 301 / Second Schedule"),
    "Criminal Law — General exceptions — Sudden fight":
        ("Criminal Law", "General exceptions / Defences", "Sudden fight", "Penal Code s 294 Exception 4"),
    "Criminal Law — General exceptions — Consent":
        ("Criminal Law", "General exceptions / Defences", "Consent", "Penal Code s 90"),
    "Criminal Law — General exceptions — Duress":
        ("Criminal Law", "General exceptions / Defences", "Duress", "Penal Code s 94"),
    "Criminal Law — General exceptions — Necessity":
        ("Criminal Law", "General exceptions / Defences", "Necessity", "Penal Code s 81"),
    "Criminal Law — General exceptions":
        ("Criminal Law", "General exceptions / Defences", "General", "Penal Code ss 76-106"),
    "Criminal Law — Defences":
        ("Criminal Law", "General exceptions / Defences", "General", "Penal Code"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — SENTENCING
    # -----------------------------------------------------------------------
    "Criminal Law — Sentencing":
        ("Criminal Law", "Sentencing", "General sentencing principles", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Principles":
        ("Criminal Law", "Sentencing", "Sentencing principles", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Benchmarks":
        ("Criminal Law", "Sentencing", "Benchmark sentences", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Sentencing bands":
        ("Criminal Law", "Sentencing", "Sentencing bands", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Mandatory minimum":
        ("Criminal Law", "Sentencing", "Mandatory minimum sentence", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Mandatory death penalty":
        ("Criminal Law", "Sentencing", "Mandatory death penalty", "Misuse of Drugs Act / Penal Code"),
    "Criminal Procedure and Sentencing — Sentencing — Reformative training":
        ("Criminal Law", "Sentencing", "Reformative training", "Criminal Procedure Code 2010 s 305"),
    "Criminal Procedure and Sentencing — Sentencing — Probation":
        ("Criminal Law", "Sentencing", "Probation", "Probation of Offenders Act"),
    "Criminal Procedure and Sentencing — Sentencing — Community based sentencing":
        ("Criminal Law", "Sentencing", "Community based sentencing", "Criminal Procedure Code 2010 ss 337-351"),
    "Criminal Procedure and Sentencing — Sentencing — Mandatory treatment order":
        ("Criminal Law", "Sentencing", "Mandatory treatment order", "Criminal Procedure Code 2010 s 339"),
    "Criminal Procedure and Sentencing — Sentencing — Day reporting order":
        ("Criminal Law", "Sentencing", "Day reporting order", "Criminal Procedure Code 2010 s 341"),
    "Criminal Procedure and Sentencing — Sentencing — Community work order":
        ("Criminal Law", "Sentencing", "Community work order", "Criminal Procedure Code 2010 s 346"),
    "Criminal Procedure and Sentencing — Sentencing — Caning":
        ("Criminal Law", "Sentencing", "Caning", "Criminal Procedure Code 2010 s 267"),
    "Criminal Procedure and Sentencing — Sentencing — Imprisonment in default":
        ("Criminal Law", "Sentencing", "Imprisonment in default", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Preventive detention":
        ("Criminal Law", "Sentencing", "Preventive detention", "Criminal Procedure Code 2010 s 304"),
    "Criminal Procedure and Sentencing — Sentencing — Corrective training":
        ("Criminal Law", "Sentencing", "Corrective training", "Criminal Procedure Code 2010 s 304"),
    "Criminal Procedure and Sentencing — Sentencing — Totality principle":
        ("Criminal Law", "Sentencing", "Totality principle", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Consecutive sentences":
        ("Criminal Law", "Sentencing", "Consecutive sentences", "Criminal Procedure Code 2010 s 322"),
    "Criminal Procedure and Sentencing — Sentencing — Concurrent sentences":
        ("Criminal Law", "Sentencing", "Concurrent sentences", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Mitigating factors":
        ("Criminal Law", "Sentencing", "Mitigating factors", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Aggravating factors":
        ("Criminal Law", "Sentencing", "Aggravating factors", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Plea of guilt":
        ("Criminal Law", "Sentencing", "Plea of guilt", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Judicial mercy":
        ("Criminal Law", "Sentencing", "Judicial mercy", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Appeals":
        ("Criminal Law", "Sentencing", "Appeals against sentence", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Deterrence":
        ("Criminal Law", "Sentencing", "Deterrence", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Rehabilitation":
        ("Criminal Law", "Sentencing", "Rehabilitation", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Prevention":
        ("Criminal Law", "Sentencing", "Prevention", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing — Retribution":
        ("Criminal Law", "Sentencing", "Retribution", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Sentencing":
        ("Criminal Law", "Sentencing", "General", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — ARREST, SEARCH & DETENTION
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing — Arrest":
        ("Criminal Procedure", "Arrest", "Arrest powers", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Search and seizure":
        ("Criminal Procedure", "Search and seizure", "Powers of search", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Detention":
        ("Criminal Procedure", "Detention", "Remand", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Bail":
        ("Criminal Procedure", "Bail", "Bail applications", "Criminal Procedure Code 2010 s 93"),
    "Criminal Procedure and Sentencing — Bail — Revocation":
        ("Criminal Procedure", "Bail", "Revocation of bail", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Bail — Forfeiture":
        ("Criminal Procedure", "Bail", "Forfeiture of bail", "Criminal Procedure Code 2010 s 107"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — CHARGE AND TRIAL
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing — Charge":
        ("Criminal Procedure", "Charge", "Framing of charge", "Criminal Procedure Code 2010 ss 124-140"),
    "Criminal Procedure and Sentencing — Charge — Alteration":
        ("Criminal Procedure", "Charge", "Alteration of charge", "Criminal Procedure Code 2010 s 128"),
    "Criminal Procedure and Sentencing — Charge — Amalgamated charges":
        ("Criminal Procedure", "Charge", "Amalgamated charges", "Criminal Procedure Code 2010 s 124A"),
    "Criminal Procedure and Sentencing — Charge — Joinder":
        ("Criminal Procedure", "Charge", "Joinder of charges", "Criminal Procedure Code 2010 s 133"),
    "Criminal Procedure and Sentencing — Trial":
        ("Criminal Procedure", "Trial", "Trial procedure", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Acquittal — Acquittal without defence":
        ("Criminal Procedure", "Trial", "Acquittal without defence being called", "Criminal Procedure Code 2010 s 230"),
    "Criminal Procedure and Sentencing — Acquittal":
        ("Criminal Procedure", "Trial", "Acquittal", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — STATEMENTS AND EVIDENCE
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing — Statements":
        ("Criminal Procedure", "Statements", "Admissibility of statements", "Criminal Procedure Code 2010 ss 258-268"),
    "Criminal Procedure and Sentencing — Statements — Admissibility":
        ("Criminal Procedure", "Statements", "Admissibility", "Criminal Procedure Code 2010 s 258"),
    "Criminal Procedure and Sentencing — Statements — Voluntariness":
        ("Criminal Procedure", "Statements", "Voluntariness", "Criminal Procedure Code 2010 s 258(3)"),
    "Criminal Procedure and Sentencing — Statements — Contemporaneous statements":
        ("Criminal Procedure", "Statements", "Contemporaneous statements", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — APPEALS AND REVIEW
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing — Appeals":
        ("Criminal Procedure", "Appeals", "Criminal appeal", "Criminal Procedure Code 2010 ss 374-404"),
    "Criminal Procedure and Sentencing — Criminal motions — Adducing fresh evidence":
        ("Criminal Procedure", "Appeals", "Adducing fresh evidence", "Criminal Procedure Code 2010 s 392"),
    "Criminal Procedure and Sentencing — Criminal review":
        ("Criminal Procedure", "Review", "Criminal review", "Criminal Procedure Code 2010 s 394H"),
    "Criminal Procedure and Sentencing — Criminal reference":
        ("Criminal Procedure", "Reference", "Criminal reference", "Criminal Procedure Code 2010 s 397"),
    "Criminal Procedure and Sentencing — Criminal revision":
        ("Criminal Procedure", "Revision", "Criminal revision", "Criminal Procedure Code 2010 s 400"),
    "Criminal Procedure and Sentencing — Criminal motions":
        ("Criminal Procedure", "Motions", "Criminal motions", "Criminal Procedure Code 2010"),
    "Criminal Procedure and Sentencing — Stay of execution":
        ("Criminal Procedure", "Execution", "Stay of execution", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — CONFISCATION AND MUTUAL LEGAL ASSISTANCE
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing — Confiscation":
        ("Criminal Procedure", "Confiscation", "Confiscation of benefits", "CDSA / CPC"),
    "Criminal Procedure and Sentencing — Mutual legal assistance":
        ("Criminal Procedure", "Mutual legal assistance", "Mutual legal assistance", "Mutual Assistance in Criminal Matters Act"),
    "Criminal Procedure and Sentencing — Extradition":
        ("Criminal Procedure", "Extradition", "Extradition", "Extradition Act"),
    "Criminal Procedure and Sentencing — Community based sentencing":
        ("Criminal Procedure", "Community based sentencing", "Community based sentencing orders", "Criminal Procedure Code 2010 ss 337-351"),

    # -----------------------------------------------------------------------
    # CRIMINAL PROCEDURE — GENERAL
    # -----------------------------------------------------------------------
    "Criminal Procedure and Sentencing":
        ("Criminal Procedure", "General", "General", "Criminal Procedure Code 2010"),
    "Criminal Law — Appeal":
        ("Criminal Procedure", "Appeals", "Criminal appeal", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # EVIDENCE IN CRIMINAL CONTEXT
    # -----------------------------------------------------------------------
    "Evidence — Similar fact evidence":
        ("Evidence", "Similar fact evidence", "Criminal proceedings", "Evidence Act s 11(b)"),
    "Evidence — Hearsay":
        ("Evidence", "Hearsay", "Admissibility of hearsay", "Evidence Act s 32"),
    "Evidence — Expert evidence":
        ("Evidence", "Expert evidence", "Expert testimony in criminal proceedings", "Evidence Act s 47"),
    "Evidence — Admissibility":
        ("Evidence", "Admissibility", "Admissibility of evidence", "Evidence Act"),
    "Evidence — Witnesses":
        ("Evidence", "Witnesses", "Witness testimony", "Evidence Act"),
    "Evidence — Corroboration":
        ("Evidence", "Corroboration", "Corroboration in criminal proceedings", "Evidence Act"),
    "Evidence — Principles":
        ("Evidence", "Principles", "General principles of evidence", "Evidence Act"),
    "Evidence — Proof of evidence":
        ("Evidence", "Proof of evidence", "General", "Evidence Act"),
    "Evidence — Presumptions":
        ("Evidence", "Presumptions", "Presumptions of fact", "Evidence Act"),
    "Evidence — Adverse inferences":
        ("Evidence", "Adverse inferences", "Drawing of adverse inferences", "Evidence Act s 116"),
    "Evidence — Weight of evidence":
        ("Evidence", "Weight of evidence", "Assessment of evidence", "Evidence Act"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — SPECIAL EXCEPTIONS (alias for General exceptions)
    # -----------------------------------------------------------------------
    "Criminal Law — Special exceptions — Sudden fight":
        ("Criminal Law", "General exceptions / Defences", "Sudden fight", "Penal Code s 294 Exception 4"),
    "Criminal Law — Special exceptions — Diminished responsibility":
        ("Criminal Law", "General exceptions / Defences", "Diminished responsibility", "Penal Code s 301 / Second Schedule"),
    "Criminal Law — Special exceptions — Provocation":
        ("Criminal Law", "General exceptions / Defences", "Provocation", "Penal Code s 294 Exception 1"),
    "Criminal Law — Special exceptions":
        ("Criminal Law", "General exceptions / Defences", "General", "Penal Code ss 76-106"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — COMPLICITY
    # -----------------------------------------------------------------------
    "Criminal Law — Complicity — Common intention":
        ("Criminal Law", "Participation in crime", "Common intention", "Penal Code s 34"),
    "Criminal Law — Complicity — Criminal conspiracy":
        ("Criminal Law", "Participation in crime", "Criminal conspiracy", "Penal Code s 120A"),
    "Criminal Law — Complicity":
        ("Criminal Law", "Participation in crime", "General", "Penal Code"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — ADDITIONAL OFFENCES
    # -----------------------------------------------------------------------
    "Criminal Law — Offences — Sexual assault":
        ("Criminal Law", "Sexual offences", "Sexual assault by penetration", "Penal Code s 376"),
    "Criminal Law — Offences — Attempted rape":
        ("Criminal Law", "Sexual offences", "Attempted rape", "Penal Code ss 375, 511"),
    "Criminal Law — Offences — Aggravated rape":
        ("Criminal Law", "Sexual offences", "Aggravated rape", "Penal Code s 375(3)"),
    "Criminal Law — Offences — Attempted murder":
        ("Criminal Law", "Offences against person", "Attempted murder", "Penal Code ss 300, 511"),
    "Criminal Law — Offences — Attempt to murder":
        ("Criminal Law", "Offences against person", "Attempted murder", "Penal Code ss 300, 511"),
    "Criminal Law — Offences — Obstructing the course of justice":
        ("Criminal Law", "Offences against administration", "Obstructing the course of justice", "Penal Code s 204A"),
    "Criminal Law — Offences — Possession of an instrument":
        ("Criminal Law", "Property offences", "Possession of housebreaking instrument", "Penal Code s 453"),
    "Criminal Law — Offences — Endangered species":
        ("Criminal Law", "Statutory offences", "Endangered species / Wildlife", "Wildlife Act / CITES"),
    "Criminal Law — Offences — Causing or allowing":
        ("Criminal Law", "Offences against person", "Causing or allowing death of child", "Penal Code s 304B"),
    "Criminal Law — Offences — Disposal of corpse":
        ("Criminal Law", "Offences against person", "Disposal of corpse", "Penal Code s 268"),
    "Criminal Law — Offences — Unnatural offences":
        ("Criminal Law", "Sexual offences", "Unnatural offences", "Penal Code s 377"),
    "Criminal Law — Offences — Perverting the course of justice":
        ("Criminal Law", "Offences against administration", "Perverting the course of justice", "Penal Code s 204A"),
    "Criminal Law — Offences — House-trespass":
        ("Criminal Law", "Property offences", "House-trespass", "Penal Code s 442"),
    "Criminal Law — Offences — Personating a public servant":
        ("Criminal Law", "Offences against administration", "Personating a public servant", "Penal Code s 170"),
    "Criminal Law — Offences — Public decency and morals":
        ("Criminal Law", "Public order offences", "Public decency and morals", "Penal Code ss 268, 294"),
    "Criminal Law — Offences — Documents":
        ("Criminal Law", "Forgery and fraud", "Documents", "Penal Code"),
    "Criminal Law — Offences — Offences against public servant":
        ("Criminal Law", "Offences against administration", "Offences against public servants", "Penal Code"),
    "Criminal Law — Offences — Causing grievous hurt":
        ("Criminal Law", "Offences against person", "Voluntarily causing grievous hurt", "Penal Code s 322"),
    "Criminal Law — Offences — Corruption":
        ("Criminal Law", "Statutory offences", "Corruption", "Prevention of Corruption Act"),
    "Criminal Law — Offences — Section 6(a) Prevention of Corruption Act":
        ("Criminal Law", "Statutory offences", "Corruption", "Prevention of Corruption Act s 6(a)"),
    "Criminal Law — Offences — Section 204A Penal Code":
        ("Criminal Law", "Offences against administration", "Obstructing public servant", "Penal Code s 204A"),
    "Criminal Law — Offences — Section 147 of the Penal Code":
        ("Criminal Law", "Public order offences", "Rioting", "Penal Code s 147"),
    "Criminal Law — Offences — Section 201 Penal Code":
        ("Criminal Law", "Offences against administration", "Fabricating evidence", "Penal Code s 201"),
    "Criminal Law — Offences — Sexual exploitation":
        ("Criminal Law", "Sexual offences", "Sexual exploitation of minor", "Penal Code s 376AA / CYPA"),
    "Criminal Law — Offences — Offences relating to":
        ("Criminal Law", "Offences against person", "General", "Penal Code"),
    "Criminal Law — Offences — Offences against pub":
        ("Criminal Law", "Offences against administration", "Offences against public servants", "Penal Code"),
    "Criminal Law — Offences":
        ("Criminal Law", "Offences", "General", "Penal Code"),
    "Criminal Law — Defamation":
        ("Criminal Law", "Statutory offences", "Defamation", "Penal Code s 499"),
    "Criminal Law — Cheating":
        ("Criminal Law", "Property offences", "Cheating", "Penal Code s 415"),
    "Criminal Law — Criminal Review":
        ("Criminal Procedure", "Review", "Criminal review", "Criminal Procedure Code 2010 s 394H"),
    "Criminal Law — Criminal motion":
        ("Criminal Procedure", "Motions", "Criminal motions", "Criminal Procedure Code 2010"),
    "Criminal Law — Statutory offence":
        ("Criminal Law", "Statutory offences", "General", "Penal Code"),

    # -----------------------------------------------------------------------
    # CRIMINAL LAW — ADDITIONAL STATUTORY OFFENCES
    # -----------------------------------------------------------------------
    "Criminal Law — Statutory offences — Companies Act":
        ("Criminal Law", "Statutory offences", "Companies Act offences", "Companies Act"),
    "Criminal Law — Statutory offences — Multi-Level Marketing":
        ("Criminal Law", "Statutory offences", "Multi-Level Marketing Act", "Multi-Level Marketing and Pyramid Selling (Prohibition) Act"),
    "Criminal Law — Statutory offences — Work Injury Compensation Act":
        ("Criminal Law", "Statutory offences", "Work Injury Compensation Act", "Work Injury Compensation Act"),
    "Criminal Law — Statutory offences — Payment Services Act":
        ("Criminal Law", "Statutory offences", "Payment Services Act", "Payment Services Act"),
    "Criminal Law — Statutory offences — Parliamentary Elections Act":
        ("Criminal Law", "Statutory offences", "Parliamentary Elections Act", "Parliamentary Elections Act"),
    "Criminal Law — Statutory offences — Children and Young Persons Act":
        ("Criminal Law", "Statutory offences", "Children and Young Persons Act", "Children and Young Persons Act"),
    "Criminal Law — Statutory offences — Residential Property Act":
        ("Criminal Law", "Statutory offences", "Residential Property Act", "Residential Property Act"),
    "Criminal Law — Statutory offences — Corrosive":
        ("Criminal Law", "Statutory offences", "Corrosive and Explosive Substances Act", "Corrosive and Explosive Substances and Offensive Weapons Act"),
    "Criminal Law — Statutory offences — Prisons Act":
        ("Criminal Law", "Statutory offences", "Prisons Act", "Prisons Act"),
    "Criminal Law — Statutory offences — Massage Establishments Act":
        ("Criminal Law", "Statutory offences", "Massage Establishments Act", "Massage Establishments Act 2017"),
    "Criminal Law — Statutory offences — Animals and Birds Act":
        ("Criminal Law", "Statutory offences", "Animals and Birds Act", "Animals and Birds Act"),
    "Criminal Law — Statutory offences — Workplace Safety and Health Act":
        ("Criminal Law", "Statutory offences", "Workplace Safety and Health Act", "Workplace Safety and Health Act"),
    "Criminal Law — Statutory offences — Workplace Safety":
        ("Criminal Law", "Statutory offences", "Workplace Safety and Health Act", "Workplace Safety and Health Act"),
    "Criminal Law — Statutory offences — Motor Vehicle":
        ("Criminal Law", "Statutory offences", "Motor Vehicles Act", "Motor Vehicles (Third-Party Risks and Compensation) Act"),
    "Criminal Law — Statutory offences — Women's Charter":
        ("Criminal Law", "Statutory offences", "Women's Charter", "Women's Charter"),
    "Criminal Law — Statutory offences — Passports Act":
        ("Criminal Law", "Statutory offences", "Passports Act", "Passports Act"),
    "Criminal Law — Statutory offences — Building Control Act":
        ("Criminal Law", "Statutory offences", "Building Control Act", "Building Control Act"),
    "Criminal Law — Statutory offences — Remote Gambling Act":
        ("Criminal Law", "Statutory offences", "Remote Gambling Act", "Remote Gambling Act"),
    "Criminal Law — Statutory offences — Section 8":
        ("Criminal Law", "Statutory offences", "Drug offences", "Misuse of Drugs Act s 8"),
    "Criminal Law — Statutory offences — s 340":
        ("Criminal Procedure", "Community based sentencing", "Day reporting order", "Criminal Procedure Code 2010 s 340"),

    # -----------------------------------------------------------------------
    # CONSTITUTIONAL LAW
    # -----------------------------------------------------------------------
    "Constitutional Law — Equal protection of the law":
        ("Constitutional Law", "Equal protection", "Equal protection of the law", "Constitution of Singapore Art 12"),
    "Constitutional Law — Equality before the law":
        ("Constitutional Law", "Equal protection", "Equality before the law", "Constitution of Singapore Art 12"),
    "Constitutional Law — Fundamental liberties":
        ("Constitutional Law", "Fundamental liberties", "General", "Constitution of Singapore Pt IV"),
    "Constitutional Law — Fundamental Liberties":
        ("Constitutional Law", "Fundamental liberties", "General", "Constitution of Singapore Pt IV"),
    "Constitutional Law — Judicial review":
        ("Constitutional Law", "Judicial review", "Judicial review", "Constitution of Singapore"),
    "Constitutional Law — Accused person":
        ("Constitutional Law", "Accused person", "Rights of accused", "Constitution of Singapore Art 9"),
    "Constitutional Law — Natural justice":
        ("Constitutional Law", "Natural justice", "Natural justice", "Constitution of Singapore Art 9"),
    "Constitutional Law — Natural Justice":
        ("Constitutional Law", "Natural justice", "Natural justice", "Constitution of Singapore Art 9"),
    "Constitutional Law — Separation of powers":
        ("Constitutional Law", "Separation of powers", "Separation of powers", "Constitution of Singapore"),
    "Constitutional Law — Judicial power":
        ("Constitutional Law", "Judicial power", "Judicial power", "Constitution of Singapore Art 93"),
    "Constitutional Law — Discrimination":
        ("Constitutional Law", "Equal protection", "Discrimination", "Constitution of Singapore Art 12"),
    "Constitutional Law — Liberty of the person":
        ("Constitutional Law", "Fundamental liberties", "Liberty of the person", "Constitution of Singapore Art 9"),
    "Constitutional Law — Attorney-General":
        ("Constitutional Law", "Attorney-General", "Prosecutorial discretion", "Constitution of Singapore Art 35"),
    "Constitutional Law — Constitution":
        ("Constitutional Law", "Constitutional interpretation", "General", "Constitution of Singapore"),
    "Constitutional Law — Judicial Review":
        ("Constitutional Law", "Judicial review", "Judicial review", "Constitution of Singapore"),

    # -----------------------------------------------------------------------
    # ROAD TRAFFIC (top-level prefix)
    # -----------------------------------------------------------------------
    "Road Traffic — Offences — Drink driving":
        ("Criminal Law", "Statutory offences", "Drink driving", "Road Traffic Act s 67"),
    "Road Traffic — Offences — Careless driving":
        ("Criminal Law", "Statutory offences", "Careless driving", "Road Traffic Act s 65"),
    "Road Traffic — Offences — Dangerous driving":
        ("Criminal Law", "Statutory offences", "Dangerous driving", "Road Traffic Act s 64"),
    "Road Traffic — Offences — Reckless driving":
        ("Criminal Law", "Statutory offences", "Reckless driving", "Road Traffic Act s 64"),
    "Road Traffic — Offences":
        ("Criminal Law", "Statutory offences", "Road Traffic Act offences", "Road Traffic Act"),
    "Road Traffic":
        ("Criminal Law", "Statutory offences", "Road Traffic Act offences", "Road Traffic Act"),

    # -----------------------------------------------------------------------
    # ADMINISTRATIVE LAW
    # -----------------------------------------------------------------------
    "Administrative Law — Remedies — Quashing order":
        ("Administrative Law", "Judicial review", "Quashing order", "Supreme Court of Judicature Act"),
    "Administrative Law — Remedies — Declaration":
        ("Administrative Law", "Judicial review", "Declaration", "Supreme Court of Judicature Act"),
    "Administrative Law — Remedies — Mandatory order":
        ("Administrative Law", "Judicial review", "Mandatory order", "Supreme Court of Judicature Act"),
    "Administrative Law — Judicial review":
        ("Administrative Law", "Judicial review", "Judicial review", "Supreme Court of Judicature Act"),
    "Administrative Law — Remedies":
        ("Administrative Law", "Judicial review", "Remedies", "Supreme Court of Judicature Act"),

    # -----------------------------------------------------------------------
    # COURTS AND JURISDICTION
    # -----------------------------------------------------------------------
    "Courts and Jurisdiction — Jurisdiction":
        ("Criminal Procedure", "Jurisdiction", "Court jurisdiction", "Supreme Court of Judicature Act"),
    "Courts and Jurisdiction — Court judgments":
        ("Criminal Procedure", "Judgments", "Court judgments", "Supreme Court of Judicature Act"),
    "Courts and Jurisdiction — Appeals":
        ("Criminal Procedure", "Appeals", "Criminal appeal", "Criminal Procedure Code 2010"),
    "Courts and Jurisdiction":
        ("Criminal Procedure", "Jurisdiction", "General", "Supreme Court of Judicature Act"),

    # -----------------------------------------------------------------------
    # ABUSE OF PROCESS
    # -----------------------------------------------------------------------
    "Abuse of process":
        ("Criminal Procedure", "Abuse of process", "General", "Criminal Procedure Code 2010"),
    "Abuse of Process":
        ("Criminal Procedure", "Abuse of process", "General", "Criminal Procedure Code 2010"),

    # -----------------------------------------------------------------------
    # STATUTORY INTERPRETATION
    # -----------------------------------------------------------------------
    "Statutory Interpretation — Construction of statutes":
        ("Statutory Interpretation", "Construction of statutes", "General principles", "Interpretation Act"),
    "Statutory Interpretation — Penal statutes":
        ("Statutory Interpretation", "Penal statutes", "Strict construction of penal statutes", "Interpretation Act"),
    "Statutory Interpretation — Interpretation Act":
        ("Statutory Interpretation", "Interpretation Act", "General", "Interpretation Act"),
    "Statutory Interpretation":
        ("Statutory Interpretation", "General", "General", "Interpretation Act"),

    # -----------------------------------------------------------------------
    # RES JUDICATA
    # -----------------------------------------------------------------------
    "Res Judicata — Issue estoppel":
        ("Civil Procedure", "Res judicata", "Issue estoppel", "Common law"),
    "Res Judicata — Extended doctrine":
        ("Civil Procedure", "Res judicata", "Extended doctrine of res judicata", "Common law"),
    "Res Judicata":
        ("Civil Procedure", "Res judicata", "General", "Common law"),

    # -----------------------------------------------------------------------
    # LEGAL PROFESSION
    # -----------------------------------------------------------------------
    "Legal Profession — Disciplinary proceedings":
        ("Legal Profession", "Disciplinary proceedings", "General", "Legal Profession Act"),
    "Legal Profession — Unauthorised person":
        ("Legal Profession", "Unauthorised practice", "Unauthorised person", "Legal Profession Act"),
    "Legal Profession":
        ("Legal Profession", "General", "General", "Legal Profession Act"),

    # -----------------------------------------------------------------------
    # MISCELLANEOUS
    # -----------------------------------------------------------------------
    "International Law — Criminal acts":
        ("International Law", "Criminal acts", "High seas", "Penal Code"),
    "Contempt of Court — Criminal contempt":
        ("Criminal Procedure", "Contempt of court", "Criminal contempt", "Administration of Justice (Protection) Act"),
    "Contempt of Court":
        ("Criminal Procedure", "Contempt of court", "General", "Administration of Justice (Protection) Act"),
    "Words and Phrases":
        ("Statutory Interpretation", "Words and Phrases", "General", "Interpretation Act"),
    "Offences — Attempted murder":
        ("Criminal Law", "Offences against person", "Attempted murder", "Penal Code ss 300, 511"),

}

# ---------------------------------------------------------------------------
# CRIMINAL-LAW AREA KEYWORDS
# Used for is_criminal_case() fallback detection
# ---------------------------------------------------------------------------
_CRIMINAL_AREA_KEYWORDS = {
    "criminal law",
    "criminal procedure",
    "criminal procedure and sentencing",
    "misuse of drugs",
    "arms offences",
    "prevention of corruption",
    "computer misuse",
    "cdsa",
    "immigration act",
    "road traffic act",
    "vandalism",
    "kidnapping act",
    "protection from harassment",
    "penal code",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _normalize_dashes(text: str) -> str:
    """Normalize en-dash, em-dash, double-dash, and hyphen to ' — ' for consistent splitting."""
    # Replace em dash (—) with standard separator
    text = text.replace("\u2014", " — ")
    # Replace double en-dash (––) before single en-dash
    text = text.replace("\u2013\u2013", " — ")
    # Replace en dash (–) with standard separator
    text = text.replace("\u2013", " — ")
    # Replace ' - ' (spaced hyphen) with standard separator
    text = re.sub(r"\s+-\s+", " — ", text)
    # Collapse multiple separators and extra whitespace
    text = re.sub(r"(\s*—\s*)+", " — ", text)
    return text.strip()


def classify_catchword(catchword: str):
    """
    Match a raw catchword string against the taxonomy using prefix matching.

    Returns:
        (area, topic, subtopic, statute, is_criminal, taxonomy_key)
        or None if no match found.
    """
    if not catchword or not catchword.strip():
        return None

    normalized = _normalize_dashes(catchword)

    # Try longest prefix match first (most specific), case-insensitive
    normalized_lower = normalized.lower()
    best_key = None
    best_len = 0
    for key in CRIMINAL_TAXONOMY:
        norm_key = _normalize_dashes(key)
        if normalized_lower.startswith(norm_key.lower()) and len(norm_key) > best_len:
            best_key = key
            best_len = len(norm_key)

    if best_key:
        area, topic, subtopic, statute = CRIMINAL_TAXONOMY[best_key]
        criminal = area in ("Criminal Law", "Criminal Procedure")
        return area, topic, subtopic, statute, criminal, best_key

    return None


def is_criminal_case(area_of_law: str) -> bool:
    """Return True if the area_of_law string indicates a criminal law matter."""
    if not area_of_law:
        return False
    lower = area_of_law.lower()
    return any(kw in lower for kw in _CRIMINAL_AREA_KEYWORDS)


def split_catchword(raw: str):
    """
    Split a raw catchword into (area, topic, subtopic) parts.
    Handles mixed dash types from elitigation.
    """
    normalized = _normalize_dashes(raw)
    parts = [p.strip() for p in normalized.split(" — ")]
    area = parts[0] if len(parts) > 0 else ""
    topic = parts[1] if len(parts) > 1 else ""
    subtopic = parts[2] if len(parts) > 2 else ""
    return area, topic, subtopic
