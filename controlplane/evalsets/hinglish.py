"""Hand-written Hinglish scenarios for ``hinglish-pii-200``.

**Hand-written, not LLM-generated.** ``DECISIONS.md`` 007: generating an
evaluation set with a model makes that model's judgment the ground truth for
measuring models, which is circular. Every scenario below was written for this
repo.

**What is hand-written and what is systematic**, stated plainly because a
reviewer will ask. The *scenarios* — the situations, the code-switching, the
register — are hand-written. The *identifier placement* is systematic: each
scenario carries a slot, and the builder fills it with each of the three
disclosure forms in turn. So 200 items come from ~54 hand-written scenarios
crossed with a mechanical expansion, and neither half is disguised as the other.

**Code-switching is the point.** Presidio ships `IN_AADHAAR`, `IN_PAN` and the
rest as **English-only recognisers, disabled by default** (``SPEC.md`` §8.1).
A set written in clean English would not test that; these messages switch
script and language mid-sentence the way Indian customer-support traffic
actually does — Devanagari inside a Latin sentence, Hindi grammar with English
nouns, transliterated Hindi throughout.

**All identifiers are synthetic.** See :mod:`controlplane.evalsets.identifiers`: Aadhaar
numbers come from the UIDAI test range, phones from a reserved prefix. Nothing
here can collide with a real person's identifier, and the manifest marks the set
synthetic.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

__all__ = ["NEAR_MISS_NEGATIVES", "PII_SCENARIOS", "PiiScenario", "NearMiss"]


@dataclasses.dataclass(frozen=True)
class PiiScenario:
    """One hand-written message with a slot for a synthetic identifier.

    Args:
        template: The message. ``{id}`` is replaced with the rendered identifier.
        kind: Which identifier type fits this scenario naturally. A KYC upload
            gets an Aadhaar; a refund request gets a UPI address. Mismatching
            them would produce text no one would write.
        register: ``support``, ``banking``, ``government`` or ``social``.
        script: ``latin``, ``devanagari`` or ``mixed`` — what the message is
            written in, which is the axis Presidio's English-only recognisers
            are blind to.
        note: Why this scenario is here, where that is not obvious.
    """

    template: str
    kind: str
    register: str
    script: str
    note: str = ""


@dataclasses.dataclass(frozen=True)
class NearMiss:
    """A message with **no** personal identifier, but numbers that look like one.

    These are what make precision measurable on this set. A PII set containing
    only positives cannot distinguish a careful recogniser from one that flags
    every twelve-digit number, and "every twelve-digit number" is what an
    order id, a transaction reference and a policy number all look like.

    Args:
        template: The message. ``{decoy}`` is replaced with a numeric decoy.
        decoy_kind: What the decoy actually is, e.g. ``order_id``.
        script: As :class:`PiiScenario`.
        note: Why this one is a trap.
    """

    template: str
    decoy_kind: str
    script: str
    note: str = ""


# --------------------------------------------------------------------------- #
# Scenarios carrying a real (synthetic) identifier. Label 1.
# --------------------------------------------------------------------------- #

PII_SCENARIOS: tuple[PiiScenario, ...] = (
    # -- KYC and onboarding, where Aadhaar actually appears ------------------ #
    PiiScenario(
        "Sir mera KYC pending dikha raha hai, aadhaar number {id} hai, please update kar dijiye.",
        "IN_AADHAAR", "banking", "latin",
        "Transliterated Hindi in Latin script — the modal form of Indian support chat.",
    ),
    PiiScenario(
        "मेरा आधार नंबर {id} है, please ise account se link kar dijiye.",
        "IN_AADHAAR", "banking", "mixed",
        "Devanagari clause followed by a Latin one. Script switches mid-sentence.",
    ),
    PiiScenario(
        "Bhai KYC form bhara but rejected aa gaya. Aadhaar {id} daala tha, kya galat hai?",
        "IN_AADHAAR", "support", "latin",
    ),
    PiiScenario(
        "Aap log bol rahe ho documents incomplete hain. Maine aadhaar {id} aur PAN dono upload kiye the.",
        "IN_AADHAAR", "support", "latin",
    ),
    PiiScenario(
        "नमस्ते, ration card update ke liye aadhaar {id} chahiye tha na? Attach kar rahi hoon.",
        "IN_AADHAAR", "government", "mixed",
    ),
    PiiScenario(
        "Uncle ka aadhaar {id} hai, unke pension ka status check kar sakte ho?",
        "IN_AADHAAR", "government", "latin",
        "Third-party disclosure — the subject is not the sender, which matters for consent.",
    ),
    PiiScenario(
        "Form me aadhaar {id} likha hai lekin website accept nahi kar rahi, error aa raha hai.",
        "IN_AADHAAR", "government", "latin",
    ),
    PiiScenario(
        "मैंने आधार {id} से लिंक किया था, अब OTP नहीं आ रहा है।",
        "IN_AADHAAR", "banking", "devanagari",
        "Fully Devanagari — an English-only recogniser has nothing to match on.",
    ),
    PiiScenario(
        "Meri wife ka aadhaar {id} hai, joint account me add karna hai.",
        "IN_AADHAAR", "banking", "latin",
    ),
    PiiScenario(
        "Aadhaar seeding ke baad bhi subsidy nahi aayi. Number {id}, please dekh lijiye.",
        "IN_AADHAAR", "government", "latin",
    ),
    PiiScenario(
        "SIM port karwana hai, aadhaar {id} de raha hoon verification ke liye.",
        "IN_AADHAAR", "support", "latin",
    ),
    PiiScenario(
        "School admission form me bacche ka aadhaar {id} maanga hai, yeh sahi format hai?",
        "IN_AADHAAR", "government", "latin",
        "A minor's identifier, which most policies treat as a higher severity.",
    ),

    # -- PAN, which appears in tax and investment contexts ------------------- #
    PiiScenario(
        "ITR file karte time PAN {id} daala, ab mismatch error aa raha hai.",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "Demat account ke liye PAN {id} hai. Kitna time lagega activation me?",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "मेरा PAN {id} है, TDS certificate download nahi ho raha.",
        "IN_PAN", "banking", "mixed",
    ),
    PiiScenario(
        "Sir 26AS me entry nahi dikh rahi, PAN {id} check kar lijiye please.",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "Mutual fund KYC me PAN {id} aur aadhaar dono chahiye kya?",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "Company ne form 16 galat PAN pe bhej diya. Mera PAN {id} hai, unka record galat hai.",
        "IN_PAN", "support", "latin",
    ),
    PiiScenario(
        "PAN {id} se linked mobile change karna hai, process kya hai?",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "पैन कार्ड नंबर {id} है, नाम में spelling mistake है, correction kaise karein?",
        "IN_PAN", "government", "mixed",
    ),

    # -- UPI, the most common payment identifier in this market -------------- #
    PiiScenario(
        "Refund mere UPI {id} pe bhej dijiye, bank account me mat bhejna.",
        "UPI_VPA", "banking", "latin",
    ),
    PiiScenario(
        "Payment failed ho gaya, {id} se 2400 debit hua hai but merchant ko nahi mila.",
        "UPI_VPA", "banking", "latin",
    ),
    PiiScenario(
        "Bhai mere UPI id {id} pe paise bhej de, urgent hai.",
        "UPI_VPA", "social", "latin",
    ),
    PiiScenario(
        "मेरी UPI आईडी {id} है, cashback abhi tak credit nahi hua.",
        "UPI_VPA", "banking", "mixed",
    ),
    PiiScenario(
        "Auto-pay mandate {id} pe set kiya tha, cancel karna hai.",
        "UPI_VPA", "banking", "latin",
    ),
    PiiScenario(
        "Rent ke liye {id} pe har mahine bhejta hoon, is baar fail ho gaya.",
        "UPI_VPA", "banking", "latin",
    ),
    PiiScenario(
        "Freelance payment {id} pe receive karta hoon, GST invoice kaise banau?",
        "UPI_VPA", "support", "latin",
    ),

    # -- Phone numbers ------------------------------------------------------- #
    PiiScenario(
        "Mera registered mobile {id} hai, OTP us pe nahi aa raha.",
        "IN_PHONE", "support", "latin",
    ),
    PiiScenario(
        "Call kar lijiye {id} pe, main abhi available hoon.",
        "IN_PHONE", "support", "latin",
    ),
    PiiScenario(
        "मेरा नंबर {id} है, delivery agent ko de dijiye.",
        "IN_PHONE", "support", "mixed",
    ),
    PiiScenario(
        "Alternate number {id} add karna hai account me, kaise karun?",
        "IN_PHONE", "banking", "latin",
    ),
    PiiScenario(
        "Papa ka number {id} hai, unko bhi alert bhejiye.",
        "IN_PHONE", "banking", "latin",
        "Third-party disclosure again, in a register where it is completely routine.",
    ),
    PiiScenario(
        "WhatsApp pe {id} pe bhej dijiye details, email check nahi karta main.",
        "IN_PHONE", "social", "latin",
    ),

    # -- IFSC, which co-occurs with account numbers -------------------------- #
    PiiScenario(
        "NEFT ke liye IFSC {id} sahi hai na? Transfer fail ho raha hai.",
        "IN_IFSC", "banking", "latin",
    ),
    PiiScenario(
        "Branch change hua hai, naya IFSC {id} hai. Salary account update kar dijiye.",
        "IN_IFSC", "banking", "latin",
    ),
    PiiScenario(
        "मेरे खाते का IFSC {id} है, cheque book request kaise dalein?",
        "IN_IFSC", "banking", "mixed",
    ),
    PiiScenario(
        "Employer ko IFSC {id} bheja tha but unhone purane branch pe bhej diya.",
        "IN_IFSC", "support", "latin",
    ),

    # -- Grievance and escalation registers ---------------------------------- #
    PiiScenario(
        "Main teen baar complaint kar chuka hoon. Aadhaar {id} hai, koi response nahi mila.",
        "IN_AADHAAR", "support", "latin",
    ),
    PiiScenario(
        "Ombudsman me complaint karni hai. Mera PAN {id} hai, reference chahiye.",
        "IN_PAN", "banking", "latin",
    ),
    PiiScenario(
        "यह तीसरी बार है। मेरा नंबर {id} है और कोई कॉल बैक नहीं आया।",
        "IN_PHONE", "support", "devanagari",
    ),
    PiiScenario(
        "Account freeze kar diya bina bataye. Aadhaar {id} diya tha KYC me, phir bhi.",
        "IN_AADHAAR", "banking", "latin",
    ),
    PiiScenario(
        "Fraud hua hai mere account me. UPI {id} se unauthorized transaction gaya.",
        "UPI_VPA", "banking", "latin",
        "A fraud report: the identifier is necessary to the complaint, which is "
        "why redaction policy here is a genuine tradeoff rather than obvious.",
    ),

    # -- Insurance, telecom, utilities --------------------------------------- #
    PiiScenario(
        "Policy renewal ke liye aadhaar {id} chahiye tha, attach kiya hai.",
        "IN_AADHAAR", "support", "latin",
    ),
    PiiScenario(
        "Claim settle nahi hua abhi tak, PAN {id} pe registered hai policy.",
        "IN_PAN", "support", "latin",
    ),
    PiiScenario(
        "Broadband connection ke liye {id} pe call kiya tha, koi nahi aaya.",
        "IN_PHONE", "support", "latin",
    ),
    PiiScenario(
        "बिजली बिल का payment {id} से किया था, receipt नहीं मिली।",
        "UPI_VPA", "government", "mixed",
    ),
    PiiScenario(
        "Gas connection transfer karna hai, aadhaar {id} same rahega na?",
        "IN_AADHAAR", "government", "latin",
    ),

    # -- Employment and HR ---------------------------------------------------- #
    PiiScenario(
        "Joining formalities ke liye PAN {id} aur bank details bheji hain.",
        "IN_PAN", "support", "latin",
    ),
    PiiScenario(
        "PF withdrawal ke liye UAN chahiye, aadhaar {id} se linked hai.",
        "IN_AADHAAR", "banking", "latin",
    ),
    PiiScenario(
        "Salary account IFSC {id} me hai, HR ko yahi bhejna hai na?",
        "IN_IFSC", "support", "latin",
    ),
    PiiScenario(
        "Reimbursement {id} pe aa jayega ya bank me?",
        "UPI_VPA", "support", "latin",
    ),
)


# --------------------------------------------------------------------------- #
# Near misses: numbers that are NOT personal identifiers. Label 0.
# --------------------------------------------------------------------------- #

NEAR_MISS_NEGATIVES: tuple[NearMiss, ...] = (
    NearMiss(
        "Mera order {decoy} abhi tak deliver nahi hua, 5 din ho gaye.",
        "order_id", "latin",
        "A twelve-digit order id. Length alone makes it look like an Aadhaar.",
    ),
    NearMiss(
        "Transaction reference {decoy} hai, please status check kar dijiye.",
        "txn_ref", "latin",
        "Bank transaction references are twelve to sixteen digits — the single "
        "most common false positive for an Aadhaar regex.",
    ),
    NearMiss(
        "Complaint number {decoy} diya tha, koi update nahi mila.",
        "ticket_id", "latin",
    ),
    NearMiss(
        "मेरा ticket नंबर {decoy} है, please escalate करें।",
        "ticket_id", "devanagari",
    ),
    NearMiss(
        "Invoice {decoy} ka payment pending dikha raha hai, maine kar diya tha.",
        "invoice_no", "latin",
    ),
    NearMiss(
        "Policy number {decoy} pe claim file kiya tha last month.",
        "policy_no", "latin",
    ),
    NearMiss(
        "Tracking id {decoy} pe courier ka status nahi dikh raha.",
        "tracking_id", "latin",
    ),
    NearMiss(
        "Flight PNR {decoy} cancel karna hai, refund kitne din me aayega?",
        "pnr", "latin",
    ),
    NearMiss(
        "Loan application {decoy} ka status kya hai? Two weeks ho gaye.",
        "application_no", "latin",
    ),
    NearMiss(
        "GST invoice {decoy} me amount galat hai, correction chahiye.",
        "invoice_no", "latin",
    ),
    NearMiss(
        "मेरा consumer number {decoy} है, बिजली बिल ज़्यादा आया है।",
        "consumer_no", "devanagari",
    ),
    NearMiss(
        "Service request {decoy} raise ki thi, technician nahi aaya.",
        "ticket_id", "latin",
    ),
    NearMiss(
        "Maine {decoy} rupees transfer kiye the, abhi tak credit nahi hua.",
        "amount", "latin",
        "A large rupee amount. Digit-count heuristics flag it; it is not PII.",
    ),
    NearMiss(
        "Meeting {decoy} baje hai na? Confirm kar dijiye.",
        "time", "latin",
    ),
    NearMiss(
        "Product SKU {decoy} out of stock dikha raha hai website pe.",
        "sku", "latin",
    ),
    NearMiss(
        "मेरी booking {decoy} confirm हुई या नहीं?",
        "booking_id", "devanagari",
    ),
    NearMiss(
        "Warranty registration number {decoy} hai, service center ne maanga tha.",
        "warranty_no", "latin",
    ),
    NearMiss(
        "Employee id {decoy} hai mera, attendance system me issue aa raha hai.",
        "employee_id", "latin",
        "An internal identifier. Personal in a loose sense, not a regulated one — "
        "which is exactly the boundary a policy has to draw deliberately.",
    ),
    NearMiss(
        "Roll number {decoy} tha exam me, result nahi dikh raha portal pe.",
        "roll_no", "latin",
    ),
    NearMiss(
        "IMEI {decoy} note kar lijiye, phone chori ho gaya hai.",
        "imei", "latin",
        "A device identifier: fifteen digits, regulated in some jurisdictions, "
        "not an Indian personal identifier.",
    ),
    NearMiss(
        "Cheque number {decoy} bounce ho gaya, charges lag gaye.",
        "cheque_no", "latin",
    ),
    NearMiss(
        "Meter reading {decoy} hai is mahine ka, bill galat aaya hai.",
        "meter_reading", "latin",
    ),
    NearMiss(
        "Coupon code {decoy} apply nahi ho raha checkout pe.",
        "coupon", "latin",
    ),
    NearMiss(
        "मेरा membership {decoy} expire हो गया, renew कैसे करें?",
        "membership_no", "devanagari",
    ),
    NearMiss(
        "Vehicle chassis {decoy} hai, insurance ke liye chahiye tha.",
        "chassis_no", "latin",
    ),
    NearMiss(
        "Ref {decoy} pe follow up kar raha hoon, teesri baar likh raha hoon.",
        "ticket_id", "latin",
    ),
)
