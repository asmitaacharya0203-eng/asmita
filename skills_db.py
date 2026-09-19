"""
skills_db.py
------------
Finance-industry skills taxonomy used by the Resume Matching Utility.

Structure:
    SKILLS[category][canonical_skill] = [aliases / spelling variants]

Matching rules (see extract_skills):
  * Matching is case-insensitive, except for short all-caps aliases (e.g. "SQL",
    "VBA", "AML", "DCF") which are matched case-sensitively to avoid false hits.
  * An alias only matches as a whole "word" (so "Java" will not match "JavaScript").

To extend the tool, simply add new skills / aliases to the dictionaries below.
"""

import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# 1. SKILL TAXONOMY
# --------------------------------------------------------------------------- #
SKILLS = {
    "Financial Analysis & Modelling": {
        "Financial Modelling": ["financial modelling", "financial modeling", "financial models", "3-statement model", "three statement model"],
        "DCF Valuation": ["DCF", "discounted cash flow"],
        "Valuation": ["valuation", "company valuation", "business valuation"],
        "Financial Statement Analysis": ["financial statement analysis", "analysis of financial statements", "financial statements analysis"],
        "Ratio Analysis": ["ratio analysis", "financial ratios", "dupont analysis"],
        "Budgeting & Forecasting": ["budgeting", "forecasting", "budget preparation", "rolling forecast", "annual operating plan"],
        "Variance Analysis": ["variance analysis", "variance reporting", "budget vs actual"],
        "FP&A": ["FP&A", "financial planning and analysis", "financial planning & analysis"],
        "Scenario & Sensitivity Analysis": ["sensitivity analysis", "scenario analysis", "scenario planning"],
        "Comparable Company Analysis": ["comparable company analysis", "comps analysis", "trading comps", "precedent transactions", "relative valuation"],
        "LBO Modelling": ["LBO", "leveraged buyout", "leveraged buy-out"],
        "Cash Flow Management": ["cash flow management", "cash flow forecasting", "working capital management", "working capital"],
        "MIS Reporting": ["MIS reporting", "MIS reports", "management reporting", "management information system"],
    },
    "Investment Banking & Capital Markets": {
        "Mergers & Acquisitions": ["M&A", "mergers and acquisitions", "mergers & acquisitions", "due diligence"],
        "Investment Banking": ["investment banking"],
        "Equity Research": ["equity research", "sell-side research", "buy-side research", "initiation report", "company research"],
        "Capital Markets": ["capital markets", "equity capital markets", "debt capital markets", "IPO", "ECM", "DCM"],
        "Fixed Income": ["fixed income", "bond valuation", "bonds", "yield curve", "duration and convexity"],
        "Derivatives": ["derivatives", "options pricing", "futures and options", "F&O", "swaps", "black-scholes", "black scholes"],
        "Portfolio Management": ["portfolio management", "portfolio construction", "portfolio optimisation", "portfolio optimization", "asset allocation"],
        "Wealth Management": ["wealth management", "private banking", "financial planning", "client advisory"],
        "Asset Management": ["asset management", "mutual funds", "mutual fund", "fund management"],
        "Trading": ["equity trading", "algorithmic trading", "trading strategies", "forex", "fx trading"],
        "Private Equity & VC": ["private equity", "venture capital", "deal sourcing", "term sheet"],
        "Pitch Books & CIMs": ["pitchbook", "pitch book", "pitch books", "information memorandum", "CIM", "teaser"],
        "Technical Analysis": ["technical analysis", "candlestick", "chart patterns"],
        "Fundamental Analysis": ["fundamental analysis", "equity valuation", "earnings analysis"],
    },
    "Risk, Compliance & Regulation": {
        "Risk Management": ["risk management", "enterprise risk", "risk assessment", "risk mitigation"],
        "Credit Risk": ["credit risk", "credit rating", "credit scoring", "probability of default", "PD/LGD", "expected credit loss", "ECL"],
        "Market Risk": ["market risk", "VaR", "value at risk", "value-at-risk"],
        "Operational Risk": ["operational risk", "operations risk", "RCSA"],
        "Stress Testing": ["stress testing", "stress tests"],
        "Basel Norms": ["Basel III", "Basel II", "Basel norms", "Basel"],
        "AML & KYC": ["AML", "KYC", "anti money laundering", "anti-money laundering", "know your customer", "fraud detection"],
        "Regulatory Compliance": ["regulatory compliance", "RBI", "SEBI", "IRDAI", "regulatory reporting", "compliance monitoring", "FEMA"],
        "Internal Audit": ["internal audit", "internal audits", "internal controls", "risk based audit", "risk-based audit", "audit planning"],
        "SOX / Internal Controls": ["SOX", "sarbanes-oxley", "sarbanes oxley", "ICFR", "controls testing"],
        "Corporate Governance": ["corporate governance", "ESG", "BRSR"],
    },
    "Accounting, Audit & Tax": {
        "Accounting": ["accounting", "bookkeeping", "book keeping", "general ledger", "journal entries"],
        "IFRS": ["IFRS"],
        "Ind AS": ["Ind AS", "IndAS", "Indian accounting standards"],
        "US GAAP": ["US GAAP", "GAAP"],
        "Statutory Audit": ["statutory audit", "statutory auditing", "external audit", "tax audit", "audit assurance"],
        "Taxation": ["taxation", "income tax", "direct tax", "indirect tax", "transfer pricing", "TDS", "tax compliance"],
        "GST": ["GST", "goods and services tax", "GSTR"],
        "Accounts Payable & Receivable": ["accounts payable", "accounts receivable", "AP/AR", "invoice processing", "vendor payments"],
        "Reconciliation": ["reconciliation", "reconciliations", "bank reconciliation", "account reconciliation"],
        "Month-end Close": ["month-end close", "month end close", "month-end closing", "financial close", "period end close"],
        "Consolidation": ["consolidation", "consolidated financial statements", "group reporting", "intercompany"],
        "Cost Accounting": ["cost accounting", "costing", "standard costing", "cost control", "management accounting"],
        "Payroll & Compliance Filing": ["payroll", "statutory filings", "ROC filings"],
    },
    "Banking, Lending & Treasury": {
        "Credit Analysis": ["credit analysis", "credit appraisal", "credit assessment", "credit underwriting", "credit memo"],
        "Loan Underwriting": ["loan underwriting", "underwriting", "loan processing", "loan origination", "retail lending", "corporate lending"],
        "Treasury Management": ["treasury", "treasury management", "liquidity management", "cash management", "ALM", "asset liability management"],
        "Trade Finance": ["trade finance", "letter of credit", "letters of credit", "bank guarantee", "LC/BG", "documentary credit"],
        "Corporate Banking": ["corporate banking", "relationship management", "commercial banking", "SME banking"],
        "Retail Banking": ["retail banking", "branch banking", "CASA", "cross-selling", "cross selling"],
        "Project Finance": ["project finance", "infrastructure finance", "debt syndication"],
        "Insurance": ["insurance", "actuarial", "claims management"],
        "Fintech & Payments": ["fintech", "digital payments", "UPI", "payment gateway", "lending platform"],
    },
    "Tools & Technology": {
        "Advanced Excel": ["advanced excel", "MS Excel", "Microsoft Excel", "Excel", "pivot tables", "pivot table", "vlookup", "xlookup", "index match", "power query"],
        "VBA / Macros": ["VBA", "macros", "excel macros"],
        "SQL": ["SQL", "MySQL", "PostgreSQL", "T-SQL", "PL/SQL"],
        "Python": ["Python", "pandas", "numpy", "scikit-learn"],
        "R Programming": ["R programming", "R language", "RStudio", "R Studio"],
        "Power BI": ["Power BI", "PowerBI"],
        "Tableau": ["Tableau"],
        "SAP": ["SAP", "SAP FICO", "SAP FI", "SAP S/4HANA", "S/4HANA"],
        "Oracle Financials": ["Oracle Financials", "Oracle EBS", "Oracle Fusion", "Oracle ERP", "Hyperion", "Oracle EPM"],
        "Tally / QuickBooks": ["Tally", "Tally ERP", "Tally Prime", "QuickBooks", "Zoho Books"],
        "Bloomberg Terminal": ["Bloomberg", "Bloomberg Terminal", "BBG"],
        "Capital IQ / FactSet / Refinitiv": ["Capital IQ", "CapIQ", "S&P Capital IQ", "FactSet", "Refinitiv", "Eikon", "Reuters Eikon", "Morningstar", "Screener.in"],
        "PowerPoint & Reporting": ["PowerPoint", "MS PowerPoint", "presentation decks", "board presentations", "MIS dashboards"],
        "Machine Learning / AI": ["machine learning", "AI/ML", "data science", "predictive modelling", "predictive modeling", "NLP"],
        "Data Analysis": ["data analysis", "data analytics", "data visualisation", "data visualization", "statistical analysis"],
        "Automation (RPA)": ["RPA", "UiPath", "process automation", "workflow automation"],
    },
    "Soft Skills": {
        "Communication": ["communication skills", "communication", "written communication", "verbal communication"],
        "Leadership": ["leadership", "team lead", "team leadership", "led a team", "managed a team"],
        "Stakeholder Management": ["stakeholder management", "client management", "stakeholder engagement", "client servicing"],
        "Analytical Thinking": ["analytical skills", "analytical thinking", "analytical ability", "critical thinking", "quantitative skills", "quantitative aptitude"],
        "Attention to Detail": ["attention to detail", "detail-oriented", "detail oriented", "accuracy"],
        "Problem Solving": ["problem solving", "problem-solving"],
        "Teamwork": ["teamwork", "team player", "collaboration", "cross-functional"],
        "Presentation Skills": ["presentation skills", "public speaking", "pitching"],
        "Time Management": ["time management", "meeting deadlines", "prioritisation", "prioritization"],
    },
}

# --------------------------------------------------------------------------- #
# 2. PROFESSIONAL CERTIFICATIONS / QUALIFICATIONS (relevant in finance)
# --------------------------------------------------------------------------- #
CERTIFICATIONS = {
    "CFA": ["CFA", "Chartered Financial Analyst", "CFA Level", "CFA Charterholder"],
    "FRM": ["FRM", "Financial Risk Manager"],
    "CA": ["Chartered Accountant", "ICAI", "CA Final", "CA Inter", "CA Qualified"],
    "CPA": ["CPA", "Certified Public Accountant"],
    "CMA": ["CMA", "Cost and Management Accountant", "Certified Management Accountant", "ICMAI"],
    "ACCA": ["ACCA"],
    "CFP": ["CFP", "Certified Financial Planner"],
    "CIA": ["Certified Internal Auditor", "CIA"],
    "CS": ["Company Secretary", "ICSI", "CS Executive", "CS Professional"],
    "NISM": ["NISM", "NISM Series"],
    "CAIIB": ["CAIIB", "JAIIB", "Certified Associate of Indian Institute of Bankers"],
    "CAIA": ["CAIA"],
    "PRM": ["PRM"],
    "CISA": ["CISA"],
    "CAMS": ["CAMS", "Certified Anti-Money Laundering Specialist"],
    "Bloomberg Market Concepts": ["Bloomberg Market Concepts", "BMC"],
    "FMVA / CFI": ["FMVA", "Corporate Finance Institute"],
}

# --------------------------------------------------------------------------- #
# 3. EDUCATION LEVELS  (higher number = higher qualification)
# --------------------------------------------------------------------------- #
EDUCATION_LEVELS = {
    0: "Not specified",
    1: "Higher Secondary (12th)",
    2: "Diploma",
    3: "Bachelor's degree",
    4: "Master's / MBA / Professional (CA, CFA, CPA...)",
    5: "Doctorate (PhD)",
}

EDUCATION_KEYWORDS = {
    5: ["ph.d", "phd", "doctorate", "doctoral"],
    4: ["mba", "pgdm", "pgpm", "m.com", "mcom", "m.sc", "msc", "m.a.", "m.tech", "mtech", "master of", "masters in",
        "post graduate", "postgraduate", "post-graduate", "mfc", "mfm", "chartered accountant", "chartered financial analyst",
        "cpa", "acca", "cost and management accountant"],
    3: ["b.com", "bcom", "bba", "bms", "b.sc", "bsc", "b.tech", "btech", "b.e.", "b.a.", "bachelor", "graduate in",
        "b.b.a", "bbi", "bfm", "bafa", "honours", "hons"],
    2: ["diploma"],
    1: ["12th", "hsc", "higher secondary", "xii", "intermediate", "senior secondary"],
}

FINANCE_DEGREE_HINTS = ["finance", "accounting", "accountancy", "accountant", "commerce", "b.com", "m.com", "bcom",
                        "mcom", "b.acc", "economics", "banking", "actuarial", "statistics", "mathematics",
                        "financial", "acca", "chartered"]

# words that show a qualification is still being pursued / only partly completed
_IN_PROGRESS = re.compile(
    r"pursuing|persuing|candidate|appearing|enrolled|in progress|aspirant|registered|passed|cleared|"
    r"level\s*(?:i{1,3}|[123])\b|part\s*(?:i{1,3}|[123])\b", re.IGNORECASE)
_COMPLETED = re.compile(r"charterholder|charter holder|qualified", re.IGNORECASE)
_ONGOING_EDU = re.compile(r"pursuing|persuing|expected|ongoing|currently|in progress|\(present\)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# 4. HELPERS
# --------------------------------------------------------------------------- #
def all_skills():
    """Flat list of every canonical skill name in the taxonomy."""
    return [skill for group in SKILLS.values() for skill in group]


def skill_category(skill):
    """Return the category to which a canonical skill belongs."""
    for category, group in SKILLS.items():
        if skill in group:
            return category
    return "Other"


# ordinary words that double as skill names -> only match with the capital letter
_FORCE_CASE_SENSITIVE = {"Excel", "Tally"}


def _is_case_sensitive(alias):
    """Short all-caps aliases (SQL, VBA, DCF, AML ...) must match case-sensitively."""
    if alias in _FORCE_CASE_SENSITIVE:
        return True
    letters = re.sub(r"[^A-Za-z]", "", alias)
    return len(letters) <= 5 and letters.isupper()


@lru_cache(maxsize=None)
def _compile_skill(variants):
    """
    Build ONE regex per skill from all its aliases so overlapping aliases
    (e.g. "DCF" and "DCF valuation") are not double counted.
    """
    parts = []
    for alias in sorted(variants, key=len, reverse=True):
        esc = re.escape(alias)
        parts.append(esc if _is_case_sensitive(alias) else f"(?i:{esc})")
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])")


def _find_in_taxonomy(text, taxonomy, include_canonical=True):
    """Return {canonical_name: number_of_mentions} for the given taxonomy."""
    found = {}
    for canonical, aliases in taxonomy.items():
        # for skills the canonical name itself is also a valid alias; for certifications
        # it is not (short codes such as "CA" / "CS" are too ambiguous on their own)
        variants = frozenset({canonical, *aliases}) if include_canonical else frozenset(aliases)
        count = len(_compile_skill(variants).findall(text))
        if count:
            found[canonical] = count
    return found


def extract_skills(text):
    """
    Extract skills from free text.
    Returns {canonical_skill: mentions}. Mentions are later used to break ties
    and to show how prominent a skill is in the resume.
    """
    found = {}
    for group in SKILLS.values():
        found.update(_find_in_taxonomy(text, group))
    return found


def extract_certifications(text):
    """
    Return (held, in_progress): two sorted lists of certification names.
    A certification counts as 'in progress' when the line that mentions it says pursuing / candidate /
    Level II / Part I / passed ... (e.g. "CFA Level II candidate"), unless it also says Charterholder / qualified.
    """
    held, progress = set(), set()
    for line in text.splitlines():
        found = _find_in_taxonomy(line, CERTIFICATIONS, include_canonical=False)
        if not found:
            continue
        partial = bool(_IN_PROGRESS.search(line)) and not _COMPLETED.search(line)
        (progress if partial else held).update(found)
    return sorted(held), sorted(progress - held)


def extract_education_level(text):
    """Return the highest COMPLETED education level (0-5) detected in the text."""
    best = 0
    for line in text.splitlines():
        if _ONGOING_EDU.search(line):           # degree still being pursued -> not counted
            continue
        lowered = line.lower()
        for level, words in EDUCATION_KEYWORDS.items():
            for w in words:
                if level > best and re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", lowered):
                    best = level
    return best


def has_finance_degree(text):
    """True if a finance / commerce / economics style degree is mentioned."""
    lowered = text.lower()
    return any(h in lowered for h in FINANCE_DEGREE_HINTS)


def canonical_lookup(name):
    """Return the canonical skill name for user input (case-insensitive), or None."""
    for skill in all_skills():
        if skill.lower() == name.strip().lower():
            return skill
    return None
