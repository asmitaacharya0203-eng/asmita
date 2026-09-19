"""
matcher.py
----------
Scoring / ranking engine.

Each candidate gets a Match Score (0-100) built from five components:

    1. Must-have skills coverage      (default weight 40)
    2. Nice-to-have skills coverage   (default weight 15)
    3. Text similarity (TF-IDF cosine between the job description and the resume)  (default weight 15)
    4. Experience fit                 (default weight 15)
    5. Education & certification fit  (default weight 15)

Weights can be changed by the user in the app and are re-normalised to add up to 100.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import skills_db

DEFAULT_WEIGHTS = {
    "must_have": 40,
    "nice_to_have": 15,
    "text_similarity": 15,
    "experience": 15,
    "education_cert": 15,
}

# A skill that is only mentioned once (e.g. just listed under "Skills") earns slightly less credit than a
# skill that is mentioned repeatedly (i.e. actually used in experience / projects).
SINGLE_MENTION_CREDIT = 0.85

# Cosine similarity between a JD and a resume is rarely above ~0.35, so 0.35 is treated as a "full" score.
SIMILARITY_CAP = 0.35


# --------------------------------------------------------------------------- #
# 1. JOB REQUIREMENT
# --------------------------------------------------------------------------- #
@dataclass
class JobRequirement:
    title: str
    description: str = ""
    must_have: list = field(default_factory=list)
    nice_to_have: list = field(default_factory=list)
    preferred_certs: list = field(default_factory=list)   # candidate needs ANY ONE of these
    min_experience: float = 0.0
    max_experience: float = 0.0                           # 0 = no upper limit
    min_education: int = 0                                # see skills_db.EDUCATION_LEVELS
    department: str = ""
    location: str = ""

    def jd_text(self):
        """All the text that represents the job, used for TF-IDF similarity."""
        return " ".join([self.title, self.description, " ".join(self.must_have) * 2,
                         " ".join(self.nice_to_have), " ".join(self.preferred_certs)])


def _split(value):
    return [v.strip() for v in str(value).split(";") if v.strip() and str(value) != "nan"]


def load_job_roles(csv_path):
    """Load the job-requirement master data (job_roles.csv) -> list[JobRequirement]."""
    df = pd.read_csv(csv_path).fillna("")
    jobs = []
    for _, r in df.iterrows():
        jobs.append(JobRequirement(
            title=r["title"],
            description=r.get("description", ""),
            must_have=_split(r.get("must_have_skills", "")),
            nice_to_have=_split(r.get("nice_to_have_skills", "")),
            preferred_certs=_split(r.get("preferred_certifications", "")),
            min_experience=float(r.get("min_experience_years") or 0),
            max_experience=float(r.get("max_experience_years") or 0),
            min_education=int(r.get("min_education_level") or 0),
            department=r.get("department", ""),
            location=r.get("location", ""),
        ))
    return jobs


# --------------------------------------------------------------------------- #
# 2. TEXT SIMILARITY (TF-IDF cosine)
# --------------------------------------------------------------------------- #
def _tokenize(text):
    return re.findall(r"[a-z][a-z&+#\-]{1,}", text.lower())


def _fallback_similarity(jd_text, resume_texts):
    """Plain bag-of-words cosine similarity - used only if scikit-learn is not installed."""
    def cosine(a, b):
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / den if den else 0.0
    jd = Counter(_tokenize(jd_text))
    return [cosine(jd, Counter(_tokenize(t))) for t in resume_texts]


def text_similarities(jd_text, resume_texts):
    """Cosine similarity (0-1) of every resume to the job description using TF-IDF vectors."""
    if not resume_texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([jd_text] + list(resume_texts))
        return [float(x) for x in cosine_similarity(matrix[0:1], matrix[1:])[0]]
    except ImportError:
        return _fallback_similarity(jd_text, resume_texts)


# --------------------------------------------------------------------------- #
# 3. COMPONENT SCORES
# --------------------------------------------------------------------------- #
def _skill_coverage(required, candidate_skills):
    """Return (score 0-1, matched list, missing list). Score gives less credit to single-mention skills."""
    if not required:
        return 1.0, [], []
    matched, missing, credit = [], [], 0.0
    for skill in required:
        mentions = candidate_skills.get(skill, 0)
        if mentions:
            matched.append(skill)
            credit += 1.0 if mentions >= 2 else SINGLE_MENTION_CREDIT
        else:
            missing.append(skill)
    return credit / len(required), matched, missing


def _experience_score(years, job):
    if job.min_experience <= 0:
        score = 1.0
    else:
        score = min(1.0, years / job.min_experience)
    over_qualified = bool(job.max_experience and years > job.max_experience)
    if over_qualified:
        score = min(score, 0.7)           # still shortlistable, but flagged and slightly penalised
    return score, over_qualified


def _education_cert_score(candidate, job):
    # --- education (80% qualification level, 20% finance-related degree) ---
    if job.min_education <= 0:
        level_fit = 1.0
    else:
        level_fit = min(1.0, candidate["education_level"] / job.min_education)
    edu = 0.8 * level_fit + 0.2 * (1.0 if candidate["finance_degree"] else 0.0)

    # --- certifications (candidate needs ANY ONE of the preferred certs) ---
    certs = candidate["certifications"]
    in_progress = candidate.get("certs_in_progress", [])
    if not job.preferred_certs:
        cert = 1.0
    elif set(certs) & set(job.preferred_certs):
        cert = 1.0
    elif set(in_progress) & set(job.preferred_certs):
        cert = 0.6                        # e.g. "CFA Level II candidate"
    elif certs:
        cert = 0.4                        # holds some other finance qualification
    else:
        cert = 0.0
    return 0.5 * edu + 0.5 * cert, sorted(set(certs) & set(job.preferred_certs))


# --------------------------------------------------------------------------- #
# 4. RANKING
# --------------------------------------------------------------------------- #
def _normalise(weights):
    total = sum(weights.values()) or 1
    return {k: v / total for k, v in weights.items()}


def _remarks(c, job, matched_must, missing_must, exp_years, over_qualified, matched_certs):
    notes = []
    if job.must_have:
        notes.append(f"{len(matched_must)}/{len(job.must_have)} must-have skills")
    if missing_must:
        notes.append("missing: " + ", ".join(missing_must[:4]) + ("..." if len(missing_must) > 4 else ""))
    if job.min_experience:
        gap = "meets" if exp_years >= job.min_experience else "below"
        notes.append(f"{exp_years:g} yrs exp ({gap} {job.min_experience:g}+ required)")
    if over_qualified:
        notes.append("may be over-qualified")
    if matched_certs:
        notes.append("certified: " + ", ".join(matched_certs))
    return "; ".join(notes)


def rank_candidates(candidates, job, weights=None, shortlist_threshold=60.0, top_n=10, min_must_coverage=0.5):
    """
    Score, rank and label every candidate against `job`.

    Decision logic
    --------------
    * Must-have coverage below `min_must_coverage`      -> "Rejected - skill gap"   (knock-out rule)
    * Score >= shortlist_threshold and within top_n      -> "Shortlisted"
    * Score >= shortlist_threshold but beyond top_n      -> "Waitlist"
    * Score >= threshold - 15                            -> "Review"
    * otherwise                                          -> "Not shortlisted"

    Returns a list of result dicts sorted best-first.
    """
    if not candidates:
        return []
    w = _normalise(weights or DEFAULT_WEIGHTS)
    sims = text_similarities(job.jd_text(), [c["text"] for c in candidates])

    results = []
    for cand, sim in zip(candidates, sims):
        must_score, matched_must, missing_must = _skill_coverage(job.must_have, cand["skills"])
        nice_score, matched_nice, missing_nice = _skill_coverage(job.nice_to_have, cand["skills"])
        sim_score = min(1.0, sim / SIMILARITY_CAP)
        exp_score, over_q = _experience_score(cand["experience_years"], job)
        edu_cert_score, matched_certs = _education_cert_score(cand, job)

        total = 100 * (w["must_have"] * must_score + w["nice_to_have"] * nice_score +
                       w["text_similarity"] * sim_score + w["experience"] * exp_score +
                       w["education_cert"] * edu_cert_score)

        must_cov = len(matched_must) / len(job.must_have) if job.must_have else 1.0
        extra_skills = [s for s in cand["skills"] if s not in job.must_have and s not in job.nice_to_have]

        results.append({
            "name": cand["name"], "file": cand["file"], "email": cand["email"], "phone": cand["phone"],
            "score": round(total, 1),
            "must_have_score": round(must_score * 100, 1),
            "nice_to_have_score": round(nice_score * 100, 1),
            "similarity_score": round(sim_score * 100, 1),
            "experience_score": round(exp_score * 100, 1),
            "edu_cert_score": round(edu_cert_score * 100, 1),
            "must_coverage": must_cov,
            "matched_must": matched_must, "missing_must": missing_must,
            "matched_nice": matched_nice, "missing_nice": missing_nice,
            "extra_skills": sorted(extra_skills, key=lambda s: -cand["skills"][s]),
            "all_skills": cand["skills"],
            "experience_years": cand["experience_years"],
            "education_level": cand["education_level"],
            "education_label": skills_db.EDUCATION_LEVELS.get(cand["education_level"], "Not specified"),
            "certifications": cand["certifications"],
            "certs_in_progress": cand.get("certs_in_progress", []),
            "over_qualified": over_q,
            "remarks": _remarks(cand, job, matched_must, missing_must, cand["experience_years"], over_q, matched_certs),
            "text": cand["text"],
        })

    # Best first. Candidates who fail the must-have knock-out rule always sit below those who pass it;
    # ties are broken by must-have coverage, then by experience.
    results.sort(key=lambda r: (r["must_coverage"] < min_must_coverage, -r["score"], -r["must_coverage"],
                                -r["experience_years"]))

    shortlisted = 0
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
        if r["must_coverage"] < min_must_coverage:
            r["decision"] = "Rejected - skill gap"
        elif r["score"] >= shortlist_threshold:
            shortlisted += 1
            r["decision"] = "Shortlisted" if shortlisted <= top_n else "Waitlist"
        elif r["score"] >= shortlist_threshold - 15:
            r["decision"] = "Review"
        else:
            r["decision"] = "Not shortlisted"
    return results


def results_to_dataframe(results):
    """Flatten the result dicts into a table suitable for display / export."""
    rows = []
    for r in results:
        rows.append({
            "Rank": r["rank"],
            "Candidate": r["name"],
            "Match Score": r["score"],
            "Decision": r["decision"],
            "Must-have Skills": f'{len(r["matched_must"])}/{len(r["matched_must"]) + len(r["missing_must"])}',
            "Missing Must-haves": ", ".join(r["missing_must"]),
            "Experience (yrs)": r["experience_years"],
            "Education": r["education_label"],
            "Certifications": ", ".join(r["certifications"] + [f"{c} (in progress)" for c in r["certs_in_progress"]]),
            "Email": r["email"],
            "Phone": r["phone"],
            "Resume File": r["file"],
            "Remarks": r["remarks"],
        })
    return pd.DataFrame(rows)


def skill_matrix(results, job):
    """Candidate x required-skill grid (1 = present, 0 = missing) for the skill-gap heat-map."""
    required = job.must_have + [s for s in job.nice_to_have if s not in job.must_have]
    data = {r["name"]: [1 if r["all_skills"].get(s) else 0 for s in required] for r in results}
    return pd.DataFrame(data, index=required).T


def pool_skill_frequency(candidates, top=15):
    """How often each skill appears across ALL resumes (talent-pool insight for HR)."""
    counter = Counter()
    for c in candidates:
        counter.update(c["skills"].keys())
    return pd.DataFrame(counter.most_common(top), columns=["Skill", "Candidates"]).set_index("Skill")


def export_shortlist(results, out_dir, job_title, only_shortlisted=True):
    """Save results to Excel (and CSV) in `out_dir`. Returns the Excel path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)
    if only_shortlisted:
        df = df[df["Decision"] == "Shortlisted"]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", job_title).strip("_")
    xlsx = out_dir / f"shortlist_{safe}.xlsx"
    df.to_excel(xlsx, index=False)
    df.to_csv(out_dir / f"shortlist_{safe}.csv", index=False)
    return xlsx
