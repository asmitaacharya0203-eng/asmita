"""
app.py  -  AI Based Resume Matching Utility (Finance Industry)
==============================================================
Run with:   streamlit run app.py

Workflow
    1. Choose (or create) a job requirement
    2. Point the tool at a folder of resumes (PDF / TXT / DOCX) or upload files
    3. Click "Run matching"  ->  resumes are parsed, finance skills extracted, candidates scored & ranked
    4. Review the shortlist, skill-gap matrix and per-candidate detail; download the shortlist as Excel / CSV
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import matcher
import resume_parser
import skills_db

BASE_DIR = Path(__file__).parent
DEFAULT_RESUME_DIR = BASE_DIR / "resumes"
ROLES_CSV = BASE_DIR / "data" / "job_roles.csv"
OUTPUT_DIR = BASE_DIR / "output"

DECISION_ICON = {
    "Shortlisted": "✅ Shortlisted",
    "Waitlist": "⏳ Waitlist",
    "Review": "🔎 Review",
    "Not shortlisted": "➖ Not shortlisted",
    "Rejected - skill gap": "⛔ Rejected - skill gap",
}

st.set_page_config(page_title="AI Resume Matcher - Finance", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_roles():
    return matcher.load_job_roles(ROLES_CSV)


def folder_signature(folder):
    """Changes whenever a resume is added / removed / edited, so the cache refreshes automatically."""
    p = Path(folder).expanduser()
    if not p.is_dir():
        return ()
    return tuple((f.name, f.stat().st_size, int(f.stat().st_mtime))
                 for f in sorted(p.rglob("*")) if f.suffix.lower() in resume_parser.SUPPORTED_EXTENSIONS)


@st.cache_data(show_spinner=False)
def cached_folder_load(folder, signature):
    return resume_parser.load_resumes_from_folder(folder)


def to_excel_bytes(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar: job requirement
# --------------------------------------------------------------------------- #
def _suggest_skills_from_description():
    """Callback for the 'Auto-detect skills' button in the custom-requirement form."""
    found = skills_db.extract_skills(st.session_state.get("custom_desc", ""))
    ordered = [s for s, _ in sorted(found.items(), key=lambda kv: -kv[1])]
    st.session_state["custom_must"] = ordered[:6]
    st.session_state["custom_nice"] = ordered[6:12]


def sidebar_job_requirement():
    st.sidebar.header("1️⃣ Job requirement")
    mode = st.sidebar.radio("Source", ["Pick a finance role", "Create custom requirement"], horizontal=False)

    if mode == "Pick a finance role":
        jobs = cached_roles()
        idx = st.sidebar.selectbox("Role", range(len(jobs)), format_func=lambda i: f"{jobs[i].title}  ({jobs[i].location})")
        job = jobs[idx]
        with st.sidebar.expander("View requirement details", expanded=False):
            st.write(job.description)
            st.markdown("**Must-have skills:** " + ", ".join(job.must_have))
            st.markdown("**Nice-to-have:** " + ", ".join(job.nice_to_have))
            st.markdown(f"**Experience:** {job.min_experience:g}+ yrs"
                        + (f" (max {job.max_experience:g})" if job.max_experience else ""))
            st.markdown("**Education:** " + skills_db.EDUCATION_LEVELS[job.min_education] + " or higher")
            st.markdown("**Preferred certifications (any one):** " + (", ".join(job.preferred_certs) or "-"))
        return job

    # ---- custom requirement ----
    title = st.sidebar.text_input("Job title", "Finance Manager")
    st.sidebar.text_area("Job description (paste here)", key="custom_desc", height=140,
                         placeholder="Paste the JD here, then click 'Auto-detect skills'...")
    st.sidebar.button("✨ Auto-detect skills from description", on_click=_suggest_skills_from_description)
    all_skills = skills_db.all_skills()
    must = st.sidebar.multiselect("Must-have skills", all_skills, key="custom_must")
    nice = st.sidebar.multiselect("Nice-to-have skills", all_skills, key="custom_nice")
    nice = [s for s in nice if s not in must]          # a skill cannot be both must-have and nice-to-have
    min_exp = st.sidebar.number_input("Minimum experience (years)", 0.0, 40.0, 2.0, 0.5)
    edu = st.sidebar.selectbox("Minimum education", list(skills_db.EDUCATION_LEVELS),
                               index=3, format_func=lambda k: skills_db.EDUCATION_LEVELS[k])
    certs = st.sidebar.multiselect("Preferred certifications (any one)", list(skills_db.CERTIFICATIONS))
    return matcher.JobRequirement(title=title, description=st.session_state.get("custom_desc", ""),
                                  must_have=must, nice_to_have=nice, preferred_certs=certs,
                                  min_experience=min_exp, min_education=edu)


# --------------------------------------------------------------------------- #
# Sidebar: resume source + settings
# --------------------------------------------------------------------------- #
def sidebar_resume_source():
    st.sidebar.header("2️⃣ Resume source")
    source = st.sidebar.radio("Where are the resumes?", ["Folder on my computer", "Upload files"])
    if source == "Folder on my computer":
        folder = st.sidebar.text_input("Folder path", str(DEFAULT_RESUME_DIR),
                                       help="All .pdf / .txt / .docx files in this folder (and sub-folders) are read.")
        return {"type": "folder", "folder": folder}
    files = st.sidebar.file_uploader("Upload resumes", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    return {"type": "upload", "files": files}


def sidebar_settings():
    st.sidebar.header("3️⃣ Scoring & shortlisting")
    with st.sidebar.expander("Score weights (auto-normalised to 100%)"):
        weights = {
            "must_have": st.slider("Must-have skills", 0, 100, matcher.DEFAULT_WEIGHTS["must_have"]),
            "nice_to_have": st.slider("Nice-to-have skills", 0, 100, matcher.DEFAULT_WEIGHTS["nice_to_have"]),
            "text_similarity": st.slider("Text similarity (TF-IDF)", 0, 100, matcher.DEFAULT_WEIGHTS["text_similarity"]),
            "experience": st.slider("Experience", 0, 100, matcher.DEFAULT_WEIGHTS["experience"]),
            "education_cert": st.slider("Education & certifications", 0, 100, matcher.DEFAULT_WEIGHTS["education_cert"]),
        }
        if sum(weights.values()) == 0:
            st.warning("All weights are zero - using defaults.")
            weights = dict(matcher.DEFAULT_WEIGHTS)
    threshold = st.sidebar.slider("Minimum match score to shortlist", 0, 100, 60)
    min_must = st.sidebar.slider("Minimum must-have skills covered (%)", 0, 100, 50,
                                 help="Knock-out rule: candidates below this are rejected regardless of score.")
    top_n = st.sidebar.number_input("Maximum candidates to shortlist", 1, 100, 10)
    return weights, threshold, min_must / 100, int(top_n)


# --------------------------------------------------------------------------- #
# Main page sections
# --------------------------------------------------------------------------- #
def show_welcome():
    st.info("👈 Choose a role, point to your resume folder and click **Run matching** to begin.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 📥 Read\nParses PDF, TXT and DOCX resumes; extracts name, contact, education, experience and certifications.")
    c2.markdown("### 🧠 Extract & Score\nDetects 100+ finance skills (modelling, valuation, risk, audit, tax, tools...) and scores each candidate against the job.")
    c3.markdown("### 🏆 Rank & Shortlist\nRanks candidates, explains gaps and produces a downloadable shortlist.")
    with st.expander("How is the Match Score calculated?"):
        st.markdown("""
| Component | Default weight | How it is measured |
|---|---|---|
| Must-have skills | 40 | Share of required skills found in the resume (a skill mentioned only once earns 85% credit, repeated mentions earn 100%) |
| Nice-to-have skills | 15 | Same method for the optional skills |
| Text similarity | 15 | TF-IDF cosine similarity between the job description and the resume text |
| Experience | 15 | Years of experience / required years (capped at 100%; flagged if above the maximum) |
| Education & certifications | 15 | Education level (80%) + finance degree (20%), averaged with certification match (any preferred cert = 100%, in progress = 60%, other cert = 40%) |

**Decision rules:** below the must-have coverage limit → *Rejected*; score ≥ threshold → *Shortlisted* (top N) / *Waitlist*; within 15 points of the threshold → *Review*.
""")
    with st.expander("Finance skills library used for extraction"):
        for category, group in skills_db.SKILLS.items():
            st.markdown(f"**{category}** ({len(group)}): " + ", ".join(group))
        st.markdown(f"**Certifications tracked ({len(skills_db.CERTIFICATIONS)}):** " + ", ".join(skills_db.CERTIFICATIONS))


def show_kpis(results, n_read, n_errors):
    shortlisted = sum(r["decision"] == "Shortlisted" for r in results)
    review = sum(r["decision"] in ("Review", "Waitlist") for r in results)
    rejected = sum(r["decision"] == "Rejected - skill gap" for r in results)
    avg = sum(r["score"] for r in results) / len(results)
    cols = st.columns(5)
    cols[0].metric("Resumes screened", n_read, f"{n_errors} unreadable" if n_errors else None, delta_color="off")
    cols[1].metric("Shortlisted", shortlisted)
    cols[2].metric("Review / Waitlist", review)
    cols[3].metric("Rejected (skill gap)", rejected)
    cols[4].metric("Average score", f"{avg:.1f}", f"top: {results[0]['score']:.1f}", delta_color="off")


def table_config():
    return {
        "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=100, format="%.1f"),
        "Experience (yrs)": st.column_config.NumberColumn(format="%.1f"),
    }


def show_shortlist_tab(results, df, job):
    short = df[df["Decision"] == "Shortlisted"]
    if short.empty:
        st.warning("No candidate meets all shortlisting rules. Try lowering the score threshold or the "
                   "must-have coverage in the sidebar.")
        return
    st.subheader(f"Top candidates for **{job.title}**")
    for r in [r for r in results if r["decision"] == "Shortlisted"][:3]:
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"#### #{r['rank']}  {r['name']}")
            a.caption(f"{r['email']}  |  {r['phone']}  |  {r['file']}")
            a.write(r["remarks"])
            b.metric("Match score", f"{r['score']:.1f}")
    st.dataframe(short.drop(columns=["Decision"]), column_config=table_config(), hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Download shortlist (Excel)", to_excel_bytes(short), "shortlist.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇️ Download shortlist (CSV)", short.to_csv(index=False), "shortlist.csv", "text/csv")
    if c3.button("💾 Save to output folder"):
        path = matcher.export_shortlist(results, OUTPUT_DIR, job.title)
        st.success(f"Saved to {path}")


def show_all_tab(df):
    options = list(DECISION_ICON.values())
    chosen = st.multiselect("Filter by decision", options, default=options)
    view = df.copy()
    view["Decision"] = view["Decision"].map(DECISION_ICON)
    view = view[view["Decision"].isin(chosen)]
    st.dataframe(view, column_config=table_config(), hide_index=True)
    st.caption("Ranking order: candidates passing the must-have rule first, then by match score.")
    st.bar_chart(view.set_index("Candidate")["Match Score"])


def show_detail_tab(results, job):
    names = [f"#{r['rank']}  {r['name']}  -  {r['score']:.1f}" for r in results]
    choice = st.selectbox("Select a candidate", range(len(results)), format_func=lambda i: names[i])
    r = results[choice]

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"### {r['name']}")
        st.write(f"📧 {r['email'] or '-'}   📞 {r['phone'] or '-'}   📎 {r['file']}")
        st.markdown(f"**Decision:** {DECISION_ICON[r['decision']]}")
        st.markdown(f"**Experience:** {r['experience_years']:g} yrs   |   **Education:** {r['education_label']}   |   "
                    f"**Certifications:** {', '.join(r['certifications']) or '-'}"
                    + (f"   (in progress: {', '.join(r['certs_in_progress'])})" if r["certs_in_progress"] else ""))
        st.info(r["remarks"])
        if r["over_qualified"]:
            st.warning("Experience exceeds the maximum for this role - possible over-qualification / salary mismatch.")
    with right:
        st.metric("Match score", f"{r['score']:.1f} / 100")
        comp = pd.Series({"Must-have skills": r["must_have_score"], "Nice-to-have": r["nice_to_have_score"],
                          "Text similarity": r["similarity_score"], "Experience": r["experience_score"],
                          "Education & certs": r["edu_cert_score"]})
        st.bar_chart(comp)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**✅ Must-have skills found**")
        st.success(", ".join(r["matched_must"]) or "None")
        st.markdown("**❌ Must-have skills missing**")
        st.error(", ".join(r["missing_must"]) or "None - full coverage")
    with c2:
        st.markdown("**✅ Nice-to-have found**")
        st.success(", ".join(r["matched_nice"]) or "None")
        st.markdown("**➖ Nice-to-have missing**")
        st.warning(", ".join(r["missing_nice"]) or "None")
    with c3:
        st.markdown("**➕ Other finance skills detected**")
        st.write(", ".join(r["extra_skills"]) or "-")

    with st.expander("Show extracted resume text"):
        st.text(r["text"])


def show_matrix_tab(results, job):
    st.subheader("Skill-gap matrix")
    st.caption("✔ = skill found in the resume, ✘ = missing. Sorted by rank.")
    grid = matcher.skill_matrix(results, job)
    shown = grid.map(lambda v: "✔" if v else "✘") if hasattr(grid, "map") else grid.applymap(lambda v: "✔" if v else "✘")
    colour = lambda v: "background-color:#d4edda;color:#155724" if v == "✔" else "background-color:#f8d7da;color:#721c24"
    styler = shown.style.map(colour) if hasattr(shown.style, "map") else shown.style.applymap(colour)
    st.dataframe(styler)


def show_pool_tab(candidates, results, job):
    st.subheader("Talent-pool insights")
    required = job.must_have + [s for s in job.nice_to_have if s not in job.must_have]
    total = len(candidates)
    availability = pd.Series({s: 100 * sum(1 for c in candidates if s in c["skills"]) / total for s in required})
    st.markdown("**How many candidates have each required skill (%)** - low bars show skills that are scarce in this pool")
    st.bar_chart(availability.sort_values())
    st.markdown("**Most common finance skills across all resumes**")
    st.bar_chart(matcher.pool_skill_frequency(candidates, top=15))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("📄 AI Resume Matching Utility")
    st.caption("Finance-industry resume screening: read → extract skills → compare with job requirements → rank → shortlist")

    job = sidebar_job_requirement()
    source = sidebar_resume_source()
    weights, threshold, min_must, top_n = sidebar_settings()

    st.sidebar.divider()
    run = st.sidebar.button("🚀 Run matching", type="primary")

    if run:
        if not job.must_have and not job.nice_to_have:
            st.error("Please select at least one required skill for the job.")
            return
        with st.spinner("Reading resumes and extracting skills..."):
            try:
                if source["type"] == "folder":
                    folder = source["folder"]
                    candidates, errors = cached_folder_load(folder, folder_signature(folder))
                else:
                    if not source["files"]:
                        st.error("Please upload at least one resume.")
                        return
                    candidates, errors = resume_parser.load_resumes_from_uploads(source["files"])
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
        if not candidates:
            st.error("No readable resumes were found (supported: PDF, TXT, DOCX).")
            for name, err in errors:
                st.caption(f"• {name}: {err}")
            return
        results = matcher.rank_candidates(candidates, job, weights, threshold, top_n, min_must)
        st.session_state["run"] = dict(job=job, candidates=candidates, errors=errors, results=results)

    state = st.session_state.get("run")
    if not state:
        show_welcome()
        return

    job, candidates, errors, results = state["job"], state["candidates"], state["errors"], state["results"]
    df = matcher.results_to_dataframe(results)

    st.markdown(f"**Job:** {job.title}  &nbsp;|&nbsp; **Must-have:** {', '.join(job.must_have) or '-'}")
    show_kpis(results, len(candidates), len(errors))
    if errors:
        with st.expander(f"⚠️ {len(errors)} file(s) could not be read"):
            for name, err in errors:
                st.write(f"**{name}** - {err}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Shortlist", "📊 All candidates", "🔍 Candidate detail",
                                            "🧩 Skill-gap matrix", "📈 Talent-pool insights"])
    with tab1:
        show_shortlist_tab(results, df, job)
    with tab2:
        show_all_tab(df)
    with tab3:
        show_detail_tab(results, job)
    with tab4:
        show_matrix_tab(results, job)
    with tab5:
        show_pool_tab(candidates, results, job)


main()
