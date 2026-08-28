"""ResumeAI - AI-Powered Resume Analysis & Career Matching Dashboard.
Full-stack Streamlit web application providing genuine ML classification,
domain skill matching, ATS compatibility auditing, and downloadable PDF reports.
"""

import sys
import io
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    SUPPORTED_DOMAINS,
    SCORING_WEIGHTS,
    RATING_TIERS,
    UI_THEME,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
)
from src.pdf_parser import pdf_parser
from src.section_extractor import section_extractor
from src.skill_extractor import skill_extractor
from src.resume_parser import resume_parser
from src.domain_classifier import domain_classifier
from src.semantic_matcher import semantic_matcher
from src.ats_analyzer import ats_analyzer
from src.scoring_engine import scoring_engine
from src.recommendation_engine import recommendation_engine
from src.job_matcher import job_matcher
from src.report_generator import report_generator
from utils.helpers import create_score_gauge, create_breakdown_bar, render_skill_badge
from utils.validators import validate_file_upload, validate_extracted_text

# ==========================================
# 1. Streamlit Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="ResumeAI — AI-Powered Resume Analysis & Career Matching",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Design System
st.markdown(
    f"""
    <style>
        /* Modern Design System & CSS Tokens */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {UI_THEME['text_primary']};
        }}

        .main {{
            background-color: {UI_THEME['background']};
        }}

        /* Header Card */
        .hero-container {{
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #7C3AED 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
        }}

        .hero-title {{
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            color: #FFFFFF;
        }}

        .hero-subtitle {{
            font-size: 1.1rem;
            opacity: 0.92;
            max-width: 800px;
            line-height: 1.6;
            color: #E2E8F0;
        }}

        /* Feature Card */
        .feature-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }}

        .feature-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }}

        /* KPI Metric Cards */
        .metric-card {{
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin: 0.25rem 0;
        }}

        .metric-label {{
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {UI_THEME['text_muted']};
        }}

        /* Score Badge */
        .score-pill {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.95rem;
        }}

        /* Tab Polish */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: #F1F5F9;
            padding: 6px;
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: #FFFFFF !important;
            color: {UI_THEME['primary']} !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        /* Callout Box */
        .callout-box {{
            background-color: #F8FAFC;
            border-left: 4px solid {UI_THEME['primary']};
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }}

        /* Privacy Banner */
        .privacy-banner {{
            font-size: 0.8rem;
            color: {UI_THEME['text_muted']};
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
            border-top: 1px solid #E2E8F0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. Cached Model & Knowledge Loaders
# ==========================================
@st.cache_resource
def load_ml_classifier():
    """Caches domain classifier instance."""
    return domain_classifier


@st.cache_resource
def load_skill_extractor():
    """Caches skill extractor instance."""
    return skill_extractor


@st.cache_data
def get_model_metadata():
    """Loads genuine model evaluation metadata."""
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ==========================================
# 3. Sidebar Navigation & Global Controls
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
    st.title("ResumeAI")
    st.markdown("**AI-Powered Resume Analysis & Career Matching**")
    st.markdown("---")

    app_mode = st.radio(
        "Navigation",
        ["Resume Analyzer", "Career Domain Explorer", "Model Diagnostics & Viva"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎯 Supported Domains")
    st.info(f"**{len(SUPPORTED_DOMAINS)}** Target Tech & Professional Domains")

    st.markdown("---")
    st.markdown("### 🔒 Data Privacy")
    st.caption(
        "Your resume is analyzed in-memory and is never permanently stored on our servers."
    )


# ==========================================
# 4. Main Workflow Engine
# ==========================================
def run_full_analysis(pdf_file, selected_domain: str, job_description_text: str = ""):
    """Executes the complete multi-stage NLP & ML analysis pipeline."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Parse PDF
    status_text.markdown("⏳ **Step 1/5:** Extracting resume text & validating document structure...")
    progress_bar.progress(20)
    parse_result = pdf_parser.parse_pdf(pdf_file)

    if not parse_result["success"]:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ **PDF Extraction Error:** {parse_result.get('error', 'Unable to parse PDF.')}")
        if parse_result.get("is_scanned"):
            st.warning("⚠️ " + parse_result["warning"])
        return None

    resume_text = parse_result["text"]

    # Step 2: Information & Skill Extraction
    status_text.markdown("⏳ **Step 2/5:** Extracting contact, education, experience, and domain skills...")
    progress_bar.progress(40)
    parsed_info = resume_parser.parse_full_resume(resume_text)
    extracted_skills_data = skill_extractor.extract_skills(resume_text)
    all_skills = extracted_skills_data["all_detected_skills"]

    # Step 3: ML Domain Prediction & Selection
    status_text.markdown("⏳ **Step 3/5:** Running Supervised ML Domain Classifier...")
    progress_bar.progress(60)
    ml_prediction = domain_classifier.predict_domain(resume_text, top_k=3)

    active_target_domain = selected_domain
    if selected_domain == "Auto-Detect Domain (AI Classifier)":
        active_target_domain = ml_prediction.get("primary_domain", "Software Development")

    # Step 4: Semantic Matching & ATS Audit
    status_text.markdown("⏳ **Step 4/5:** Computing semantic cosine alignment and ATS compliance...")
    progress_bar.progress(80)
    semantic_res = semantic_matcher.calculate_domain_similarity(resume_text, active_target_domain)
    ats_res = ats_analyzer.analyze(resume_text, parsed_info, all_skills, active_target_domain)

    # Step 5: Scoring & Recommendations
    status_text.markdown("⏳ **Step 5/5:** Calculating 7-component Fit Score and generating insights...")
    progress_bar.progress(100)
    scoring_res = scoring_engine.compute_overall_score(
        resume_text, parsed_info, all_skills, ats_res, active_target_domain
    )

    strengths = recommendation_engine.generate_strengths(
        parsed_info, scoring_res["skill_alignment"]["matched_skills"], ats_res, active_target_domain
    )
    weaknesses = recommendation_engine.generate_weaknesses(
        parsed_info, scoring_res["skill_alignment"]["missing_skills"], ats_res, active_target_domain
    )
    project_recs = recommendation_engine.generate_project_recommendations(
        scoring_res["skill_alignment"]["missing_skills"], active_target_domain
    )
    improvements = recommendation_engine.generate_actionable_improvements(
        parsed_info, scoring_res["skill_alignment"]["missing_skills"], ats_res, active_target_domain
    )

    # Step 6: Optional Job Description Matching
    job_match_res = {}
    if job_description_text and len(job_description_text.strip().split()) >= 10:
        job_match_res = job_matcher.match_job_description(resume_text, job_description_text, all_skills)

    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()

    return {
        "resume_text": resume_text,
        "parsed_data": parsed_info,
        "extracted_skills": extracted_skills_data,
        "ml_prediction": ml_prediction,
        "target_domain": active_target_domain,
        "domain_match_pct": semantic_res["similarity_score_pct"],
        "semantic_data": semantic_res,
        "ats_data": ats_res,
        "scoring_data": scoring_res,
        "overall_fit_score": scoring_res["overall_fit_score"],
        "rating_label": scoring_res["rating_label"],
        "rating_color": scoring_res["rating_color"],
        "rating_badge": scoring_res["rating_badge"],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "project_recommendations": project_recs,
        "improvements": improvements,
        "job_match_result": job_match_res,
    }


# ==========================================
# 5. Application Views
# ==========================================
if app_mode == "Resume Analyzer":
    # Hero Section
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI-Powered Resume Analyzer & Career Matcher</div>
            <div class="hero-subtitle">
                Evaluate your resume with real NLP and supervised machine learning.
                Discover your ATS compatibility score, detect skill gaps for 22+ professional domains,
                and receive evidence-based portfolio recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three Feature Highlights
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🎯 ML Domain Classifier</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Supervised TF-IDF classifier predicts your optimal career domain with calibrated confidence probabilities.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_f2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📊 7-Component Fit Score</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Transparent scoring across Skill Match (30%), Experience (20%), Projects (15%), Education, and ATS metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_f3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🚀 Actionable Insights</h4>
                <p style="color: #64748B; font-size: 0.9rem;">
                    Evidence-based strengths, prioritized missing skills, and tailored project ideas without fake claims.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload & Configuration Card
    with st.container():
        st.markdown("### 📤 Upload Resume & Configure Analysis")
        col_up, col_cfg = st.columns([1.2, 1.0])

        with col_up:
            uploaded_file = st.file_uploader(
                "Upload your Resume (PDF format only)",
                type=["pdf"],
                help="Upload a standard text-based PDF resume (Max 10MB).",
            )

        with col_cfg:
            domain_options = ["Auto-Detect Domain (AI Classifier)"] + SUPPORTED_DOMAINS
            selected_domain = st.selectbox(
                "Select Target Career / Domain",
                options=domain_options,
                index=0,
                help="Choose your target job domain or let the ML classifier predict it automatically.",
            )

        with st.expander("📝 Optional: Paste Job Description (for Tailored Matching)", expanded=False):
            job_description_input = st.text_area(
                "Paste target Job Description text below:",
                height=130,
                placeholder="Paste the full job posting requirements to get targeted keyword gap analysis...",
            )

        analyze_button = st.button("🚀 Analyze My Resume", type="primary", use_container_width=True)

    # Process Analysis
    if analyze_button:
        is_valid, err_msg = validate_file_upload(uploaded_file)
        if not is_valid:
            st.error(f"❌ {err_msg}")
        else:
            with st.spinner("Analyzing resume content..."):
                analysis_res = run_full_analysis(
                    uploaded_file, selected_domain, job_description_input
                )
                if analysis_res:
                    st.session_state["latest_analysis"] = analysis_res

    # Display Results if available
    if "latest_analysis" in st.session_state:
        res = st.session_state["latest_analysis"]
        st.markdown("---")

        # Top Results Banner
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Assessment Target</span>
                    <h2 style="margin: 0; color: #1E293B;">{res['target_domain']}</h2>
                    <span style="color: #64748B; font-size: 0.9rem;">Candidate: <b>{res['parsed_data']['personal_info']['name']}</b> | ML Prediction: <b>{res['ml_prediction'].get('primary_domain', 'N/A')}</b> ({res['ml_prediction'].get('confidence_pct', 0)}% confidence)</span>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {res['rating_color']}; color: white; padding: 8px 18px; border-radius: 9999px; font-weight: 800; font-size: 1.1rem;">
                        {res['overall_fit_score']} / 100 — {res['rating_label'].upper()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Metric KPI Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">AI Resume Fit</div>
                    <div class="metric-value" style="color: {res['rating_color']};">{res['overall_fit_score']}<small style="font-size: 1rem;">/100</small></div>
                    <div style="font-size: 0.8rem; color: #64748B;">{res['rating_badge']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Domain Match</div>
                    <div class="metric-value" style="color: {UI_THEME['primary']};">{res['domain_match_pct']}%</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Semantic Alignment</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            skill_pct = res["scoring_data"]["skill_alignment"]["skill_match_pct"]
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Skill Match</div>
                    <div class="metric-value" style="color: {UI_THEME['secondary']};">{skill_pct}%</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Target Role Coverage</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            ats_val = res["ats_data"]["ats_score"]
            ats_color = UI_THEME["success"] if ats_val >= 80 else (UI_THEME["warning"] if ats_val >= 65 else UI_THEME["danger"])
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ATS Compatibility</div>
                    <div class="metric-value" style="color: {ats_color};">{ats_val}<small style="font-size: 1rem;">/100</small></div>
                    <div style="font-size: 0.8rem; color: #64748B;">Parser Audit Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 7 Rich Interactive Tabs
        tab_overview, tab_skills, tab_ats, tab_exp, tab_job, tab_recs, tab_download = st.tabs([
            "📊 Overview",
            "🛠️ Skill Analysis",
            "🤖 ATS Audit",
            "💼 Experience & Projects",
            "🎯 Job Matcher",
            "💡 Recommendations",
            "📥 Download Report",
        ])

        # ==========================================
        # Tab 1: Overview
        # ==========================================
        with tab_overview:
            col_ov_left, col_ov_right = st.columns([1.1, 1.2])

            with col_ov_left:
                st.markdown("#### 🎯 Score Breakdown")
                st.plotly_chart(
                    create_breakdown_bar(res["scoring_data"]["breakdown"]),
                    use_container_width=True,
                )
                st.caption(
                    "Weights: Skill Match (30%), Experience (20%), Projects (15%), Education (10%), ATS (10%), Structure (10%), Formatting (5%)."
                )

            with col_ov_right:
                st.markdown("#### 🌟 Key Strengths")
                for s in res["strengths"][:4]:
                    st.markdown(f"- ✅ **{s}**")

                st.markdown("#### ⚠️ Primary Areas for Growth")
                for w in res["weaknesses"][:4]:
                    st.markdown(f"- ⚠️ {w}")

        # ==========================================
        # Tab 2: Skills
        # ==========================================
        with tab_skills:
            st.markdown("### 🔍 Detailed Skill Gap Analysis")

            col_sk_m, col_sk_gap = st.columns(2)

            with col_sk_m:
                st.markdown("#### ✅ Matched Target Domain Skills")
                matched_items = res["scoring_data"]["skill_alignment"]["matched_skills"]
                if matched_items:
                    badges_html = "<div>"
                    for item in matched_items:
                        badges_html += render_skill_badge(
                            item["skill_name"], item.get("importance", "Medium"), matched=True
                        )
                    badges_html += "</div>"
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.warning("No explicit target domain skills detected in this resume.")

            with col_sk_gap:
                st.markdown("#### ❌ Missing Recommended Skills (Prioritized)")
                missing_items = res["scoring_data"]["skill_alignment"]["missing_skills"]
                if missing_items:
                    badges_html = "<div>"
                    for item in missing_items[:12]:
                        badges_html += render_skill_badge(
                            item["skill_name"], item.get("importance", "Medium"), matched=False
                        )
                    badges_html += "</div>"
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.success("Excellent! You match all major skills for this target domain.")

            st.markdown("---")
            st.markdown("#### 📚 All Detected Skills by Technical Category")
            categorized = res["extracted_skills"]["categorized_skills"]
            for cat, s_list in categorized.items():
                if s_list:
                    with st.expander(f"{cat} ({len(s_list)} detected)", expanded=True):
                        st.write(", ".join(s_list))

        # ==========================================
        # Tab 3: ATS Analysis
        # ==========================================
        with tab_ats:
            st.markdown("### 🤖 Applicant Tracking System (ATS) Audit")
            col_ats_g, col_ats_details = st.columns([1, 1.4])

            with col_ats_g:
                st.plotly_chart(
                    create_score_gauge(
                        res["ats_data"]["ats_score"],
                        title="ATS Compatibility Score",
                        color=UI_THEME["success"] if res["ats_data"]["ats_score"] >= 80 else UI_THEME["warning"],
                    ),
                    use_container_width=True,
                )

            with col_ats_details:
                st.markdown("#### 📋 Compliance Checklist")
                for s in res["ats_data"]["strengths"]:
                    st.markdown(f"- ✅ {s}")

                if res["ats_data"]["issues"]:
                    st.markdown("#### ⚠️ Identified Parsing Issues")
                    for iss in res["ats_data"]["issues"]:
                        st.markdown(f"- ⚠️ {iss}")

            if res["ats_data"]["suggestions"]:
                st.markdown("#### 💡 ATS Optimization Recommendations")
                for sug in res["ats_data"]["suggestions"]:
                    st.info(f"💡 **Action Item:** {sug}")

        # ==========================================
        # Tab 4: Experience & Projects
        # ==========================================
        with tab_exp:
            st.markdown("### 💼 Extracted Experience & Project Profile")
            col_p1, col_p2 = st.columns(2)

            with col_p1:
                st.markdown("#### 👤 Candidate Contact & Education")
                p_info = res["parsed_data"]["personal_info"]
                edu_info = res["parsed_data"]["education"]

                st.write(f"**Name:** {p_info.get('name')}")
                st.write(f"**Email:** {p_info.get('email')}")
                st.write(f"**Phone:** {p_info.get('phone')}")
                st.write(f"**LinkedIn:** {p_info.get('linkedin')}")
                st.write(f"**GitHub:** {p_info.get('github')}")

                st.markdown("---")
                st.write(f"**Degrees:** {', '.join(edu_info.get('degrees', ['Not detected']))}")
                st.write(f"**Majors:** {', '.join(edu_info.get('majors', ['Not detected']))}")
                st.write(f"**Graduation Years:** {', '.join(edu_info.get('grad_years', ['Not detected']))}")

            with col_p2:
                st.markdown("#### 📈 Experience Impact & Action Verbs")
                exp_info = res["parsed_data"]["experience"]
                proj_info = res["parsed_data"]["projects"]

                st.write(f"**Detected Titles:** {', '.join(exp_info.get('detected_titles', ['Not detected']))}")
                st.write(f"**Estimated Experience:** {exp_info.get('estimated_years')}")
                st.write(f"**Action Verbs Count:** {exp_info.get('action_verbs_count')}")
                if exp_info.get("action_verbs_sample"):
                    st.caption(f"Verbs: {', '.join(exp_info['action_verbs_sample'])}")

                st.markdown("---")
                st.write(f"**Projects Section Detected:** {'Yes' if proj_info.get('has_projects_section') else 'No'}")
                st.write(f"**Measurable Metrics Found:** {'Yes' if exp_info.get('metrics_count', 0) > 0 else 'No'}")
                if exp_info.get("metrics_sample"):
                    st.caption(f"Sample Metrics: {', '.join(exp_info['metrics_sample'])}")

        # ==========================================
        # Tab 5: Job Matcher
        # ==========================================
        with tab_job:
            st.markdown("### 🎯 Job Description Compatibility Matcher")
            if res.get("job_match_result", {}).get("success"):
                jm = res["job_match_result"]
                st.success(f"**Job Match Score: {jm['match_pct']}%**")

                col_jm1, col_jm2 = st.columns(2)
                with col_jm1:
                    st.markdown("#### ✅ Matched Job Requirements")
                    st.write(", ".join(jm.get("matched_skills", [])) or "None detected")
                with col_jm2:
                    st.markdown("#### ❌ Missing Job Requirements")
                    st.write(", ".join(jm.get("missing_skills", [])) or "None! All requirements matched.")

                st.markdown("#### 💡 Tailored Job Recommendations")
                for rec in jm.get("recommendations", []):
                    st.info(rec)
            else:
                st.info("Paste a target job description in the upload area above to see exact keyword match percentages.")

        # ==========================================
        # Tab 6: Recommendations
        # ==========================================
        with tab_recs:
            st.markdown("### 🚀 Actionable Growth Roadmap & Project Recommendations")

            st.markdown("#### 🛠️ Recommended Portfolio Projects (to bridge missing skills)")
            for proj in res["project_recommendations"]:
                with st.expander(f"📌 {proj['title']} ({proj['difficulty']})", expanded=True):
                    st.write(proj["description"])
                    st.caption(f"**Skills Developed:** {', '.join(proj['skills_addressed'])}")

            st.markdown("---")
            st.markdown("#### ✍️ Actionable Resume Enhancement Items")
            for imp in res["improvements"]:
                st.markdown(f"**{imp['category']}**")
                st.write(f"- *Issue:* {imp['issue']}")
                st.info(f"💡 *Action:* {imp['recommendation']}")

            st.markdown("---")
            st.markdown("#### 🌐 Top 3 Domain Matches (ML Classifier)")
            for idx, d_info in enumerate(res["ml_prediction"].get("top_domains", []), 1):
                st.markdown(f"{idx}. **{d_info['domain']}** — {d_info['confidence_pct']}% confidence")

        # ==========================================
        # Tab 7: Download Report
        # ==========================================
        with tab_download:
            st.markdown("### 📥 Download Assessment Report")
            st.write("Export your complete resume analysis report in PDF or Markdown format.")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                # PDF Download
                try:
                    pdf_path = report_generator.generate_pdf_report(res)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="📄 Download Full PDF Report",
                        data=pdf_bytes,
                        file_name=f"ResumeAI_Report_{res['target_domain'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error generating PDF report: {e}")

            with col_d2:
                # Markdown Download
                md_content = report_generator.generate_markdown_report(res)
                st.download_button(
                    label="📝 Download Markdown Report",
                    data=md_content,
                    file_name=f"ResumeAI_Analysis_{res['target_domain'].replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

# ==========================================
# View 2: Career Domain Explorer
# ==========================================
elif app_mode == "Career Domain Explorer":
    st.title("🌐 Career Domain Explorer & Knowledge Layer")
    st.markdown(
        "Explore required skills, educational baselines, and industry standards across 22 professional domains."
    )

    selected_exp_domain = st.selectbox("Select Domain to Inspect", options=SUPPORTED_DOMAINS)

    if Path("data/domain_requirements/occupations.csv").exists():
        df_occ = pd.read_csv("data/domain_requirements/occupations.csv")
        domain_row = df_occ[df_occ["domain_name"] == selected_exp_domain]

        if not domain_row.empty:
            row = domain_row.iloc[0]
            st.markdown(f"### {row['domain_name']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Education Expectation:** {row['education_expectation']}")
            st.write(f"**Experience Baseline:** {row['experience_expectation']}")
            st.write(f"**Related Roles:** {row['related_roles']}")

    if Path("data/domain_requirements/domain_mapping.csv").exists():
        df_map = pd.read_csv("data/domain_requirements/domain_mapping.csv")
        domain_skills = df_map[df_map["domain_name"] == selected_exp_domain]
        st.markdown("#### Required & Important Skills")
        st.dataframe(domain_skills[["skill_name", "importance", "category"]], use_container_width=True)

# ==========================================
# View 3: Model Diagnostics & Viva Portal
# ==========================================
elif app_mode == "Model Diagnostics & Viva":
    st.title("🔬 AI/ML Model Diagnostics & Viva Portal")
    st.markdown(
        "Demonstrates genuine supervised machine learning metrics, training methodology, and confusion matrices for project review."
    )

    meta = get_model_metadata()

    if not meta:
        st.warning("Model metadata not found. Run `python scripts/train_model.py` to generate authentic metrics.")
    else:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Best Model", meta.get("best_model_name", "N/A"))
        with col_m2:
            st.metric("Test Accuracy", f"{meta.get('test_metrics', {}).get('accuracy', 0) * 100:.2f}%")
        with col_m3:
            st.metric("Test Macro F1", f"{meta.get('test_metrics', {}).get('macro_f1', 0) * 100:.2f}%")
        with col_m4:
            st.metric("Total Resumes", meta.get("total_dataset_size", 0))

        st.markdown("---")
        st.markdown("### 📊 Model Validation Comparison")
        val_res = meta.get("validation_results", {})
        if val_res:
            val_df = pd.DataFrame(val_res).T.reset_index().rename(columns={"index": "Model Name"})
            val_df["val_accuracy"] = val_df["val_accuracy"].apply(lambda x: f"{x * 100:.2f}%")
            val_df["val_macro_f1"] = val_df["val_macro_f1"].apply(lambda x: f"{x * 100:.2f}%")
            val_df["val_weighted_f1"] = val_df["val_weighted_f1"].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(val_df, use_container_width=True)

        st.markdown("### 🗺️ Confusion Matrix on Held-Out Test Data")
        classes = meta.get("classes", [])
        conf_mat = meta.get("confusion_matrix", [])
        if classes and conf_mat:
            fig_cm = px.imshow(
                np.array(conf_mat),
                x=classes,
                y=classes,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Domain", y="Actual Domain", color="Count"),
            )
            fig_cm.update_layout(height=600, font=dict(family=UI_THEME["font_family"]))
            st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(f"Artifacts trained & serialized on: {meta.get('training_timestamp')}")


# Footer
st.markdown(
    """
    <div class="privacy-banner">
        <b>ResumeAI</b> — Academic & Professional AI/ML Portfolio Project | Built with Scikit-learn, TF-IDF, PyMuPDF & Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)
