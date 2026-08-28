"""Generates a realistic test resume PDF for automated verification and user testing.
"""

from pathlib import Path
from fpdf import FPDF

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PDF = DATA_DIR / "sample_resume.pdf"


def create_sample_resume():
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Alex Mercer", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, "alex.mercer@email.com | (555) 234-5678 | San Francisco, CA", ln=True, align="C")
    pdf.cell(0, 5, "linkedin.com/in/alexmercer-tech | github.com/alexmercer-dev", ln=True, align="C")
    pdf.ln(5)

    # Professional Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 7, "PROFESSIONAL SUMMARY", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        "Senior Data Scientist and Machine Learning Engineer with 5+ years of experience building predictive models, "
        "recommender engines, and scalable backend APIs. Proficient in Python, SQL, Scikit-learn, PyTorch, and Docker. "
        "Proven track record of optimizing model inference latency by 40% and increasing business revenue by $1.2M."
    )
    pdf.ln(4)

    # Technical Skills
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 7, "TECHNICAL SKILLS", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        "- Programming Languages: Python, SQL, C++, Bash\n"
        "- Machine Learning & AI: Scikit-learn, PyTorch, Pandas, NumPy, Deep Learning, NLP, Feature Engineering\n"
        "- Cloud & DevOps: Docker, Kubernetes, AWS, Git, CI/CD\n"
        "- Databases & Tools: PostgreSQL, Redis, Tableau, Power BI, Jira"
    )
    pdf.ln(4)

    # Work Experience
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 7, "WORK EXPERIENCE", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(140, 6, "Senior Data Scientist | CloudScale Analytics")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(50, 6, "2021 - Present", ln=True, align="R")

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        "- Architected a real-time recommendation engine using Python, PyTorch, and FastAPI, boosting user engagement by 28%.\n"
        "- Automated ETL feature pipelines in PostgreSQL and Docker, reducing data processing time by 45%.\n"
        "- Spearheaded cross-functional team of 6 engineers to deploy scalable microservices on AWS."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(140, 6, "Machine Learning Developer | DataVenture Inc.")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(50, 6, "2019 - 2021", ln=True, align="R")

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 5,
        "- Developed churn prediction models using Scikit-learn and Pandas, achieving 91% precision.\n"
        "- Engineered automated SQL data pipelines processing 10M+ daily records."
    )
    pdf.ln(4)

    # Education
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 7, "EDUCATION", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(140, 5, "Master of Science in Computer Science")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(50, 5, "Graduated 2019", ln=True, align="R")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, "University of California, Berkeley", ln=True)

    pdf.output(str(OUTPUT_PDF))
    print(f"Sample PDF created at: {OUTPUT_PDF}")


if __name__ == "__main__":
    create_sample_resume()
