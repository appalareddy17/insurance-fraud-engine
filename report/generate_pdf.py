import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
)


def _fig_to_image(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


def generate_report(
    output_path: str,
    dataset_stats: dict,
    metrics_df: pd.DataFrame,
    fig_roc: plt.Figure,
    fig_cm: plt.Figure,
    fig_shap: plt.Figure,
) -> None:
    """
    dataset_stats keys: total_rows, fraud_count, fraud_pct, legit_count
    metrics_df: index=model name, columns=[roc_auc, pr_auc, precision, recall, f1]
    """
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=20, spaceAfter=6, alignment=TA_CENTER,
    )
    story.append(Paragraph("Insurance Fraud Claims Detection Engine", title_style))
    story.append(Paragraph("Case Study Report", styles['Heading2']))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("1. Project Overview", styles['Heading1']))
    story.append(Paragraph(
        "This project develops a machine learning classifier to detect fraudulent auto insurance "
        "claims. Two models are compared: a Logistic Regression baseline and an XGBoost classifier. "
        "SHAP values provide model explainability at both global and individual claim levels.",
        styles['BodyText'],
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("2. Dataset Analysis", styles['Heading1']))
    story.append(Paragraph(
        f"Total claims: {dataset_stats['total_rows']:,}  |  "
        f"Fraud: {dataset_stats['fraud_count']:,} ({dataset_stats['fraud_pct']:.1f}%)  |  "
        f"Legitimate: {dataset_stats['legit_count']:,} ({100 - dataset_stats['fraud_pct']:.1f}%)",
        styles['BodyText'],
    ))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("3. Model Results", styles['Heading1']))

    table_data = [['Model', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1']]
    for model_name, row in metrics_df.iterrows():
        table_data.append([
            model_name,
            f"{row['roc_auc']:.3f}",
            f"{row['pr_auc']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
        ])
    col_widths = [2.0 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F4F8')]),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    roc_buf = _fig_to_image(fig_roc)
    story.append(Image(roc_buf, width=5 * inch, height=3.5 * inch))
    story.append(Spacer(1, 0.15 * inch))

    cm_buf = _fig_to_image(fig_cm)
    story.append(Image(cm_buf, width=3.5 * inch, height=2.8 * inch))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("4. Key Findings", styles['Heading1']))
    story.append(Paragraph(
        "SHAP analysis identifies the most influential fraud predictors. "
        "High claim-to-premium ratios, night-time incidents, and the absence of both "
        "a police report and witnesses are consistent fraud indicators. "
        "XGBoost outperforms Logistic Regression on PR-AUC, which is the more meaningful "
        "metric given class imbalance.",
        styles['BodyText'],
    ))
    story.append(Spacer(1, 0.15 * inch))

    shap_buf = _fig_to_image(fig_shap)
    story.append(Image(shap_buf, width=5.5 * inch, height=3.5 * inch))

    doc.build(story)
