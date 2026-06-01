"""
A/B Test Plan
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Couleurs ──
VIOLET  = colors.HexColor("#6C63FF")
TEAL    = colors.HexColor("#00C49F")
RED     = colors.HexColor("#FF6B6B")
DARK    = colors.HexColor("#1A1A2E")
LIGHT   = colors.HexColor("#F0F0FF")
ORANGE  = colors.HexColor("#FFB347")
GREY    = colors.HexColor("#666666")
WHITE   = colors.white

def build_ab_test_plan(output_path="AB_Test_Plan_PrixFixe_Maroc.pdf"):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── Styles personnalisés ──
    title_style = ParagraphStyle("title", parent=styles["Title"],
        fontSize=20, textColor=DARK, spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
        fontSize=11, textColor=VIOLET, spaceAfter=12, alignment=TA_CENTER, fontName="Helvetica-BoldOblique")
    h1_style = ParagraphStyle("h1", parent=styles["Heading1"],
        fontSize=13, textColor=WHITE, backColor=VIOLET, spaceAfter=6, spaceBefore=12,
        leftIndent=-0.5*cm, rightIndent=-0.5*cm, borderPad=5, fontName="Helvetica-Bold")
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=11, textColor=VIOLET, spaceAfter=4, spaceBefore=8, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=9.5, textColor=DARK, leading=15, alignment=TA_JUSTIFY)
    bullet_style = ParagraphStyle("bullet", parent=styles["Normal"],
        fontSize=9.5, textColor=DARK, leading=14, leftIndent=12, bulletIndent=0)
    note_style = ParagraphStyle("note", parent=styles["Normal"],
        fontSize=8.5, textColor=GREY, leading=13, alignment=TA_JUSTIFY)
    meta_style = ParagraphStyle("meta", parent=styles["Normal"],
        fontSize=9, textColor=GREY, alignment=TA_CENTER)

    # ══════════════════════════════════════════════
    # PAGE 1
    # ══════════════════════════════════════════════

    # ── En-tête ──
    story.append(Paragraph("A/B TEST PLAN", title_style))
    story.append(Paragraph("Viabilité du Modèle Prix Fixe au Maroc face à inDrive", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=VIOLET))
    story.append(Spacer(1, 6))
    meta = [
        ["Module :", "Data-Driven Decision Making"],
        ["Équipe :", "Coéquipier A (Phases 1-3) · Coéquipier B (Phases 4-6)"],
        ["Date :", "Juin 2026"],
        ["Version :", "v1.0 — Livrable Final"]
    ]
    meta_table = Table(meta, colWidths=[3*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0,0),(-1,-1),"Helvetica"),
        ("FONTSIZE",  (0,0),(-1,-1), 9),
        ("TEXTCOLOR", (0,0),(0,-1), VIOLET),
        ("FONTNAME",  (0,0),(0,-1),"Helvetica-Bold"),
        ("TEXTCOLOR", (1,0),(1,-1), DARK),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY))
    story.append(Spacer(1, 8))

    # ── Section 1 : Contexte ──
    story.append(Paragraph("1. CONTEXTE & OBJECTIF DE L'EXPÉRIENCE", h1_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Le marché marocain du ride-hailing est dominé par inDrive, dont le modèle repose sur la négociation "
        "tarifaire entre conducteurs et passagers. Notre analyse data (XGBoost AUC=0.9997, Logistic AUC=0.73) "
        "révèle que 40% des utilisateurs sont sensibles au prix et pourraient résister à l'adoption d'un modèle "
        "à prix fixe algorithmique. Avant tout déploiement à grande échelle, il est impératif de valider "
        "expérimentalement les recommandations issues des modèles prédictifs.", body_style))
    story.append(Spacer(1, 6))

    # ── Section 2 : Hypothèses ──
    story.append(Paragraph("2. HYPOTHÈSES DE TRAVAIL", h1_style))
    story.append(Spacer(1, 6))

    hyp_data = [
        ["Test", "Hypothèse nulle (H0)", "Hypothèse alternative (H1)", "Seuil alpha"],
        ["A/B #1\nAdoption", "Le taux d'adoption du prix fixe ne\ndiffère pas entre les segments\néconomiques et premium.",
         "Les segments premium (clusters 2-3)\nadoptent le prix fixe 15%+ plus\nqu'les segments économiques.", "5%"],
        ["A/B #2\nRétention", "La remise de 10% (-10%) n'a pas d'effet\nsignificatif sur le churn des utilisateurs\nà risque (prob_churn > 0.30).",
         "La remise de 10% réduit le churn\nd'au moins 25% chez les utilisateurs\nà risque identifiés par LogReg.", "5%"],
        ["A/B #3\nPrix dynamique", "Un prix variable (±20% selon heure)\nne diffère pas du prix fixe constant\nsur le taux de complétion des courses.",
         "Le pricing dynamique augmente\nle taux de complétion de 8%+ aux\nheures de pointe.", "5%"],
    ]
    hyp_table = Table(hyp_data, colWidths=[2.2*cm, 5.5*cm, 5.5*cm, 2*cm])
    hyp_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), VIOLET),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("FONTNAME",    (0,1),(-1,-1), "Helvetica"),
        ("TEXTCOLOR",   (0,1),(-1,-1), DARK),
        ("BACKGROUND",  (0,1),(-1,1), LIGHT),
        ("BACKGROUND",  (0,3),(-1,3), LIGHT),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [LIGHT, WHITE]),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("LEADING",     (0,0),(-1,-1), 12),
    ]))
    story.append(hyp_table)
    story.append(Spacer(1, 10))

    # ── Section 3 : Design expérimental ──
    story.append(Paragraph("3. DESIGN EXPÉRIMENTAL", h1_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.1 Calcul de la taille d'échantillon (Power Analysis)", h2_style))
    story.append(Paragraph(
        "Paramètres statistiques retenus pour le calcul de puissance :", body_style))
    story.append(Spacer(1, 4))

    power_data = [
        ["Paramètre", "A/B #1 — Adoption", "A/B #2 — Rétention", "A/B #3 — Prix dyn."],
        ["Taux baseline", "24.2%", "6.8%", "82.0%"],
        ["Effet minimal détectable (MDE)", "15% relatif (+3.6pp)", "25% relatif (-1.7pp)", "8% relatif (+6.6pp)"],
        ["Puissance statistique (1-beta)", "80%", "80%", "80%"],
        ["Niveau de signif. alpha", "5% (bilatéral)", "5% (bilatéral)", "5% (bilatéral)"],
        ["Taille par groupe (n)", "~1 840", "~2 950", "~620"],
        ["Total participants", "~3 680", "~5 900", "~1 240"],
    ]
    power_table = Table(power_data, colWidths=[4*cm, 3.8*cm, 3.8*cm, 3.8*cm])
    power_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), TEAL),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("ALIGN",       (1,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [LIGHT, WHITE]),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
        ("TEXTCOLOR",   (0,6),(-1,6), VIOLET),
        ("FONTNAME",    (0,6),(-1,6), "Helvetica-Bold"),
    ]))
    story.append(power_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Note méthodologique : Calculs effectués via la formule Cohen (1988) pour comparaison de proportions. "
        "Correction de Bonferroni appliquée pour les 3 tests simultanés (alpha ajusté = 1.67%).",
        note_style))

    # ══════════════════════════════════════════════
    # PAGE 2
    # ══════════════════════════════════════════════
    story.append(PageBreak())

    # ── Section 4 : Protocole ──
    story.append(Paragraph("4. PROTOCOLE D'EXÉCUTION", h1_style))
    story.append(Spacer(1, 6))

    proto_data = [
        ["Phase", "Durée", "Actions", "Responsable"],
        ["Pré-lancement", "Sem. 1-2", "Randomisation des utilisateurs (stratifiée par cluster). "
         "Configuration tracking. Validation technique. Gel de l'algorithme de prix.", "Data Engineering"],
        ["Lancement", "Sem. 3", "Activation progressive (10% trafic → 50% → 100%). "
         "Monitoring temps réel des métriques de guardrail.", "Product + Data"],
        ["Collecte", "Sem. 4-6\n(21 jours)", "Collecte automatisée. Rapport hebdomadaire. "
         "Alerte si guardrail violé (churn +3pp, rating -0.3).", "Data Analyst"],
        ["Analyse", "Sem. 7", "Test de significativité statistique. SHAP post-hoc. "
         "Rapport de synthèse avec recommandation Go/No-Go.", "Coéquipier B"],
        ["Décision", "Sem. 8", "Présentation jury. Déploiement ou rollback.", "Direction"],
    ]
    proto_table = Table(proto_data, colWidths=[2.5*cm, 2.3*cm, 9.2*cm, 2.5*cm])
    proto_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), DARK),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1),(0,-1), VIOLET),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("ALIGN",       (2,1),(2,-1), "LEFT"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [LIGHT, WHITE]),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
    ]))
    story.append(proto_table)
    story.append(Spacer(1, 10))

    # ── Section 5 : Métriques ──
    story.append(Paragraph("5. MÉTRIQUES DE SUIVI", h1_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1 Métriques primaires (KPIs décisionnels)", h2_style))
    kpis = [
        ["KPI", "Définition", "Source", "Fréquence"],
        ["Taux d'adoption prix fixe", "% utilisateurs choisissant prix algorithmique\nvs négociation libre", "App logs", "Quotidien"],
        ["Taux de churn", "% utilisateurs inactifs >30j après essai\n(modèle LogReg, AUC=0.73)", "Rides DB", "Hebdo"],
        ["Taux de complétion", "% courses complétées / courses initiées", "App logs", "Quotidien"],
        ["Net Promoter Score (NPS)", "Satisfaction utilisateur post-course\n(échelle 0-10)", "Push notif.", "Bi-hebdo"],
    ]
    kpi_table = Table(kpis, colWidths=[3.8*cm, 6.2*cm, 2.5*cm, 2.5*cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), VIOLET),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [LIGHT, WHITE]),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",  (0,0),(-1,-1), 5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.2 Métriques guardrail (sécurité de l'expérience)", h2_style))
    guardrails = [
        "Note moyenne conducteur < 3.5/5 → Arrêt immédiat",
        "Taux de plaintes > 5% → Alerte + réunion d'urgence",
        "Churn global +3 points de % vs baseline → Rollback automatique",
        "Revenus conducteurs -15% vs semaine précédente → Pause et analyse",
    ]
    for g in guardrails:
        story.append(Paragraph(f"• {g}", bullet_style))
    story.append(Spacer(1, 10))

    # ── Section 6 : Impact ──
    story.append(Paragraph("6. ESTIMATION DE L'IMPACT FINANCIER ET OPÉRATIONNEL", h1_style))
    story.append(Spacer(1, 6))

    impact_data = [
        ["Scénario", "Adoption\n(%)", "Rev. annuel\n(M MAD)", "Churn\nréduction", "Investissement\n(M MAD)", "ROI Net\n(M MAD)"],
        ["Pessimiste (H0 non rejetées)", "10%", "8.3", "~0%", "2.1", "6.2"],
        ["Réaliste (H1 partiellement)", "25%", "20.8", "-18%", "2.1", "18.7"],
        ["Optimiste (H1 toutes rejetées)", "42%", "34.9", "-28%", "2.1", "32.8"],
    ]
    impact_table = Table(impact_data, colWidths=[4.2*cm, 2*cm, 2.5*cm, 2.3*cm, 2.7*cm, 2.5*cm])
    impact_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), DARK),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,1),(0,1), RED),
        ("TEXTCOLOR",   (0,2),(0,2), ORANGE),
        ("TEXTCOLOR",   (0,3),(0,3), TEAL),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.HexColor("#FFF5F5"), colors.HexColor("#FFFBF0"), colors.HexColor("#F0FFF8")]),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("TEXTCOLOR",   (5,1),(5,3), VIOLET),
        ("FONTNAME",    (5,1),(5,3), "Helvetica-Bold"),
    ]))
    story.append(impact_table)
    story.append(Spacer(1, 8))

    # ── Section 7 : Critères ──
    story.append(Paragraph("7. CRITÈRES DE DÉCISION GO / NO-GO", h1_style))
    story.append(Spacer(1, 6))

    decision_data = [
        ["Critère", "Seuil GO", "Seuil NO-GO"],
        ["Adoption prix fixe (p-value)", "< 0.05 ET delta > 15% relatif", "> 0.05 OU delta < 5%"],
        ["Réduction churn", "> 20% chez segments cibles", "< 10%"],
        ["NPS post-expérience", "> 35 (groupe test)", "< 25"],
        ["Taux complétion", "Stable ou +5%", "-5% vs baseline"],
        ["Revenus conducteurs", "Stable ou hausse", "-10% vs baseline"],
    ]
    dec_table = Table(decision_data, colWidths=[4.5*cm, 5.5*cm, 5.5*cm])
    dec_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0), VIOLET),
        ("TEXTCOLOR",   (0,0),(-1,0), WHITE),
        ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 8.5),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ("GRID",        (0,0),(-1,-1), 0.5, GREY),
        ("BACKGROUND",  (1,1),(1,-1), colors.HexColor("#EFFFEF")),
        ("BACKGROUND",  (2,1),(2,-1), colors.HexColor("#FFF0F0")),
        ("TEXTCOLOR",   (1,1),(1,-1), colors.HexColor("#00AA44")),
        ("TEXTCOLOR",   (2,1),(2,-1), colors.HexColor("#CC2200")),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
    ]))
    story.append(dec_table)
    story.append(Spacer(1, 10))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1, color=VIOLET))
    story.append(Spacer(1, 6))
    footer_content = [
        ["Projet DDDM — Module Data-Driven Decision Making | Deadline : 07 Juin 2026",
         "Données : Uber Fares (Kaggle) + Ride Hailing Transactions (Kaggle) | 236 003 lignes brutes"],
        ["Modèles : LR (R²=0.648) · XGBoost (AUC=0.9997) · LogReg (AUC=0.735)",
         "Code : github.com/<username>/projet-dddm-prix-fixe-maroc — Tag v1.0"],
    ]
    for row in footer_content:
        story.append(Paragraph(" | ".join(row), note_style))

    doc.build(story)
    print(f"PDF généré : {output_path}")
    return output_path

if __name__ == "__main__":
    build_ab_test_plan("AB_Test_Plan_PrixFixe_Maroc.pdf")
