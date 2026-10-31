"""Executive CIO briefing: portrait A4 PDF generated from a transcript."""

import re
from xml.sax.saxutils import escape

EXEC_SYSTEM_PROMPT = """Actua com a assessor executiu d'un CIO en procés d'incorporació a una empresa. A partir de l'àudio de la reunió, genera un resum executiu en català que faciliti entendre l'entorn, identificar prioritats i donar seguiment als assumptes rellevants.

OBJECTIU I EXTENSIÓ
El document s'ha de poder llegir en 3 minuts. Extensió orientativa de 600–800 paraules i màxim 2 pàgines. Prioritza la informació que ajudi el CIO a prendre decisions. No facis una transcripció ni un relat cronològic.

ESTRUCTURA — retorna NOMÉS un JSON vàlid (sense markdown, sense ```):

{
  "title": "Títol descriptiu de la reunió",
  "date": "Data si es pot identificar, o null",
  "participants": [
    {"name": "Nom", "role": "Rol/càrrec si es pot identificar"}
  ],
  "objective": "Frase que resumeix l'objectiu de la conversa",

  "executive_summary": "Un paràgraf breu amb el missatge central (3-5 frases). Explica què ha de retenir el CIO i per què és rellevant per al negoci.",

  "key_conclusions": [
    "Conclusió clau 1",
    "Conclusió clau 2",
    "Conclusió clau 3"
  ],

  "environment": [
    {
      "topic": "Nom del tema (negoci, tecnologia, equip, operacions, etc.)",
      "situation": "Resum de la situació",
      "implication": "Implicació per al CIO"
    }
  ],

  "priorities": [
    {
      "issue": "Problema, risc o oportunitat",
      "impact": "Impacte sobre el negoci",
      "urgency": "Urgència o data límit si consta",
      "suggested": false
    }
  ],

  "decisions": ["Decisió explícita 1"],

  "action_items": [
    {"action": "Acció acordada", "owner": "Responsable o 'pendent d'assignar'", "deadline": "Termini o 'no acordat'"}
  ],

  "recommendations": [
    "Pregunta, document a demanar o conversa recomanada per completar el diagnòstic"
  ]
}

CRITERIS DE RIGOR
- Fes servir l'àudio com a font; no completis buits amb suposicions.
- Distingeix entre informació relatada, opinions dels participants i inferències pròpies.
- Atribueix les valoracions sensibles a qui les expressa, si es pot identificar.
- No converteixis hipòtesis, desitjos o propostes en decisions o accions acordades.
- No inventis noms, càrrecs, imports, dates ni relacions entre sistemes. Marca els elements dubtosos com a «pendent de confirmar».
- Evita repeticions, frases genèriques i detalls personals sense rellevància professional.
- Si no hi ha decisions o accions acordades, retorna llistes buides.
- Si la reunió aporta poca informació, genera un document més curt sense omplir-lo artificialment.
- Si proposes una prioritat que no s'ha acordat a la reunió, marca "suggested": true.
- Màxim 4 prioritats, 6 punts d'entorn, 5 conclusions clau, 3 recomanacions.
- Mateix idioma que la transcripció (normalment català o castellà)."""


# --- PDF rendering (portrait A4, formal document style) --------------------

W, H = 595, 842  # A4 portrait (pt)
MARGIN_L = 56
MARGIN_R = 56
MARGIN_T = 60
MARGIN_B = 52
NAVY = "#1a2a3a"
TEAL = "#0d6e6e"
DARK = "#1e293b"
GRAY = "#64748b"
LIGHT_GRAY = "#94a3b8"
ACCENT_BG = "#f0f7f7"
TABLE_HEAD_BG = "#1a2a3a"
TABLE_ALT_BG = "#f1f5f9"
LINE_COLOR = "#cbd5e1"
USABLE_W = W - MARGIN_L - MARGIN_R


def _fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    base = "/usr/share/fonts/truetype/dejavu/"
    try:
        pdfmetrics.registerFont(TTFont("DV", base + "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DV-B", base + "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DV-I", base + "DejaVuSans-Oblique.ttf"))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")
        return "DV", "DV-B", "DV-I"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def render_executive_pdf(data: dict, out_path: str, footer: str):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    reg, bold, italic = _fonts()

    def st(size, font=None, color=DARK, align=TA_LEFT, lead=1.45, space_after=0):
        return ParagraphStyle("s", fontName=font or reg, fontSize=size,
                              leading=size * lead, textColor=colors.HexColor(color),
                              alignment=align, spaceAfter=space_after)

    c = canvas.Canvas(out_path, pagesize=(W, H))
    title_text = str(data.get("title", "Resum executiu"))
    c.setTitle(title_text)

    pages_content = []  # list of drawable elements per page
    elements = []  # flat list, we'll paginate later

    # --- Build all content as (type, data, height) tuples ---

    def measure_para(text, style_obj, max_w=USABLE_W):
        p = Paragraph(text, style_obj)
        w, h = p.wrap(max_w, 10000)
        return p, h

    # Section 1: Context
    elements.append(("section", "1. Context de la reunió", None))
    elements.append(("title", title_text, None))

    date_str = data.get("date")
    participants = data.get("participants", [])
    parts_text = ""
    if date_str:
        parts_text += f"<b>Data:</b> {escape(str(date_str))}<br/>"
    if participants:
        plist = []
        for p in participants:
            name = escape(str(p.get("name", "")))
            role = escape(str(p.get("role", "")))
            plist.append(f"{name} ({role})" if role else name)
        parts_text += f"<b>Participants:</b> {', '.join(plist)}<br/>"
    objective = data.get("objective", "")
    if objective:
        parts_text += f"<b>Objectiu:</b> {escape(str(objective))}"
    if parts_text:
        elements.append(("para", parts_text, st(9, lead=1.5)))

    # Section 2: Executive summary
    summary = data.get("executive_summary", "")
    conclusions = data.get("key_conclusions", [])
    if summary or conclusions:
        elements.append(("section", "2. Síntesi executiva", None))
        if summary:
            elements.append(("para", escape(str(summary)), st(9.5, lead=1.5)))
        if conclusions:
            bullets = "".join(f"<br/>• {escape(str(c_))}" for c_ in conclusions)
            elements.append(("para", f"<b>Conclusions clau:</b>{bullets}", st(9, lead=1.55)))

    # Section 3: Environment
    env = data.get("environment", [])
    if env:
        elements.append(("section", "3. Coneixement rellevant de l'entorn", None))
        for item in env[:6]:
            topic = escape(str(item.get("topic", "")))
            situation = escape(str(item.get("situation", "")))
            implication = escape(str(item.get("implication", "")))
            text = f"<b>{topic}.</b> {situation}"
            if implication:
                text += f" <i>Implicació: {implication}</i>"
            elements.append(("para", text, st(9, lead=1.5)))

    # Section 4: Priorities
    priorities = data.get("priorities", [])
    if priorities:
        elements.append(("section", "4. Prioritats i riscos", None))
        for i, pri in enumerate(priorities[:4], 1):
            issue = escape(str(pri.get("issue", "")))
            impact = escape(str(pri.get("impact", "")))
            urgency = escape(str(pri.get("urgency", "")))
            suggested = pri.get("suggested", False)
            label = " (prioritat suggerida)" if suggested else ""
            text = f"<b>P{i}. {issue}{label}</b><br/>"
            text += f"Impacte: {impact}"
            if urgency:
                text += f"<br/>Urgència: {urgency}"
            elements.append(("para", text, st(9, lead=1.5)))

    # Section 5: Decisions and next steps
    decisions = data.get("decisions", [])
    action_items = data.get("action_items", [])
    recommendations = data.get("recommendations", [])
    if decisions or action_items or recommendations:
        elements.append(("section", "5. Decisions i pròxims passos", None))

    if decisions:
        bullets = "".join(f"<br/>• {escape(str(d))}" for d in decisions)
        elements.append(("para", f"<b>Decisions explícites:</b>{bullets}", st(9, lead=1.5)))
    elif action_items:
        elements.append(("para", "<i>No s'han pres decisions explícites en aquesta reunió.</i>", st(9, color=GRAY)))

    if action_items:
        elements.append(("table", action_items, None))

    if recommendations:
        elements.append(("subsection", "Recomanacions per completar el diagnòstic", None))
        bullets = "".join(f"<br/>• {escape(str(r))}" for r in recommendations[:3])
        elements.append(("para", bullets.lstrip("<br/>"), st(9, color=GRAY, lead=1.5)))

    # --- Render to PDF with pagination ---

    page_num = [1]
    total_pages = [1]  # will be updated
    y = [H - MARGIN_T]

    def draw_header():
        c.setStrokeColor(colors.HexColor(TEAL))
        c.setLineWidth(2.5)
        c.line(MARGIN_L, H - 28, W - MARGIN_R, H - 28)
        c.setFont(reg, 7)
        c.setFillColor(colors.HexColor(LIGHT_GRAY))
        c.drawString(MARGIN_L, H - 24, "RESUM EXECUTIU · CONFIDENCIAL")

    def draw_footer():
        c.setStrokeColor(colors.HexColor(LINE_COLOR))
        c.setLineWidth(0.5)
        c.line(MARGIN_L, MARGIN_B - 4, W - MARGIN_R, MARGIN_B - 4)
        c.setFont(reg, 6.5)
        c.setFillColor(colors.HexColor(LIGHT_GRAY))
        c.drawString(MARGIN_L, MARGIN_B - 16, footer)
        c.setFont(bold, 7)
        c.drawRightString(W - MARGIN_R, MARGIN_B - 16, f"{page_num[0]}")

    def new_page():
        draw_footer()
        c.showPage()
        page_num[0] += 1
        y[0] = H - MARGIN_T
        draw_header()
        y[0] -= 18

    def ensure_space(needed):
        if y[0] - needed < MARGIN_B + 10:
            new_page()

    draw_header()
    y[0] -= 18

    for etype, edata, estyle in elements:
        if etype == "title":
            ensure_space(40)
            p, h = measure_para(escape(str(edata)), st(16, bold, NAVY, lead=1.25))
            p.drawOn(c, MARGIN_L, y[0] - h)
            y[0] -= h + 8

        elif etype == "section":
            ensure_space(35)
            y[0] -= 10
            c.setStrokeColor(colors.HexColor(TEAL))
            c.setLineWidth(1.5)
            c.line(MARGIN_L, y[0], MARGIN_L + 24, y[0])
            y[0] -= 4
            p, h = measure_para(escape(str(edata)), st(11, bold, NAVY, lead=1.2))
            p.drawOn(c, MARGIN_L, y[0] - h)
            y[0] -= h + 6

        elif etype == "subsection":
            ensure_space(25)
            y[0] -= 6
            p, h = measure_para(escape(str(edata)), st(9.5, bold, TEAL, lead=1.2))
            p.drawOn(c, MARGIN_L, y[0] - h)
            y[0] -= h + 4

        elif etype == "para":
            p, h = measure_para(str(edata), estyle)
            ensure_space(h + 4)
            p.drawOn(c, MARGIN_L, y[0] - h)
            y[0] -= h + 5

        elif etype == "table":
            rows_data = edata
            head_st = st(8, bold, "#ffffff")
            cell_st = st(8)
            tdata = [
                [Paragraph("<b>Acció</b>", head_st),
                 Paragraph("<b>Responsable</b>", head_st),
                 Paragraph("<b>Termini</b>", head_st)]
            ]
            for item in rows_data:
                tdata.append([
                    Paragraph(escape(str(item.get("action", ""))), cell_st),
                    Paragraph(escape(str(item.get("owner", "pendent d'assignar"))), cell_st),
                    Paragraph(escape(str(item.get("deadline", "no acordat"))), cell_st),
                ])
            col_w = [USABLE_W * 0.55, USABLE_W * 0.25, USABLE_W * 0.20]
            tbl = Table(tdata, colWidths=col_w, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TABLE_HEAD_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(TABLE_ALT_BG)]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE_COLOR)),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.HexColor(TABLE_HEAD_BG)),
            ]))
            _, tblh = tbl.wrap(USABLE_W, 10000)
            ensure_space(tblh + 8)
            y[0] -= 4
            tbl.drawOn(c, MARGIN_L, y[0] - tblh)
            y[0] -= tblh + 8

    draw_footer()
    c.save()
