"""Executive CIO briefing: landscape card-based PDF deck generated from a transcript."""

import re
from xml.sax.saxutils import escape

EXEC_SYSTEM_PROMPT = """Actua com a assessor executiu d'un CIO en procés d'incorporació a una empresa. A partir de l'àudio de la reunió, genera un resum executiu en format presentació (pàgines apaïsades amb targetes), pensat perquè el CIO el llegeixi en 5 minuts i pugui prendre decisions.

PRINCIPIS
- Prioritza la informació que ajudi el CIO a prendre decisions. No facis una transcripció ni un relat cronològic.
- Cada títol de pàgina és una CONCLUSIÓ (missatge), no un tema. Ex: "Operacions i SEIDOR: consolidar i exigir", no "Proveïdors".
- Frases curtes, denses i concretes: noms, xifres, sistemes, terminis. Res genèric ni de manual.
- RIGOR: distingeix sempre la naturalesa de cada afirmació amb un prefix al començament del paràgraf:
  "R:" relat explícit de la conversa · "P:" percepció o opinió d'un participant (no contrastada) · "I:" inferència o proposta teva · "V:" element a verificar (grafies dubtoses, xifres ambigües, contradiccions).
- No inventis res. Si un nom propi pot estar mal transcrit, marca'l amb * i inclou-lo a la pàgina de verificació.
- Les xifres són orientatives tal com es diuen; indica-ho quan calgui ("prop del milió anual", interpretat en euros, a validar).
- Quan citis un moment concret, posa'n el temps entre parèntesis (mm:ss) a partir de les marques [mm:ss] de la transcripció.
- Els prefixos R:/P:/I:/V: van només als paràgrafs d'anàlisi. NO els posis a vinyetes (•), llistes numerades, línies "Resultat:", línies de focus en majúscules ni a les preguntes: en aquestes targetes la naturalesa ja la indica el títol o el subtítol.
- Títols de pàgina curts: màxim 60 caràcters.
- Mateix idioma que la transcripció.
- Si el context de l'usuari diu qui és el destinatari (p. ex. el nou CIO), orienta les propostes a ell.

CRITERIS DE RIGOR CIO
- Fes servir l'àudio com a font; no completis buits amb suposicions.
- Distingeix entre informació relatada, opinions dels participants i inferències pròpies.
- Atribueix les valoracions sensibles a qui les expressa, si es pot identificar.
- No converteixis hipòtesis, desitjos o propostes en decisions o accions acordades.
- No inventis noms, càrrecs, imports, dates ni relacions entre sistemes. Marca els elements dubtosos com a «pendent de confirmar».
- Evita repeticions, frases genèriques i detalls personals sense rellevància professional.
- Si no hi ha decisions o accions acordades, indica-ho breument.
- Si la reunió aporta poca informació, genera un document més curt sense omplir-lo artificialment.
- Si proposes una prioritat que no s'ha acordat a la reunió, identifica-la com a «prioritat suggerida».
- Diferencia importància i urgència.

ESTRUCTURA RECOMANADA (adapta-la al contingut; entre 6 i 9 pàgines):
1. Portada-resum: títol que sintetitzi la situació (conclusió-missatge del CIO); "banner" amb el mandat/missatge clau que emergeix de la conversa (2-3 frases); 6 targetes numerades "01 Títol"... "06 Títol" amb els grans eixos (30-40 paraules cadascuna, sense prefixos).
2. "Com s'ha de llegir aquest document": 4 targetes R/P/I/V explicant què és cada tipus en aquest cas concret i, a V, les grafies i dades a verificar. Tons: R i I "primary", P i V "accent".
3. Organització i persones clau: banner amb l'estructura de direcció; una targeta per persona/àrea rellevant (rol, què se'n diu, proposta).
4-5. Les 1-2 temàtiques principals (p. ex. proveïdor clau, tecnologia, ciberseguretat, adquisicions): 4-5 targetes amb abast, balanç, situació contractual/xifres i agenda de revisió (llista numerada).
6. Riscos i dependències: TAULA amb columnes ["", "Focus", "Senyal del relat", "Impacte potencial", "Evidència / resposta"]; primera columna "P1" o "P2"; 5-8 files. Nota final explicant P1/P2.
7. Agenda d'entrada / pla d'acció orientat al CIO: 3 targetes per fases (p. ex. "Primers 10 dies", "Dies 11-30", "Dies 31-60"), cadascuna amb una línia en majúscules del focus, vinyetes i "Resultat: ..."; i una targeta "full": true amb 4-6 preguntes clau numerades per al CIO.
8. Cultura, lideratge o claus d'encaix (si n'hi ha prou material): 4 targetes amb observacions sobre dinàmiques, estils de treball i expectatives.
9. Evidència localitzable: TAULA ["Temps", "Punt clau", "Conclusió i límit"] amb 5-8 moments concrets de l'àudio.

LÍMITS DE MIDA (imprescindible perquè hi càpiga):
- Màxim 6 targetes per pàgina (més una "full" opcional). Cada targeta: 25-90 paraules en total.
- Taules: màxim 8 files; cel·les de màxim 18 paraules.
- Subtítol de pàgina: una línia (màxim 18 paraules) que matisi l'abast o el límit de la pàgina.

Retorna NOMÉS un JSON vàlid (sense markdown):
{
  "kicker": "EMPRESA / TEMA EN MAJÚSCULES (curt)",
  "pages": [
    {
      "title": "Titular-conclusió",
      "subtitle": "Matís o límit de la pàgina",
      "banner": {"title": "TÍTOL CURT", "text": "Text"},
      "cards": [
        {"title": "Títol", "tone": "primary", "full": false,
         "body": ["R: paràgraf", "P: paràgraf", "• vinyeta", "1. element numerat", "Resultat: ..."]}
      ],
      "table": {"columns": ["Col1", "Col2"], "rows": [["...", "..."]]},
      "note": "Nota al peu opcional"
    }
  ]
}
"banner", "cards", "table" i "note" són opcionals a cada pàgina. Una pàgina amb "table" no porta "cards"."""


# --- PDF rendering ---------------------------------------------------------

W, H = 842, 595  # A4 landscape (pt)
MARGIN = 38
NAVY = "#13293d"
TEAL = "#0f7f7a"
AMBER = "#a86b1f"
INK = "#22313f"
GRAY = "#5b6b7b"
BG = "#f1f5f9"
ROW_ALT = "#e9eff6"
CONTENT_TOP = H - 118
CONTENT_BOTTOM = 52
GAP = 16


def _fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    base = "/usr/share/fonts/truetype/dejavu/"
    try:
        pdfmetrics.registerFont(TTFont("DV", base + "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DV-B", base + "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
        return "DV", "DV-B"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _rich(text: str, bold: str) -> str:
    """Escape text and highlight a leading R:/P:/I:/V: tag."""
    t = escape(str(text))
    # Tags are noise on list items ("I: • ...", "I: 1. ...")
    t = re.sub(r"^[RPIV]: (?=•|\d+\.|Resultat:)", "", t)
    for tag in ("R:", "P:", "I:", "V:"):
        if t.startswith(tag):
            color = AMBER if tag in ("P:", "V:") else TEAL
            return f'<font name="{bold}" color="{color}">{tag}</font>{t[2:]}'
    return t


def render_executive_pdf(data: dict, out_path: str, footer: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    reg, bold = _fonts()
    kicker = str(data.get("kicker", "")).upper()
    pages = [p for p in data.get("pages", []) if isinstance(p, dict)]
    c = canvas.Canvas(out_path, pagesize=(W, H))
    c.setTitle(pages[0].get("title", "Informe executiu") if pages else "Informe executiu")

    def style(size, font=None, color=INK, lead=1.38):
        return ParagraphStyle("s", fontName=font or reg, fontSize=size,
                              leading=size * lead, textColor=colors.HexColor(color))

    def para(text, st):
        return Paragraph(text, st)

    def card_parts(card, fs, width):
        tone = AMBER if card.get("tone") == "accent" else TEAL
        inner = width - 34
        title = para(escape(str(card.get("title", ""))), style(fs + 1.5, bold, tone, 1.25))
        body = []
        for item in card.get("body", []) or []:
            body.append(para(_rich(item, bold), style(fs)))
        h = 18 + title.wrap(inner, 1e4)[1] + 8
        for p in body:
            h += p.wrap(inner, 1e4)[1] + fs * 0.55
        return tone, title, body, h + 14

    def draw_box(x, y_top, w, h, bar):
        c.setFillColor(colors.white)
        c.roundRect(x, y_top - h, w, h, 8, stroke=0, fill=1)
        c.setFillColor(colors.HexColor(bar))
        c.rect(x, y_top - h, 3.5, h, stroke=0, fill=1)

    def draw_card(card_parts_, x, y_top, w, h, fs):
        tone, title, body, _ = card_parts_
        draw_box(x, y_top, w, h, tone)
        inner = w - 34
        y = y_top - 18
        _, th = title.wrap(inner, 1e4)
        title.drawOn(c, x + 16, y - th)
        y -= th + 8
        for p in body:
            _, ph = p.wrap(inner, 1e4)
            p.drawOn(c, x + 16, y - ph)
            y -= ph + fs * 0.55

    def rows_for(cards):
        regular = [cd for cd in cards if not cd.get("full")]
        full = [cd for cd in cards if cd.get("full")]
        n = len(regular)
        sizes = {1: [1], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2], 6: [3, 3]}.get(
            n, [3] * (n // 3) + ([n % 3] if n % 3 else []))
        rows, i = [], 0
        for s in sizes:
            rows.append(regular[i:i + s])
            i += s
        return rows + [[cd] for cd in full]

    total = len(pages)
    for num, page in enumerate(pages, 1):
        c.setFillColor(colors.HexColor(BG))
        c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(colors.HexColor(NAVY))
        c.rect(0, H - 10, W, 10, stroke=0, fill=1)

        c.setFillColor(colors.HexColor(TEAL))
        c.setFont(bold, 8.5)
        c.drawString(MARGIN, H - 38, kicker)
        tp = para(escape(str(page.get("title", ""))), style(23, bold, NAVY, 1.15))
        _, th = tp.wrap(W - 2 * MARGIN, 100)
        tp.drawOn(c, MARGIN, H - 50 - th)
        top = H - 50 - th - 8
        if page.get("subtitle"):
            sp = para(escape(str(page["subtitle"])), style(9.5, reg, GRAY))
            _, sh = sp.wrap(W - 2 * MARGIN, 60)
            sp.drawOn(c, MARGIN, top - sh)
            top -= sh
        top = min(top - 18, CONTENT_TOP)

        # footer
        c.setStrokeColor(colors.HexColor("#cfd8e3"))
        c.setLineWidth(0.6)
        c.line(MARGIN, 40, W - MARGIN, 40)
        c.setFillColor(colors.HexColor(GRAY))
        c.setFont(reg, 7)
        c.drawString(MARGIN, 28, footer)
        c.setFont(bold, 7.5)
        c.drawRightString(W - MARGIN, 28, f"{num:02d} / {total:02d}")

        bottom = CONTENT_BOTTOM
        note = page.get("note")
        note_p = None
        if note:
            note_p = para(_rich(note, bold), style(8.5, reg, GRAY))
            _, nh = note_p.wrap(W - 2 * MARGIN, 200)
            note_p.drawOn(c, MARGIN, bottom)
            bottom += nh + 10

        banner = page.get("banner") if isinstance(page.get("banner"), dict) else None
        cards = [cd for cd in page.get("cards", []) or [] if isinstance(cd, dict)]
        table = page.get("table") if isinstance(page.get("table"), dict) else None

        # Shrink font until everything fits in the available height
        for fs in (9.5, 9, 8.5, 8, 7.5, 7, 6.5):
            used = 0
            bparts = None
            if banner:
                bt = para(escape(str(banner.get("title", "")).upper()), style(fs + 0.5, bold, NAVY))
                bx = para(_rich(banner.get("text", ""), bold), style(fs + 0.5))
                bh = 16 + bt.wrap(W - 2 * MARGIN - 34, 1e4)[1] + 6 + bx.wrap(W - 2 * MARGIN - 34, 1e4)[1] + 14
                bparts = (bt, bx, bh)
                used += bh + GAP
            layout = []
            for row in rows_for(cards):
                cw = (W - 2 * MARGIN - GAP * (len(row) - 1)) / len(row)
                parts = [card_parts(cd, fs, cw) for cd in row]
                rh = max(p[3] for p in parts)
                layout.append((cw, parts, rh))
                used += rh + GAP
            tbl = None
            if table:
                cols = [str(x) for x in table.get("columns", [])]
                rows = [[str(x) for x in r] for r in table.get("rows", []) if isinstance(r, list)]
                ncol = max([len(cols)] + [len(r) for r in rows]) or 1
                head_st = style(fs, bold, TEAL)
                cell_st = style(fs - 0.5)
                tag_st = style(fs - 0.5, bold, AMBER)
                key_st = style(fs - 0.5, bold, INK)
                tdata = [[para(escape(x), head_st) for x in cols + [""] * (ncol - len(cols))]]
                first_is_tag = all(len(r) and len(r[0]) <= 3 for r in rows)
                for r in rows:
                    r = r + [""] * (ncol - len(r))
                    cells = []
                    for j, x in enumerate(r):
                        st_ = tag_st if (j == 0 and first_is_tag) else (
                            key_st if j <= 1 else cell_st)
                        cells.append(para(_rich(x, bold), st_))
                    tdata.append(cells)
                avail_w = W - 2 * MARGIN
                if first_is_tag:
                    rest = (avail_w - 40) / (ncol - 1) if ncol > 1 else avail_w
                    widths = [40] + [rest * (0.8 if j == 1 else 1) for j in range(1, ncol)]
                    widths = [widths[0]] + [w_ * (avail_w - 40) / sum(widths[1:]) for w_ in widths[1:]]
                elif ncol == 3:
                    widths = [avail_w * 0.14, avail_w * 0.26, avail_w * 0.60]
                else:
                    widths = [avail_w / ncol] * ncol
                tbl = Table(tdata, colWidths=widths, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(ROW_ALT)]),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("LINEBELOW", (0, 1), (-1, -1), 3, colors.HexColor(BG)),
                ]))
                _, tblh = tbl.wrap(avail_w, 1e4)
                used += tblh + GAP
            if used - GAP <= top - bottom:
                break

        slack = max(0, (top - bottom) - (used - GAP))
        extra = min(slack / max(1, len(layout)), 70) if layout else 0

        y = top
        if bparts:
            bt, bx, bh = bparts
            draw_box(MARGIN, y, W - 2 * MARGIN, bh, NAVY)
            yy = y - 16
            _, h1 = bt.wrap(W - 2 * MARGIN - 34, 1e4)
            bt.drawOn(c, MARGIN + 16, yy - h1)
            yy -= h1 + 6
            _, h2 = bx.wrap(W - 2 * MARGIN - 34, 1e4)
            bx.drawOn(c, MARGIN + 16, yy - h2)
            y -= bh + GAP
        for cw, parts, rh in layout:
            x = MARGIN
            for p in parts:
                draw_card(p, x, y, cw, rh + extra, fs)
                x += cw + GAP
            y -= rh + extra + GAP
        if tbl:
            _, tblh = tbl.wrap(W - 2 * MARGIN, 1e4)
            tbl.drawOn(c, MARGIN, y - tblh)

        c.showPage()
    c.save()
