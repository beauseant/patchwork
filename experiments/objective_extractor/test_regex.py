import re
import unittest

ES_HEADS = r"""
    objeto\s+de\s+la\s+contrataci[oó]n
  | objeto\s+del\s+contrato
  | objeto\s+dei\s+contrato
  | objeto\s+del\s+procedimiento\s+de\s+contrataci[oó]n
  | informaci[oó]n\s+sobre\s+el\s+procedimiento\s+de\s+contrataci[oó]n
  | objetivos?\s+del\s+contrato
  | objeto\s+del\s+pliego
"""

CA_HEADS = r"""
    objecte\s+de\s+la\s+contractaci[oó]
  | objecte\s+del\s+contracte
  | objecte\s+del\s+procediment\s+de\s+contractaci[oó]n?
  | informaci[oó]?\s+sobre\s+el\s+procediment\s+de\s+contractaci[oó]n?
  | objectius?\s+del\s+contracte
  | objecte\s+del\s+plec
"""

GL_HEADS = r"""
    obxecto\s+da\s+contrataci[oó]n
  | obxecto\s+do\s+contrato
  | obxecto\s+do\s+prego
"""

EU_HEADS = r"""
    kontratuaren\s+xedea
  | kontratazioaren\s+xedea
  | kontratuaren\s+helburua
  | kontratazioaren\s+helburua
"""

# Intro / verbs
ES_VERBS = r"""
    el\s+presente\s+(?:pliego|proyecto|contrato)\s+(?:tiene|tendr[aá])\s+por\s+objeto
  | tiene\s+por\s+objeto
  | definir\s+las\s+obras\s+de
  | suministro\s+de
  | el\s+objeto\s+es
  | el\s+objetivo\s+es
"""

CA_VERBS = r"""
    el\s+present\s+(?:plec|projecte|contracte)\s+(?:t[eé]|tindr[aà])\s+per\s+objecte
  | t[eé]\s+per\s+objecte
  | definir\s+les\s+obres\s+de
  | subministrament\s+de
  | l'?objecte\s+é?s?
  | l'?objectiu\s+é?s?
"""

GL_VERBS = r"""
    ten\s+por\s+obxecto
  | definir\s+as\s+obras\s+de
  | subministraci[oó]n\s+de
  | o\s+obxecto\s+é?s
  | o\s+obxectivo\s+é?s
"""

EU_VERBS = r"""
    xedea\s+da
  | helburua\s+da
  | lanak\s+definitzea
  | hornidura\s+de
"""

RE_ANCHOR = re.compile(
    rf"""
    \b(
        {ES_HEADS}
      | {ES_VERBS}
      | {CA_HEADS}
      | {CA_VERBS}
      | {GL_HEADS}
      | {GL_VERBS}
      | {EU_HEADS}
      | {EU_VERBS}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

END = r"(?=[\.\n;]|$)"

def after(pattern: str) -> str:
    return rf"(?:{pattern})[ \t]*[:\-]?[ \t]*([^\n\.;]+)"

PATTERNS_ANCHOR = [
    # ES
    after(r"el\s+presente\s+(?:pliego|proyecto|contrato)\s+(?:tiene|tendr[aá])\s+por\s+objeto"),
    after(r"tiene\s+por\s+objeto"),
    after(r"el\s+objeto\s+es"),
    after(r"el\s+objetivo\s+es"),
    after(r"definir\s+las\s+obras\s+de"),
    after(r"suministro\s+de"),
    # CA
    after(r"el\s+present\s+(?:plec|projecte|contracte)\s+(?:t[eé]|tindr[aà])\s+per\s+objecte"),
    after(r"t[eé]\s+per\s+objecte"),
    after(r"l'?objecte\s+é?s"),
    after(r"l'?objectiu\s+é?s"),
    after(r"definir\s+les\s+obres\s+de"),
    after(r"subministrament\s+de"),
    # GL
    after(r"ten\s+por\s+obxecto"),
    after(r"o\s+obxecto\s+é?s"),
    after(r"o\s+obxectivo\s+é?s"),
    after(r"definir\s+as\s+obras\s+de"),
    after(r"subministraci[oó]n\s+de"),
    # EU
    after(r"xedea\s+da"),
    after(r"helburua\s+da"),
    after(r"lanak\s+definitzea"),
    after(r"hornidura\s+de"),
]

RE_BAD = re.compile(
    r'\b(art[ií]culo|cap[ií]tulo|normativa|legislaci[oó]n|obligaciones?|protecci[oó]n\s+de\s+datos|'
    r'control\s+de\s+calidad|prevenci[oó]n|seguridad\s+y\s+salud|revisi[oó]n\s+de\s+precios|'
    r'valoraci[oó]n|abono\s+de\s+las\s+obras|maquinaria|medios\s+personales|director[a]?\s+de\s+obra|D\.?O\.?)\b',
    re.IGNORECASE,
)

RE_DOT_LEADER = re.compile(r'\.{5,}')
RE_TOC_LINE = re.compile(
    r'^\s*(?:\d+(?:[\.\s]\d+){0,3})\s*[-–.]?\s*[A-ZÁÉÍÓÚÑa-záéíóúñ][^.\n]{2,}\.{3,}\s*\d+\s*$'
)
RE_NUMBERED_HEADER = re.compile(
    r'^\s*(?:\d+(?:[\.\-\s]\d+){0,4})\s*[-–\.]*\s*[A-ZÁÉÍÓÚÑa-záéíóúñ].{0,120}$'
)
RE_MACHINERY_LINE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*\.-?\s*)?[A-ZÁÉÍÓÚÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÑ]{3,}){0,6}\.?(\s*\.)*\s*$'
)
RE_PAGE_ONLY = re.compile(r'^\s*\d+\s*(?:/\s*\d+)?\s*$')

# Minimal versions of the functions that rely on the regexes
def _find_object_snippet(text: str, min_length: int = 15, max_length: int = 350):
    t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'[ \t\f\v]+', ' ', t)   # collapse spaces/tabs/etc, but NOT '\n'
    t = re.sub(r'\n+', '\n', t).strip()
    
    for pat in PATTERNS_ANCHOR:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            cand = re.sub(r'\s+', ' ', m.group(1).strip())
            if min_length <= len(cand) <= max_length:
                return cand
    return None

def _noise_score(context_candidate: str, min_denominator: int = 8, words_per_bad_hit: int = 30):
    if not context_candidate:
        return 1.0
    bad_hits = len(RE_BAD.findall(context_candidate))
    words = max(1, len(re.findall(r'\w+', context_candidate)))
    ratio = min(1.0, bad_hits / max(min_denominator, words / words_per_bad_hit))
    return ratio

def _strip_toc_and_equipment(text: str, dot_leader_line_max_length=200, numbered_header_max_length=80) -> str:
    cleaned_lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if RE_ANCHOR.search(line):
            if RE_TOC_LINE.match(line) or RE_PAGE_ONLY.match(line):
                continue
            cleaned_lines.append(raw)
            continue
        if RE_PAGE_ONLY.match(line):
            continue
        if RE_TOC_LINE.match(line):
            continue
        if RE_DOT_LEADER.search(line) and len(line) < dot_leader_line_max_length:
            continue
        if RE_MACHINERY_LINE.match(line):
            continue
        if RE_NUMBERED_HEADER.match(line) and len(line) <= numbered_header_max_length:
            continue
        cleaned_lines.append(raw)
    out = "\n".join(cleaned_lines)
    out = re.sub(r'\.{4,}', '…', out)
    return out

def _purpose_signal(text: str) -> float:
    score = 0.0
    if RE_ANCHOR.search(text or ''):
        score += 1.0
    if RE_BAD.search(text or ''):
        score -= 0.5
    return max(0.0, min(1.0, score))

################################################################################
# Tests
################################################################################
class TestAnchorDetection(unittest.TestCase):
    def test_es_anchor_detects(self):
        s = "El presente pliego tiene por objeto el suministro de 20 farolas LED."
        self.assertIsNotNone(RE_ANCHOR.search(s))

    def test_ca_anchor_detects(self):
        s = "El present plec té per objecte la millora de vies urbanes."
        self.assertIsNotNone(RE_ANCHOR.search(s))

    def test_gl_anchor_detects(self):
        s = "Ten por obxecto a subministración de material eléctrico."
        self.assertIsNotNone(RE_ANCHOR.search(s))

    def test_eu_anchor_detects(self):
        s = "Kontratuaren xedea da hiri bideen hobekuntza."
        self.assertIsNotNone(RE_ANCHOR.search(s))

    def test_non_anchor_no_match(self):
        s = "Este documento describe condiciones generales sin especificar el objeto."
        self.assertIsNone(RE_ANCHOR.search(s))


class TestPatternsAnchorExtraction(unittest.TestCase):
    def _extract(self, text):
        return _find_object_snippet(text, min_length=3, max_length=350)  # relaxed for tests

    def test_es_extract_until_period(self):
        s = "Tiene por objeto: la mejora de la vía pública."
        self.assertEqual(self._extract(s), "la mejora de la vía pública")

    def test_es_extract_until_newline(self):
        s = "El objeto es la renovación de luminarias\ny otros trabajos."
        self.assertEqual(self._extract(s), "la renovación de luminarias")

    def test_es_extract_until_semicolon(self):
        s = "El objetivo es pavimentar la calle; incluyendo aceras."
        self.assertEqual(self._extract(s), "pavimentar la calle")

    def test_ca_extract(self):
        s = "L'objecte és la millora del clavegueram."
        self.assertEqual(self._extract(s), "la millora del clavegueram")

    def test_gl_extract(self):
        s = "O obxecto é: a subministración de papeleiras municipais."
        self.assertEqual(self._extract(s), "a subministración de papeleiras municipais")

    def test_eu_extract(self):
        s = "Helburua da hiri argiteria berritzea."
        self.assertEqual(self._extract(s), "hiri argiteria berritzea")

    def test_too_short_returns_none(self):
        s = "Tiene por objeto: X."
        self.assertIsNone(_find_object_snippet(s, min_length=5, max_length=350))

    def test_too_long_returns_none(self):
        long_obj = "x" * 400
        s = "El objeto es " + long_obj
        self.assertIsNone(_find_object_snippet(s, min_length=1, max_length=350))


class TestNoiseAndPurpose(unittest.TestCase):
    def test_noise_score_zero_when_clean(self):
        s = "La mejora del firme y reposición de marcas viales."
        self.assertAlmostEqual(_noise_score(s), 0.0, places=6)

    def test_noise_score_increases_with_bad_terms(self):
        s = "Artículo 3. Seguridad y salud en el trabajo. Revisión de precios."
        self.assertGreater(_noise_score(s), 0.0)

    def test_purpose_signal_anchor_only(self):
        s = "Tiene por objeto la pavimentación del vial."
        self.assertEqual(_purpose_signal(s), 1.0)

    def test_purpose_signal_anchor_plus_bad(self):
        s = "Tiene por objeto la pavimentación. Artículo 5: normativa aplicable."
        # 1.0 (anchor) - 0.5 (bad) = 0.5, clamped to [0,1]
        self.assertAlmostEqual(_purpose_signal(s), 0.5, places=6)

    def test_purpose_signal_bad_only(self):
        s = "Capítulo 2. Normativa y legislación aplicable."
        self.assertEqual(_purpose_signal(s), 0.0)


class TestTOCAndCleanup(unittest.TestCase):
    def test_toc_line_detects(self):
        line = "1.2 Objeto del contrato..........................15"
        self.assertIsNotNone(RE_TOC_LINE.match(line))

    def test_numbered_header_detects(self):
        line = "2.3 Condiciones generales"
        self.assertIsNotNone(RE_NUMBERED_HEADER.match(line))

    def test_machinery_line_detects(self):
        line = "EXCAVADORA HIDRÁULICA"
        self.assertIsNotNone(RE_MACHINERY_LINE.match(line))

    def test_page_only_detects(self):
        self.assertIsNotNone(RE_PAGE_ONLY.match("12"))
        self.assertIsNotNone(RE_PAGE_ONLY.match("12/45"))

    def test_dot_leader_detects(self):
        self.assertIsNotNone(RE_DOT_LEADER.search("Título.........."))

    def test_strip_removes_toc_machinery_page_headers_but_keeps_anchor(self):
        text = "\n".join([
            "1.2 Objeto del contrato......................15",  # TOC -> remove
            "12",                                            # page only -> remove
            "EXCAVADORA HIDRÁULICA",                         # machinery -> remove
            "2.3 Condiciones generales",                     # numbered header (short) -> remove
            "Tiene por objeto la renovación del alumbrado.", # anchor -> keep
            "Texto normal sin patrones.",                    # keep
            "Título.........."                               # dot leader short -> remove
        ])
        cleaned = _strip_toc_and_equipment(text)
        self.assertIn("Tiene por objeto la renovación del alumbrado.", cleaned)
        self.assertIn("Texto normal sin patrones.", cleaned)
        self.assertNotIn("Objeto del contrato......................15", cleaned)
        self.assertNotIn("12", cleaned)
        self.assertNotIn("EXCAVADORA HIDRÁULICA", cleaned)
        self.assertNotIn("2.3 Condiciones generales", cleaned)
        self.assertNotIn("Título..........", cleaned)

    def test_strip_does_not_remove_anchor_lines_that_are_not_toc(self):
        text = "Objeto del contrato y finalidad general del mismo"
        # It's a heading-like phrase but not a TOC line -> keep
        cleaned = _strip_toc_and_equipment(text)
        self.assertEqual(cleaned, text)


class TestEndBoundary(unittest.TestCase):
    def test_end_at_period(self):
        s = "El objeto es la reparación de la cubierta. Más detalles."
        self.assertEqual(_find_object_snippet(s), "la reparación de la cubierta")

    def test_end_at_newline(self):
        s = "El objetivo es renovar la red\nIncluye pruebas."
        self.assertEqual(_find_object_snippet(s), "renovar la red")

    def test_end_at_semicolon(self):
        s = "El presente contrato tiene por objeto: la sustitución de calderas; y controles."
        self.assertEqual(_find_object_snippet(s), "la sustitución de calderas")

    def test_end_at_end_of_text(self):
        s = "Tiene por objeto la limpieza viaria"
        self.assertEqual(_find_object_snippet(s), "la limpieza viaria")


if __name__ == "__main__":
    unittest.main(verbosity=2)
