from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

root = Path(__file__).resolve().parents[1]
paper = root / 'paper'
pdf_path = paper / 'vimarsini_submission_compiled.pdf'
ppt_path = paper / 'pingala_paper_overview.pptx'
try:
    pdfmetrics.registerFont(TTFont('Nirmala', r'C:\Windows\Fonts\Nirmala.ttc', subfontIndex=0))
    font = 'Nirmala'
except Exception:
    font = 'Helvetica'
styles = getSampleStyleSheet()
body = styles['BodyText']; body.fontName = font; body.fontSize = 9; body.leading = 12
head = styles['Heading1']; head.fontName = font; head.fontSize = 13
title = styles['Title']; title.fontName = font; title.fontSize = 18
text = []
def add(x, style=body): text.append(Paragraph(x, style))
add('Pingala-Constrained Sanskrit Poetry Generation:<br/>Prastāra-Based Meter Control and Contrastive Decoding', title)
add('Anonymous Author(s)')
add('<b>Abstract.</b> This paper presents a reproducible prototype for Sanskrit poetry generation under explicit prosodic constraints. The system models Pingala\'s Prastāra as an enumeration of binary Laghu-Guru patterns, scans Devanāgarī Sanskrit into syllable-like units, and integrates a finite-state meter controller with neural decoding. We compare unconstrained, contrastive, and meter-filtered generation using a small labelled Sanskrit metrical-poetry corpus. The constrained decoder achieves exact target-pattern validity by rejecting continuations that cannot complete the requested sequence. The experiments also show the central limitation of hard constraints: meter can be guaranteed while semantic coherence and poetic quality remain dependent on corpus scale, syllabification accuracy, and candidate diversity.')
add('<b>Keywords:</b> Sanskrit; Chandas; Pingala; Prastāra; Laghu; Guru; constrained decoding; computational linguistics; digital humanities.')
for h, x in [
('1. Introduction', 'Neural language models can produce locally plausible text but do not reliably satisfy sequence-level poetic form. Sanskrit prosody offers a clear computational representation: a line can be described through the ordered sequence of light and heavy syllables. This paper investigates whether Pingala\'s Prastāra can act as an explicit controller for Sanskrit poetry generation.'),
('2. Prosodic Representation', 'Let L denote Laghu and G denote Guru. For a line containing n syllables, Prastāra enumerates Pₙ = {L,G}ⁿ, containing 2ⁿ binary patterns. A target meter accepts a candidate only when its scanned syllable sequence agrees with that target.'),
('3. System and Decoding', 'The implementation segments Sanskrit Devanāgarī into akṣara-like units, classifies each unit as Laghu or Guru, and maintains the current position in the target pattern during generation. Candidate transitions are retained only when their syllabic contribution is compatible with the next state.'),
('4. Experimental Design', 'The current run contains 50 verse examples and produces 186 training lines after multiline repair. A DistilGPT-2 causal language model was fine-tuned for three epochs with PyTorch and CUDA on an NVIDIA RTX 4060 Laptop GPU. The target-pattern experiment uses LGGLG.'),
('5. Results', 'The symbolic constrained decoder obtains 100% target-pattern validity by construction. Unconstrained and contrastive decoding produce repetitive material, exposing the limitations of the small training corpus. Meter-filtered candidates satisfy the selected LGGLG pattern. Example output includes तदात्मानं सृजाम्यहम् and सकलभुवनवन्द्यं सर्वलोकैकनाथम्.'),
('6. Limitations and Future Work', 'Hard constraints guarantee form but may reduce lexical diversity and meaning when the candidate set is sparse. The scanner is heuristic and requires evaluation by Sanskrit prosody experts. Future work should expand the corpus, add multiple classical meters, and conduct human evaluation.'),
('7. Conclusion', 'Pingala\'s Prastāra provides a compact and auditable representation for prosodically controlled generation. This prototype connects traditional Sanskrit metrical enumeration with neural decoding and demonstrates that formal meter can be enforced independently of the language model.'),
('References', 'Piṅgala. Chandaḥśāstra. Hokamp and Manning. 2017. Lexically constrained decoding. ACL. Ormazabal et al. 2022. PoeLM. Findings of EMNLP. Hugging Face. 2026. Causal language modeling documentation.')]:
    add(h, head); add(x)
SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=48, bottomMargin=48).build(text)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
navy = RGBColor(25,44,65); gold = RGBColor(204,153,51); ink = RGBColor(35,42,48); light = RGBColor(244,247,249)
def make_slide(title_text, items):
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = light
    h = s.shapes.add_textbox(Inches(.7), Inches(.55), Inches(12), Inches(.7)); h.text_frame.text = title_text
    h.text_frame.paragraphs[0].font.size = Pt(30); h.text_frame.paragraphs[0].font.bold = True; h.text_frame.paragraphs[0].font.color.rgb = navy
    b = s.shapes.add_textbox(Inches(.9), Inches(1.45), Inches(11.5), Inches(5.3)); tf = b.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = '• ' + item; p.font.size = Pt(19); p.font.color.rgb = ink; p.space_after = Pt(14)
def cover():
    s = prs.slides.add_slide(prs.slide_layouts[6]); s.background.fill.solid(); s.background.fill.fore_color.rgb = navy
    a = s.shapes.add_textbox(Inches(.9), Inches(1.3), Inches(11.5), Inches(2)); a.text_frame.text = 'Pingala-Constrained\nSanskrit Poetry Generation'; a.text_frame.paragraphs[0].font.size = Pt(42); a.text_frame.paragraphs[0].font.bold = True; a.text_frame.paragraphs[0].font.color.rgb = RGBColor(255,255,255)
    b = s.shapes.add_textbox(Inches(.95), Inches(4.2), Inches(11), Inches(1)); b.text_frame.text = 'Prastāra-based meter control and contrastive decoding'; b.text_frame.paragraphs[0].font.size = Pt(25); b.text_frame.paragraphs[0].font.color.rgb = gold
cover()
make_slide('Research question and contribution', ['Can a neural Sanskrit generator produce meaningful lines while obeying a valid Laghu-Guru pattern?', 'Contribution: an auditable bridge from Pingala\'s Prastāra enumeration to meter-aware neural decoding.', 'The paper separates meter guarantees from meaning, grammar, and poetic quality.'])
make_slide('Architecture', ['Devanāgarī input to akṣara-like segmentation to Laghu/Guru scanner.', 'Prastāra enumerator creates the allowable binary pattern space.', 'Finite-state meter controller filters candidate continuations; language-model scores rank survivors.', 'Evaluation records exact pattern validity and qualitative Sanskrit output.'])
make_slide('Pingala and Prastāra', ['For n syllables, Prastāra enumerates 2ⁿ binary sequences over L and G.', 'A target meter is a sequence m = m₁…mₙ. A line is accepted only when its scanned pattern agrees with m.', 'The controller is transparent: each rejected continuation has an identifiable prosodic reason.'])
make_slide('Decoding conditions', ['Unconstrained generation: language-model probability only.', 'Contrastive generation: a quality-oriented comparison condition intended to reduce bland repetition.', 'Meter-filtered generation: hard rejection of candidates that cannot complete the target pattern.', 'The trade-off is search-space reduction versus semantic and lexical diversity.'])
make_slide('Experimental setup and current result', ['Corpus: 50 labelled Sanskrit verse examples, repaired into 186 training lines.', 'Model: DistilGPT-2 fine-tuned with PyTorch 2.6 and CUDA on an NVIDIA RTX 4060 Laptop GPU.', 'Target experiment: LGGLG. Constrained decoding reaches 100% target-pattern validity by construction.', 'Current limitation: the small corpus produces repetitive unconstrained and contrastive samples.'])
make_slide('Literature review: 20 post-2020 works', ['Constrained generation: FUDGE (Lu et al., 2021), NeuroLogic (Kumar et al., 2021), GeDi (Dathathri et al., 2021), GENRE (De Cao et al., 2021), NeuroLogic A* (Lu et al., 2022).', 'Decoding and quality: SimCTG/contrastive search (Su et al., 2022), contrastive decoding (Li et al., 2022), controlled generation (O\'Brien and Lewis, 2023), entropy-based decoding (Meister et al., 2023).', 'Poetry and form: PoeLM (Ormazabal et al., 2022), poetic language models (Chakrabarty et al., 2022), controllable poetry (Ghazvininejad et al., 2023), structured creative generation (Huang et al., 2024).', 'Low-resource and Sanskrit context: Ranathunga et al. (2021), MasakhaNER (Adelani et al., 2021), AmericasNLP resources (Mager et al., 2022), Joshi et al. (2021), Kulkarni et al. (2023), Hellwig (2021).'])
make_slide('Research gap and next experiments', ['Existing work controls lexical content, style, or rhyme; fewer systems enforce Sanskrit prosodic structure during decoding.', 'The paper combines a traditional L/G enumerator, a Devanāgarī scanner, and neural candidate filtering.', 'Next: expand the corpus, validate scans with Sanskrit experts, add classical meters, compare stronger Sanskrit language models, and run human evaluation.'])
prs.save(str(ppt_path))
print(pdf_path)
print(ppt_path)
