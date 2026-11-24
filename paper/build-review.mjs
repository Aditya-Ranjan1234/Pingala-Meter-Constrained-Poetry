import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const [modules, format] = process.argv.slice(2);
const rvce = format === 'rvce';
const run = JSON.parse(await fs.readFile(path.join(dir,'../results/sanskrit_generation.json'),'utf8'));
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(modules, '@oai', 'artifact-tool', 'dist', 'artifact_tool.mjs')
).href);
const manuscript = await fs.readFile(path.join(dir, 'review-sources.txt'), 'utf8');
const groups = [];
for (const line of manuscript.trim().split(/\r?\n/)) {
  if (!line.includes('|')) {
    groups.push({ title: line, papers: [] });
    continue;
  }
  const [authors, year, label, title, venue, id, finding] = line.split('|');
  const url = id.startsWith('https:') ? id : 'https://aclanthology.org/' + id + '/';
  groups.at(-1).papers.push({ authors, year, label, title, venue, url, finding });
}
const papers = groups.flatMap(group => group.papers);
if (papers.length < 21 || new Set(papers.map(p => p.url)).size !== papers.length) {
  throw new Error('Expected at least 21 distinct papers');
}
const escape = text => text.replace(/([&%#_])/g, '\\$1');
const references = [...papers].sort((a,b) => a.authors.localeCompare(b.authors))
  .map(p => {
    const suffix = p.label === 'Chandomitra' ? 'a' : p.label === 'Pingala' ? 'b' : '';
    return '\\item ' + escape(p.authors) + '. ' + p.year + suffix + '. ' +
      '\\href{' + p.url + '}{' + escape(p.title) + '}. \\textit{' + escape(p.venue) + '}.';
  }).join('\n');
if (!rvce) await fs.writeFile(path.join(dir,'review-references.tex'),
  '\\section*{References}\n\\begin{list}{}{\\setlength{\\leftmargin}{1em}\\setlength{\\itemindent}{-1em}\\setlength{\\itemsep}{4pt}}\n' +
  references + '\n\\item dukeofmtnview. \\href{https://huggingface.co/datasets/dukeofmtnview/sanskrit-metrical-poetry}{Sanskrit Metrical Poetry}. Dataset.\n\\end{list}\n');

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = '#17273A', muted = '#435670', blue = '#165B62', light = '#F7F7F2';
const green = '#DCEDE8', warm = '#F5E9D5', gold = '#F0BE69', white = '#FFFFFF';
function tx(slide, value, x, y, w, h, size, options = {}) {
  const shape = slide.shapes.add({
    name: value.slice(0, 70), geometry: options.geometry || 'textbox',
    position: { left:x, top:y, width:w, height:h },
    fill: options.fill || 'none',
    line: { fill: 'none', width: 0 },
  });
  shape.text = value;
  shape.text.style = { fontFamily:options.font || 'Aptos', fontSize:size,
    color:options.color || ink, bold:!!options.bold, alignment:options.align || 'left',
    verticalAlignment:'middle', wrap: options.wrap === false ? 'none':'square',
    insets:{left:options.fill ? 10 : 0,right:options.fill ? 10 : 0,top:0,bottom:0} };
  return shape;
}
function slide(title, chapter) {
  const s = deck.slides.add();
  s.background.fill = light;
  tx(s,chapter.toUpperCase(),55,32,1160,25,19,{color:blue,bold:true});
  tx(s,title,55,70,1170,64,46,{bold:true});
  tx(s,rvce?'RV COLLEGE OF ENGINEERING, BENGALURU':'SANSKRIT PROSODY RESEARCH PILOT',55,666,950,24,14,{color:muted});
  tx(s,String(deck.slides.items.length).padStart(2,'0'),1150,666,72,24,16,{align:'right',color:muted});
  return s;
}
function box(s, text, x, y, w, h, fill = green) {
  return tx(s,text,x,y,w,h,24,{geometry:'roundRect',fill,align:'center',bold:true});
}
function arrow(s, glyph, x, y, w=48) {
  return tx(s,glyph,x,y,w,48,32,{color:blue,bold:true,align:'center'});
}
{
  const s = deck.slides.add(); s.background.fill = ink;
  tx(s,rvce?'RV COLLEGE OF ENGINEERING, BENGALURU':'SANSKRIT PROSODY RESEARCH PILOT',68,72,1120,30,22,{bold:true,color:gold});
  if (rvce) tx(s,'Aditya Ranjan     ·     Gnanendra Naidu N     ·     Nischitha S K',68,627,1150,38,23,{color:white});
  tx(s,'Prastāra and metrical control\nin Sanskrit text generation',68,151,1140,190,65,{bold:true,color:white});
  tx(s,'Does a syllable-weight constraint help a language model produce meaningful verse?',68,370,1060,85,31,{color:'#D7E3E9'});
  ['L','G','G','L','G'].forEach((letter,i) =>
    tx(s,letter,68+i*81,531,67,65,36,{fill:i%2?green:warm,align:'center',bold:true}));
  tx(s,'Example target pattern',495,540,450,46,23,{color:white});
  s.speakerNotes.text='Research scope: enumeration, scanner-based corpus retrieval, and two neural baselines. The target is a diagnostic binary sequence.';
}
{
  const s=slide('System architecture','Implementation');
  tx(s,'Language model path',55,150,560,40,29,{bold:true,color:blue});
  tx(s,'Corpus matching path',690,150,540,40,29,{bold:true,color:blue});
  const left=['50 verse examples → 186 training lines','DistilGPT-2 fine-tuning on CUDA','Nucleus sampling or local contrastive search','2 saved model continuations'];
  const right=['Read corpus segments','Scan each segment for L / G weights','Compare with fixed target LGGLG','2 unique corpus matches'];
  for(let i=0;i<4;i++){
    box(s,left[i],55,202+i*103,535,65);
    box(s,right[i],690,202+i*103,535,65,warm);
    if(i<3){arrow(s,'↓',296,264+i*103);arrow(s,'↓',933,264+i*103);}
  }
  s.speakerNotes.text='Implementation sources: src/prepare_sanskrit.py, src/train_lm.py, src/generate_sanskrit.py and results/sanskrit_generation.json. Both paths are separate in the saved experiment.';
}
{
  const s=slide('Prastāra and the meter filter','Algorithm');
  tx(s,'32 possible strings at length five',55,154,1160,39,30,{color:blue,bold:true});
  tx(s,'The Sanskrit experiment sets LGGLG directly. The enumerator supplies the general L/G pattern space.',55,202,1130,53,23);
  box(s,'Corpus segment',65,297,275,72,warm);
  arrow(s,'→',342,307,95);
  box(s,'Devanāgarī scanner',440,297,300,72);
  arrow(s,'→',750,307,95);
  box(s,'Equals LGGLG?',850,297,305,72,warm);
  arrow(s,'↙',911,375);
  arrow(s,'↓',1071,375);
  tx(s,'YES',843,423,130,39,22,{color:blue,bold:true,align:'center'});
  tx(s,'NO',1044,423,105,39,22,{color:blue,bold:true,align:'center'});
  box(s,'Keep once',774,473,218,71);
  box(s,'Skip',1010,473,165,71,warm);
  if (rvce) {
    tx(s,'Worked example  ·  '+run.constrained[0]+'  →  तदा / त्मा / नं / सृ / जाम्यहम्  →  LGGLG  →  Keep',60,568,1150,42,24,{font:'Nirmala UI',color:blue,bold:true});
    tx(s,'These are scanner groups. The heuristic can merge pronounced syllables.',60,613,1135,34,20,{color:muted});
  } else tx(s,'The scanner groups written characters and can miss syllables with inherent vowels.',60,577,1135,48,24,{color:muted});
  s.speakerNotes.text='Sources: src/meter.py, src/sanskrit.py, src/generate_sanskrit.py. Exact match is against the internal scanner, which needs validation against linguistic syllables.';
}
{
  const s=slide('What the saved experiment produced','Results');
  if (rvce) {
    const examples=[
      ['Nucleus sampling  ·  GGGGGGGGGG  ·  No match',run.unconstrained],
      ['Local contrastive search  ·  GGGGGGGG  ·  No match',run.contrastive],
      ['Corpus retrieval  ·  LGGLG for both  ·  Two matches',run.constrained.join('     |     ')],
    ];
    examples.forEach(([header,output],i)=>{
      const y=155+i*137;
      tx(s,header,55,y,1170,42,23,{fill:ink,bold:true,color:white});
      tx(s,output,55,y+43,1170,68,25,{font:'Nirmala UI',fill:i===2?green:'#EBEFF1'});
    });
    tx(s,'The corpus filter returned two existing lines. The symbol � appears in the saved nucleus output.',55,585,1160,52,21,{color:muted});
  } else {
  const headers=['Test condition','Internal scan','Matched target'];
  const offsets=[55,501,895], widths=[445,393,330];
  headers.forEach((x,i)=>tx(s,x,offsets[i],160,widths[i],52,24,{fill:ink,color:white,bold:true}));
  const rows=[['Nucleus sampling','GGGGGGGGGG','0 of 1'],['Local contrastive search','GGGGGGGG','0 of 1'],['Corpus retrieval','LGGLG, LGGLG','2 of 2 returned']];
  rows.forEach((row,i)=>row.forEach((x,j)=>tx(s,x,offsets[j],219+i*60,widths[j],55,25,{fill:i===2?green:'#EBEFF1'})));
  tx(s,'Both model continuations repeated कर्मः.',55,423,1130,49,28,{font:'Nirmala UI',bold:true,color:blue});
  tx(s,'The 2 corpus matches passed the same scanner used to select them. One continuation per model condition gives too little evidence to compare the decoders.',55,496,1135,102,26,{color:muted});
  }
  s.speakerNotes.text='Source: results/sanskrit_generation.json. The two retrieved examples came from the corpus. Both model outputs include the prompt. The saved run contains one continuation per neural condition.';
}
if (rvce) {
  const s=slide('Training and equal-budget results','Follow-up experiment');
  tx(s,'Chandomitra: 15,433 training, 996 validation and 2,836 test lines',55,150,1170,50,30,{bold:true,color:blue});
  tx(s,'Same checkpoint and 64 fresh prompts · Up to 4,096 sampled tokens per prompt',55,211,1170,48,23);
  const values=[
    ['Measure','Generate and filter','Metre-guided'],
    ['Basic cadence matches','29 / 64','40 / 64'],
    ['Total sampled tokens','193,856','159,580'],
    ['Median tokens on 21 shared successes','1,952','589'],
  ];
  const table=s.tables.add({rows:4,columns:3,left:55,top:280,width:1170,height:228,columnWidths:[385,400,385],values});
  values.forEach((row,i)=>row.forEach((_,j)=>{
    const cell=table.getCell(i,j);
    cell.fill=i===0?ink:i%2?white:green;
    cell.text.style={fontSize:23,color:i===0?white:ink,bold:i===0};
  }));
  tx(s,'Guidance prunes incompatible prefixes at word boundaries and keeps the last two syllables provisional.',55,552,1170,64,23,{color:muted});
  s.speakerNotes.text='Source: results/guided_reader_review/automatic_summary.json. Paired exact p = 0.052. Sampled tokens do not measure elapsed time. Slides 2–4 describe the earlier pilot.';
}
for(const [index,group] of groups.entries()){
  const selected = group.papers;
  const s=slide(group.title,'Literature review  /  '+(index+1)+' of 3');
  if (rvce) {
    const values = [['Paper','Authors and year','Venue','Relevance to this study'],
      ...selected.map(p=>[p.label,p.authors+' ('+p.year+')',p.venue,p.finding])];
    const table = s.tables.add({rows:values.length,columns:4,left:55,top:157,width:1170,height:455,columnWidths:[265,251,150,504],values});
    values.forEach((row,i)=>row.forEach((_,j)=>{
      const cell=table.getCell(i,j);
      cell.fill=i===0?ink:i%2?white:green;
      cell.text.style={fontSize:20,color:i===0?white:ink,bold:i===0||j===0};
    }));
    s.speakerNotes.text=selected.map(p=>p.authors+' ('+p.year+'). '+p.title+'. '+p.venue+'. '+p.url).join('\n\n');
    continue;
  }
  tx(s,'Study',55,143,500,30,20,{bold:true,color:blue});
  tx(s,'Finding relevant to this project',570,143,650,30,20,{bold:true,color:blue});
  for(const [row,p] of group.papers.entries()){
    const y=183+row*65;
    const fill=row%2?'#EDF0ED':'#F7F7F2';
    tx(s,p.label,55,y,500,35,23,{bold:true,fill});
    tx(s,p.authors+' · '+p.year+' · '+p.venue,55,y+34,500,23,17,{color:muted,fill});
    tx(s,p.finding,560,y,662,57,21,{fill});
  }
  s.speakerNotes.text=group.papers.map(p=>p.authors+' ('+p.year+'). '+p.title+'. '+p.venue+'. '+p.url).join('\n\n');
}
if (rvce) {
  const s=slide('Blinded LLM review of 36 lines','Linguistic quality');
  tx(s,'12 shared prompts · 3 lines per prompt · Scores from 1 to 5',55,150,1170,45,27,{bold:true,color:blue});
  const values=[['Source','Lines','Grammar','Meaning'],['Corpus reference','12','4.92','4.50'],
    ['Generate and filter','12','1.75','1.83'],['Metre guided','12','1.58','1.58']];
  const table=s.tables.add({rows:4,columns:4,left:55,top:224,width:1170,height:245,columnWidths:[510,160,250,250],values});
  values.forEach((row,i)=>row.forEach((_,j)=>{
    const cell=table.getCell(i,j);
    cell.fill=i===0?ink:i%2?white:green;
    cell.text.style={fontSize:24,color:i===0?white:ink,bold:i===0};
  }));
  tx(s,'Both generated groups scored low, even after passing the cadence check.',55,507,1170,55,28,{bold:true,color:blue});
  tx(s,'Separate LLM scansion spot-check: 8/8 agreed with chandas, including 3 guided lines.',55,566,1170,41,23,{color:muted});
  tx(s,'SLP1 phoneme cross-check: 36/36 agreed with chandas. No human ratings collected.',55,611,1170,30,19,{color:muted});
  s.speakerNotes.text='Source: results/guided_reader_review/llm_pilot_ratings.json. The model judged grammar and meaning without per-item labels. The sample includes only paired successes. The observed score difference does not establish that guidance worsens quality.';
} else {
  const s=slide('Evaluation of original metrical verse','Next experiment');
  box(s,'Same prompts and named meters',55,166,355,83,warm);
  arrow(s,'→',414,181,65);
  box(s,'Four decoding conditions',480,166,340,83);
  arrow(s,'→',825,181,60);
  box(s,'Independent evaluation',888,166,333,83,warm);
  tx(s,'Hold out complete source works.',65,290,335,75,25);
  tx(s,'Nucleus, contrastive search, rejection sampling and incremental meter control.',490,290,318,132,25);
  tx(s,'Expert scansion, grammaticality, fidelity to the prompt and poetry quality.',898,290,306,132,25);
  tx(s,'Report completion failures and corpus overlap alongside meter adherence.',55,500,1150,88,32,{bold:true,color:blue});
  s.speakerNotes.text='Planned design. Chandomitra and Pingala are direct precedents for Sanskrit generation and evaluation. See the primary-source links on slide 5.';
}
if(deck.slides.items.length!==(rvce?9:8)) throw new Error('Unexpected slide count');
const ppt=await PresentationFile.exportPptx(deck);
await ppt.save(path.join(dir,rvce?'Pingala_Metre_Research.pptx':'pingala_paper_overview_revised.pptx'));
const preview=path.join(dir,rvce?'.rvce-preview':'.review-preview');
await fs.mkdir(preview,{recursive:true});
for(const [i,s] of deck.slides.items.entries()){
  const img=await deck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(preview,'slide-'+(i+1)+'.png'),new Uint8Array(await img.arrayBuffer()));
}
