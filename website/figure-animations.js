(() => {
'use strict';
const NS='http://www.w3.org/2000/svg', accent='#9a7424', blue='#628faa', gray='#707070';
const clamp=v=>Math.max(0,Math.min(1,v)), ease=v=>{v=clamp(v);return v*v*(3-2*v)};
const mix=(a,b,t)=>a+(b-a)*t;
const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
function el(tag,attrs={},parent){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(parent)parent.append(n);return n;}
function label(parent,x,y,text,attrs={}){const n=el('text',{x,y,...attrs},parent);n.textContent=text;return n;}
function line(parent,x1,y1,x2,y2,attrs={}){return el('line',{x1,y1,x2,y2,stroke:'#b8bdc4',...attrs},parent)}
function path(points){return points.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}
function star(parent,x,y,color=accent){let points=[];for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,r=i%2?4.5:10;points.push([x+r*Math.cos(a),y+r*Math.sin(a)])}return el('path',{d:path(points)+'Z',fill:color,stroke:'white','stroke-width':1},parent)}
function mathImage(parent,x,y,width,height,name){return el('image',{x,y,width,height,href:`assets/formula-${name}.svg?v=golden-bronze`,preserveAspectRatio:'xMinYMid meet'},parent)}
function visibility(node,value){node.style.opacity=clamp(value)}
function traced(node,amount){node.setAttribute('pathLength','1');node.setAttribute('stroke-dasharray','1');node.setAttribute('stroke-dashoffset',1-clamp(amount))}
function player(root,steps,durations,draw){
 root.classList.add('animation-ready');
 const controls=root.querySelector('.animation-controls'), toggle=root.querySelector('[data-play]'), replay=root.querySelector('[data-replay]'), title=root.querySelector('.animation-title'), description=root.querySelector('.animation-description'), bar=root.querySelector('.animation-progress span');
 const nav=root.querySelector('.animation-stages');
 let time=0, playing=!reduced.matches, visible=false,last=0,frame=0,current=-1;
 const total=durations.reduce((a,b)=>a+b,0),starts=durations.map((_,i)=>durations.slice(0,i).reduce((a,b)=>a+b,0));
 const buttons=steps.map((step,i)=>{const b=document.createElement('button');b.type='button';b.textContent=`${i+1}. ${step.name}`;if(step.name==='GSBR'){b.textContent=`${i+1}. `;const name=document.createElement('span');name.className='method-name';name.textContent=step.name;b.append(name);}b.addEventListener('click',()=>{playing=false;time=starts[i]+durations[i]-.001;render();sync();});nav.append(b);return b});
 function sync(){toggle.textContent=playing?'Pause':'Play';toggle.setAttribute('aria-label',`${playing?'Pause':'Play'} Figure ${root.dataset.figure} animation`);if(playing&&visible&&!document.hidden&&!frame){frame=requestAnimationFrame(tick)}else if((!playing||!visible||document.hidden)&&frame){cancelAnimationFrame(frame);frame=0;last=0}}
 function render(){let stage=durations.length-1;for(let i=0;i<durations.length;i++)if(time<starts[i]+durations[i]){stage=i;break}const progress=clamp((time-starts[stage])/durations[stage]);if(stage!==current){current=stage;if(title)title.textContent=steps[stage].title;if(description)description.textContent=steps[stage].description;buttons.forEach((b,i)=>{b.classList.toggle('is-current',i===stage);b.setAttribute('aria-pressed',String(i===stage));});}draw(stage,progress,playing);bar.style.width=`${time/total*100}%`;root.dataset.stage=stage;root.dataset.playing=String(playing)}
 function tick(now){frame=0;if(!visible||document.hidden){last=0;return}if(last)time=Math.min(total,time+(now-last)/1000);last=now;if(time>=total)playing=false;render();sync()}
 toggle.addEventListener('click',()=>{if(time>=total)time=0;playing=!playing;last=0;render();sync()});
 replay.addEventListener('click',()=>{time=0;playing=true;last=0;render();sync()});
 window.GSBRMotion.observe(root.querySelector('.animation-viewport'),value=>{visible=value;sync()});
 document.addEventListener('visibilitychange',sync);
 reduced.addEventListener('change',()=>{if(reduced.matches){time=total;playing=false;render();sync()}});
 if(reduced.matches)time=total;controls.hidden=false;render();sync();
}
function figure1(root,data){
 const svg=root.querySelector('svg'),d=data.figure1,c=d.ideal_comparison;
 svg.setAttribute('viewBox','0 0 1080 440');
 label(svg,540,24,'Base policy distribution',{'text-anchor':'middle','font-size':20,fill:gray});
 const density=a=>d.mixture_weights.reduce((v,w,i)=>v+w*Math.exp(-.5*((a-d.mixture_means[i])/d.mixture_standard_deviations[i])**2)/(d.mixture_standard_deviations[i]*Math.sqrt(2*Math.PI)),0);
 const distance=a=>Math.min(...c.support.map(([lo,hi])=>Math.max(lo-a,a-hi,0)));
 const loss=[a=>c.coefficient*(a-d.mixture_mean)**2,a=>distance(a)===0?0:Infinity,a=>c.coefficient*distance(a)**2];
 const names=['Behavior cloning','Support constraint','GSBR'];
 const outcomes=['Pulled toward the mean','Stops at the boundary','Refines beyond support'];
 const panels=[];
 for(let k=0;k<3;k++){
  const left=40+350*k,width=300,X=a=>left+(a+3.55)/8.1*width,baseY=208,lossY=378,color=k===2?accent:blue;
  const panel=el('g',{'data-comparison':k,transform:'translate(0 -45)'},svg);
  const heading=label(panel,left+width/2,440,names[k],{'class':'comparison-name','text-anchor':'middle','font-size':24,fill:color});
  const support=el('g',{},panel);
  c.support.forEach(([lo,hi])=>el('rect',{x:X(lo),y:100,width:X(hi)-X(lo),height:108,fill:color,opacity:.095},support));
  const prior=el('g',{},panel),points=Array.from({length:501},(_,i)=>{const a=-3.55+i/500*8.1;return[X(a),baseY-density(a)*120]});
  el('path',{d:`M${left},${baseY} ${path(points).replace(/^M/,'L')} L${left+width},${baseY}Z`,fill:gray,opacity:.1},prior);
  const curve=el('path',{d:path(points),fill:'none',stroke:gray,'stroke-width':2},prior);
  line(panel,left,baseY,left+width,baseY);
  label(panel,left+width,baseY+23,'Action',{'text-anchor':'end','font-size':15,fill:gray});
  if(k===0)label(panel,left-12,148,'Likelihood',{'text-anchor':'middle',transform:`rotate(-90 ${left-12} 148)`,'font-size':18,fill:gray});
  const epsilon=el('g',{'data-epsilon':k},panel);line(epsilon,left,baseY-d.epsilon*120,left+width,baseY-d.epsilon*120,{stroke:color,'stroke-dasharray':'4 4','stroke-width':1.3});label(epsilon,left-16,baseY-d.epsilon*120+5,'ε',{fill:color,'font-size':18});
  const optimum=el('g',{},panel),sx=X(d.q_peak_action),optimumColor=k===2?accent:gray;line(optimum,sx,168,sx,208,{stroke:optimumColor});star(optimum,sx,168,optimumColor);const optimumLabel=label(optimum,sx,146,'𝑎',{'text-anchor':'middle','font-size':20,fill:optimumColor});el('tspan',{'baseline-shift':'super','font-size':13,'font-style':'normal'},optimumLabel).textContent='*';
  const penalty=el('g',{},panel);
  const defs=el('defs',{},penalty),clip=el('clipPath',{id:`comparison-clip-${k}`},defs);el('rect',{x:left-1,y:259,width:width+2,height:121},clip);
  const clipped=el('g',{'clip-path':`url(#comparison-clip-${k})`},penalty);
  if(k>0)c.support.forEach(([lo,hi])=>el('rect',{x:X(lo),y:264,width:X(hi)-X(lo),height:114,fill:color,opacity:.11},clipped));
  let lossCurve;
  if(k===1){
   c.support.forEach(([lo,hi])=>{line(clipped,X(lo),lossY,X(hi),lossY,{stroke:blue,'stroke-width':2.6});for(const a of [lo,hi])line(clipped,X(a),lossY,X(a),269,{stroke:blue,'stroke-dasharray':'4 4','stroke-width':1.6})});
   const ranges=[[-3.55,c.support[0][0]],[c.support[0][1],c.support[1][0]],[c.support[1][1],4.55]];
   ranges.forEach(([lo,hi])=>line(clipped,X(lo),270,X(hi),270,{stroke:blue,'stroke-width':2}));label(penalty,left+width-4,294,'∞',{fill:blue,'font-size':25,'text-anchor':'end','data-infinity-label':''});
  }else lossCurve=el('path',{d:path(Array.from({length:601},(_,i)=>{const a=-3.55+i/600*8.1;return[X(a),lossY-loss[k](a)*48]})),stroke:color,'stroke-width':2.7,fill:'none'},clipped);
  if(k===0)label(penalty,left-12,320,'Regularizer',{'text-anchor':'middle',transform:`rotate(-90 ${left-12} 320)`,'font-size':18,fill:gray});
  line(penalty,left,lossY,left+width,lossY);label(penalty,left+width,lossY+16,'Action',{'text-anchor':'end','font-size':15,fill:gray});
  const action=el('g',{},panel),guide=line(action,0,0,0,0,{stroke:color,'stroke-dasharray':'4 4'}),dots=[0,1].map(()=>el('circle',{r:5.5,fill:color,stroke:'white','stroke-width':1.5},action));
  const actionLabel=label(action,0,lossY+22,'𝑎',{'text-anchor':'middle','font-size':20,fill:color});
  const outcome=label(panel,left+width/2,470,outcomes[k],{'class':'comparison-subtitle','text-anchor':'middle',fill:gray,'font-size':20});
  panels.push({k,X,heading,prior,curve,support,epsilon,optimum,penalty,lossCurve,action,actionLabel,guide,dots,outcome,baseY,lossY});
 }
 const steps=[
 {name:'Base policy',title:'Start with the base-policy distribution.',description:''},
 {name:'Optimum',title:'The optimal action lies outside ε-support.',description:''},
 {name:'BC',title:'Behavior cloning pulls toward the mean.',description:''},
 {name:'Support',title:'A hard constraint stops improvement at the boundary.',description:''},
 {name:'GSBR',title:'GSBR selects good behaviors and refines beyond the support.',description:''}
 ];
 player(root,steps,[1.6,1.5,4.2,4.2,4.5],(stage,p)=>{
  panels.forEach(v=>{
   const revealed=stage>=v.k+2,local=stage===v.k+2?p:stage>v.k+2?1:0;
   const elapsed=local*[4.2,4.2,4.5][v.k];
   visibility(v.prior,stage===0?ease(p*1.6/.56):1);traced(v.curve,stage===0?p*1.6/.7:1);
   visibility(v.epsilon,v.k>0&&revealed?ease(elapsed):0);
   visibility(v.support,v.k>0&&revealed?ease(elapsed):0);
   visibility(v.optimum,stage>=1?(stage===1?ease(p*1.5/.8):1):0);
   visibility(v.heading,revealed?ease(elapsed/.6):0);
   visibility(v.penalty,revealed?ease(elapsed/1.1):0);
   visibility(v.action,revealed?ease((elapsed-.9)/.7):0);visibility(v.outcome,revealed?ease((elapsed-2.8)/.5):0);
   if(v.lossCurve)traced(v.lossCurve,elapsed/1.9);
   const key=['bc','hard','gsbr'][v.k],target=c.display_actions[key],start=v.k===0?d.mixture_mean:d.mixture_means[1],a=mix(start,target,ease((elapsed-.9)/1.8));
   const x=v.X(a),y1=v.baseY-density(a)*120,y2=v.lossY-(v.k===1?0:loss[v.k](a))*48;
   v.actionLabel.setAttribute('x',x);v.actionLabel.setAttribute('y',Math.max(y2+35,v.lossY+34));
   v.guide.setAttribute('x1',x);v.guide.setAttribute('x2',x);v.guide.setAttribute('y1',y1);v.guide.setAttribute('y2',y2);
   v.dots.forEach((dot,i)=>{dot.setAttribute('cx',x);dot.setAttribute('cy',i?y2:y1)});
  });
 });
}
function figure2(root,data){
 const scene=root.querySelector('svg'),d=data.figure2,scale=88,count=d.samples.length;
 scene.setAttribute('viewBox','0 0 760 450');
 const diagramOffset=76,svg=el('g',{'data-support-diagram':'',transform:`translate(${diagramOffset} 0)`},scene);
 const isolatedIndex=d.distances.indexOf(Math.max(...d.distances));
 const X=x=>50+(x+2.65)*scale,Y=y=>400-(y+1.7)*scale;
 const density=el('g',{},svg);
 d.contours.forEach(c=>el('path',{d:path(c.points.map(([x,y])=>[X(x),Y(y)]))+'Z',fill:blue,'fill-opacity':c.level===.15?.07:c.level===.4?.12:.18,stroke:blue,'stroke-opacity':.4,'stroke-width':1},density));
 const priorTitle=mathImage(svg,50,9,260,29,'prior2');
 const unionTitle=el('g',{'data-support-legend':'','aria-label':'Estimated support region C'},svg);
 el('rect',{x:129,y:369,width:16,height:16,fill:accent,'fill-opacity':.18,stroke:accent,'stroke-width':1.8},unionTitle);
 mathImage(unionTitle,156,359,285,31,'support-region');
 line(svg,50,405,516,405);mathImage(svg,500,418,19,20,'a1');line(svg,50,405,50,62);mathImage(svg,26,63,19,20,'a2');
 const defs=el('defs',{},svg),unionClip=el('clipPath',{id:'support-growth'},defs);
 const growthNodes=d.samples.map(([x,y])=>el('circle',{cx:X(x),cy:Y(y),r:0},unionClip));
 const union=el('g',{'clip-path':'url(#support-growth)'},svg);
 const unionPath=d.support_union.map(points=>path(points.map(([x,y])=>[X(x),Y(y)]))+'Z').join(' ');
 el('path',{d:unionPath,fill:accent,'fill-opacity':.18,stroke:accent,'stroke-width':1.8,'fill-rule':'evenodd'},union);
 const contourOverlay=el('g',{'data-support-contours':''},svg);
 d.contours.forEach(c=>el('path',{d:path(c.points.map(([x,y])=>[X(x),Y(y)]))+'Z',fill:'none',stroke:blue,'stroke-opacity':.65,'stroke-width':1.2},contourOverlay));
 const edges=el('g',{},svg),edgeNodes=d.samples.map((b,i)=>{const n=d.samples[d.nearest[i]];return line(edges,X(b[0]),Y(b[1]),X(n[0]),Y(n[1]),{stroke:i===isolatedIndex?blue:'#9ba3ab','stroke-width':1.7,'stroke-dasharray':i===isolatedIndex?'5 4':'none'})});
 const constructionBalls=el('g',{'data-construction-balls':''},svg);
 const ballNodes=d.samples.map(([x,y])=>el('circle',{cx:X(x),cy:Y(y),r:0,fill:'none',stroke:accent,'stroke-width':1.2},constructionBalls));
 const samples=el('g',{},svg),sampleNodes=d.samples.map(([x,y])=>el('circle',{cx:X(x),cy:Y(y),r:4.5,fill:blue,stroke:'white','stroke-width':1.4},samples));
 const bubble=el('g',{'data-distance-bubble':'','aria-label':'Nearest-neighbor distance thought bubble'},scene);
 const bubbleFill='#fcfdff',bubbleStroke='#c9d7df';
 el('circle',{cx:495,cy:310,r:3.5,fill:bubbleFill,stroke:bubbleStroke,'stroke-width':1.2},bubble);
 el('circle',{cx:507,cy:295,r:7,fill:bubbleFill,stroke:bubbleStroke,'stroke-width':1.2},bubble);
 el('rect',{x:504,y:116,width:246,height:178,rx:22,fill:bubbleFill,stroke:bubbleStroke,'stroke-width':1.2},bubble);
 const chart=el('g',{},bubble),cx=524,cy=262,cw=206,ch=112;
 const order=d.distances.map((u,i)=>({u,i})).sort((a,b)=>a.u-b.u),rank=Object.fromEntries(order.map((o,i)=>[o.i,i]));
 const DX=i=>cx+rank[i]/(count-1)*cw,DY=u=>cy-Math.log(u/.1)/Math.log(20)*ch;
 label(chart,cx,138,'Nearest-neighbor distance',{'font-size':13,fill:gray});
 [.1,.2,.5,1,2].forEach(u=>line(chart,cx,DY(u),cx+cw,DY(u),{stroke:'#e8edf1'}));
 line(chart,cx,cy,cx+cw,cy);label(chart,cx+cw/2,282,'Sample rank',{'text-anchor':'middle','font-size':12,fill:gray});
 const quantile=el('g',{},bubble),qy=DY(d.radius);
 line(quantile,cx,qy,cx+cw,qy,{stroke:accent,'stroke-width':1.5,'stroke-dasharray':'4 3'});
 el('circle',{cx:cx+d.quantile*cw,cy:qy,r:5.5,fill:bubbleFill,stroke:accent,'stroke-width':1.8},quantile);
 label(quantile,cx+cw,qy-12,'q = 0.75',{'text-anchor':'end',fill:accent,'font-size':12});
 // Copies travel in scene coordinates from each action to its ranked distance.
 const distances=el('g',{'data-ranked-samples':''},scene),distanceNodes=d.samples.map(()=>el('circle',{r:3.8,fill:blue,stroke:'white','stroke-width':1},distances));
 const example=d.penalty_example,ax=X(example.action[0]),ay=Y(example.action[1]),bx=X(example.closest_point[0]),by=Y(example.closest_point[1]);
 const radiusSample=d.samples.reduce((rightmost,sample)=>sample[0]>rightmost[0]?sample:rightmost),rx=X(radiusSample[0]),ry=Y(radiusSample[1]);
 const radiusLabelX=rx+d.radius*scale+9,radiusLabelY=ry-16;
 const radiusGroup=el('g',{'data-radius-guide':''},svg);
 line(radiusGroup,rx,ry,rx+d.radius*scale,ry,{stroke:accent,'stroke-width':2});
 const optimum=el('g',{'data-optimal-action':''},svg);star(optimum,ax,ay);
 const actionLabel=el('foreignObject',{x:ax+12,y:ay-16,width:32,height:36,'class':'method-math variable-label','aria-label':'Actor action a'},optimum);
 actionLabel.innerHTML='<div xmlns="http://www.w3.org/1999/xhtml" style="color:#9a7424;font-size:18px"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi mathvariant="bold-italic">a</mi></math></div>';
 const gap=el('g',{'data-support-distance':''},svg),gapLine=line(gap,bx,by,ax,ay,{stroke:accent,'stroke-width':2.5});
 const gapEndpoint=el('circle',{cx:bx,cy:by,r:4.5,fill:'white',stroke:accent,'stroke-width':2},gap);
 // A parallel ruler marks the measured segment; the loss is its squared length.
 const gapLength=Math.hypot(ax-bx,ay-by),nx=-(ay-by)/gapLength,ny=(ax-bx)/gapLength;
 const ruler=el('g',{'data-loss-ruler':''},gap);
 line(ruler,bx+nx*10,by+ny*10,ax+nx*10,ay+ny*10,{stroke:accent,'stroke-width':1});
 [[bx,by],[ax,ay]].forEach(([x,y])=>line(ruler,x+nx*6,y+ny*6,x+nx*14,y+ny*14,{stroke:accent,'stroke-width':1}));

 // Short, anchored annotations replace the separate prose panel.
 function callout(x,y,title,lines,target){
  const group=el('g',{'class':'method-callout'},scene);
  if(target){line(group,x+38,y+25+lines.length*19,target[0],target[1],{stroke:'#b6a67f','stroke-width':1.2});el('circle',{cx:target[0],cy:target[1],r:3,fill:accent},group);}
  el('rect',{x:x-10,y:y-19,width:330,height:30+lines.length*19,rx:7,fill:'white','fill-opacity':.96},group);
  label(group,x,y,title,{'class':'method-callout-title','font-size':17,fill:accent});
  lines.forEach((text,i)=>label(group,x,y+24+i*19,text,{'font-size':14,fill:'#363633'}));
  return group;
 }
 const sampleCallout=callout(350,140,'Sample actions',['Draw actions from the base policy.']);
 const neighborCallout=callout(350,125,'Measure local spacing',['Nearest-neighbor distance','for every sample.']);
 const quantileCallout=callout(382,55,'Choose a distance quantile',['so isolated samples do not inflate r.']);
 const constructCallout=callout(378,125,'Construct the support region C',['Union of balls of radius r','around each sampled action.']);
 const supportCallout=callout(365,118,'Penalize distance from support region',['Zero penalty inside C.','Squared L₂ penalty outside.']);
 supportCallout.querySelector('.method-callout-title').setAttribute('font-size',16);
 const lossEquation=el('foreignObject',{x:diagramOffset+ax+18,y:ay+18,width:240,height:57,'class':'method-math','data-gsbr-loss':'','aria-label':'GSBR loss equals the squared positive part of the minimum L2 distance to a reference action minus radius r'},scene);
 lossEquation.innerHTML='<div xmlns="http://www.w3.org/1999/xhtml" style="color:#9a7424;font-size:18px"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>ℓ</mi><mtext>GSBR</mtext></msub><mo>=</mo><msubsup><mrow><mo>[</mo><munder><mo>min</mo><mi>i</mi></munder><msub><mrow><mo>‖</mo><mi>a</mi><mo>−</mo><msub><mi>b</mi><mi>i</mi></msub><mo>‖</mo></mrow><mn>2</mn></msub><mo>−</mo><mi>r</mi><mo>]</mo></mrow><mo>+</mo><mn>2</mn></msubsup></mrow></math></div>';
 // One MathML r travels from the chart and remains as the final radius label.
 const radiusTransfer=el('foreignObject',{x:cx-16,y:qy-16,width:32,height:36,'class':'method-math variable-label','data-radius-transfer':'','aria-label':'Radius r'},scene);
 radiusTransfer.innerHTML='<div xmlns="http://www.w3.org/1999/xhtml" style="color:#9a7424;font-size:18px"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi mathvariant="bold-italic">r</mi></math></div>';
 const steps=[
 {name:'Sample',title:'Sample actions',description:'Draw actions from the base policy.'},
 {name:'Neighbors',title:'Measure local spacing',description:'Measure the nearest-neighbor distance for every sample.'},
 {name:'Quantile',title:'Choose a distance quantile',description:'so isolated samples do not inflate r.'},
 {name:'Support',title:'Construct the support region C',description:'Union of balls of radius r around each sampled action.'},
 {name:'Penalty',title:'Penalize distance from support region',description:'No behavior penalty inside C; a squared-distance penalty outside.'}
 ];
 const methodDurations=[3.1,3,5.2,5.5,5];
 player(root,steps,methodDurations,(stage,p,playing)=>{
  const elapsed=p*methodDurations[stage],remaining=(1-p)*methodDurations[stage];
  // Keep descriptions readable when paused or selecting a stage manually.
  const descriptionOpacity=playing?ease(elapsed/.65)*(stage<4?ease(remaining/.45):1):1;
  [sampleCallout,neighborCallout,quantileCallout,constructCallout,supportCallout].forEach((node,i)=>visibility(node,i===stage?descriptionOpacity:0));
  // Trim the first two holds while preserving point and line reveal speeds.
  if(stage===0)p*=3.1/4.5;
  if(stage===1)p*=3/3.5;
  const distanceProgress=stage===4?clamp((p-.2)/.35):0;
  const measurementReveal=stage===4&&distanceProgress===1?ease((p-.58)/.16):0;
  visibility(lossEquation,measurementReveal);visibility(ruler,measurementReveal);visibility(gapEndpoint,measurementReveal);
  visibility(density,1);
  visibility(contourOverlay,stage>=3?1:0);
  visibility(optimum,stage===4?ease(p/.3):0);
  visibility(gap,stage===4?ease((p-.2)/.35):0);traced(gapLine,distanceProgress);
  sampleNodes.forEach(node=>{const t=stage>0?1:ease((p-.08)/.26);visibility(node,stage===2?1-.72*ease((p-.15)/.3):t);node.setAttribute('r',4.5*t)});
  edgeNodes.forEach((node,i)=>{const t=stage===1?ease(p/.3):stage===2?1-ease((p-.1)/.4):0;visibility(node,t);const b=d.samples[i],n=d.samples[d.nearest[i]];node.setAttribute('x2',mix(X(b[0]),X(n[0]),t));node.setAttribute('y2',mix(Y(b[1]),Y(n[1]),t));});
  const bubbleAmount=stage===2?ease(p/.15):stage===3?1-ease(p/.14):0;
  visibility(bubble,bubbleAmount);
  distanceNodes.forEach((node,i)=>{
   const t=stage>2?1:stage===2?ease((p-.12-rank[i]/count*.06)/.28):0;
   visibility(node,stage===2?ease(p/.14):stage===3?1-ease(p/.14):0);
   node.setAttribute('cx',mix(diagramOffset+X(d.samples[i][0]),DX(i),t));
   node.setAttribute('cy',mix(Y(d.samples[i][1]),DY(d.distances[i]),t)-35*Math.sin(Math.PI*t));
  });
  visibility(quantile,stage>2?1:stage===2?ease((p-.55)/.18):0);
  const transfer=stage>3?1:stage===3?ease(p/.35):0;
  visibility(radiusTransfer,stage>=3?1:stage===2?ease((p-.55)/.18):0);
  radiusTransfer.setAttribute('x',mix(cx-16,diagramOffset+radiusLabelX,transfer));
  radiusTransfer.setAttribute('y',mix(qy-16,radiusLabelY,transfer)-45*Math.sin(Math.PI*transfer));
  const growth=stage>3?1:stage===3?ease((p-.35)/.45):0;
  growthNodes.forEach(node=>node.setAttribute('r',d.radius*scale*growth));
  ballNodes.forEach(node=>node.setAttribute('r',d.radius*scale*growth));
  visibility(constructionBalls,stage===3?1-ease((p-.78)/.22):0);
  visibility(radiusGroup,stage>3?1:stage===3?ease((p-.33)/.1):0);
  visibility(priorTitle,stage<3?1:0);visibility(unionTitle,growth);

 });
}
fetch('assets/figure-animation-data.json?v=support-penalty').then(r=>{if(!r.ok)throw Error('Figure data unavailable');return r.json()}).then(data=>{
 figure1(document.querySelector('[data-figure="1"]'),data);
 figure2(document.querySelector('[data-figure="2"]'),data);
}).catch(error=>{console.error(error);document.querySelectorAll('.animated-figure').forEach(root=>{root.classList.remove('animation-ready');root.querySelector('.animation-controls').hidden=true;});});
})();
