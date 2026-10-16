const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const src = f => path.join(__dirname, '..', 'src', 's1ced', f);
const fixture = f => path.join(__dirname, 'fixtures', f);

// Execute the shipped script. Only the browser DOM and HTTP transport are replaced.
function page(fixtures = null) {
  const elements = new Map();
  const el = s => {
    if (!elements.has(s)) elements.set(s, {innerHTML: s === '#out' ? 'EMPTY' : '', textContent: '', value: '', checked: false, hidden: true, disabled: false,
      style: {}, dataset: {}, classList: {add(){}, remove(){}, toggle(){}}, addEventListener(){}, focus(){}, scrollIntoView(){}, select(){},
      showModal(){this.open=true;}, close(){this.open=false;}, setCustomValidity(){},
      querySelector(){return null;}, insertBefore(child){child.parentElement=this;}});
    return elements.get(s);
  };
  const context = vm.createContext({console, document: {querySelector: el, querySelectorAll: () => [], addEventListener(){}},
    localStorage: {getItem: () => '1', setItem(){}}, fetch: async (path, options) => ({ok:true, json:async()=> {
      if (path === '/api/examples') return fixtures ? JSON.parse(fs.readFileSync(src('examples.json'),'utf8')) : {groups:[]};
      if (path === '/api/decide' && fixtures) return fixtures.find(x=>x.request.state === JSON.parse(options.body).request.state).out;
      return {local:{model:'test',device:'cpu',max_state_tokens:510,max_options:8,threshold:null},remote:{model:'test-remote'},key:{set:false}};
    }})});
  const html = fs.readFileSync(src('playground.html'),'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
  return {el, context, run: s => vm.runInContext(s,context)};
}

test('late response cannot repopulate results after scenario switch', async () => {
  const p=page(); await p.run('examplesReady');
  p.run("api = () => new Promise(resolve => globalThis.finish = resolve)");
  const pending=p.run('run()'); p.run('loadPreset(1)');
  p.run("finish({local:{answers:{safe_to_run:{type:'noul',noul:0.9}}}})"); await pending;
  assert.equal(p.el('#out').innerHTML,'EMPTY');
});
test('keyboard/programmatic repeats cannot launch overlapping decisions',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run('globalThis.calls=0; api=()=>{calls++; return new Promise(resolve=>globalThis.finish=resolve)}');
  const a=p.run('run()'); const b=p.run('run()');
  assert.equal(p.run('calls'),1);
  p.run('finish({local:{answers:{}}})'); await Promise.all([a,b]);
});
test('score verdict selects the same discrete level as the probability bars',()=>{
  const p=page(); assert.equal(p.run("headline({score:1.4,probabilities:{0:0.4,1:0.1,2:0.2,3:0.3}},'score')"),0);
});
test('missing boolean answers cannot count as no',()=>{
  const p=page(); assert.equal(p.run("headline({},'noul')"),null);
});
test('confidence in a low qualification level cannot recommend booking a sales call',()=>{
  const p=page(); p.run('loadPreset(2); lastGates=clone(gates)');
  const label=p.run("gateTag('lead_quality',{score:0,confidence:0.99},'score')");
  assert.ok(!label.includes('Book an AE'));
});
test('JSON questions without type default to choice and malformed types are rejected',()=>{
  const p=page(); p.run("mode='json'");
  p.el('#json').value=JSON.stringify({questions:{team:{criteria:{billing:null,tech:null}}}});
  assert.equal(p.run('currentRequest().questions.team.type'),'choice');
  p.el('#json').value=JSON.stringify({questions:{team:{type:'Choice',criteria:{billing:null,tech:null}}}});
  assert.throws(()=>p.run('currentRequest()'),/type/i);
});
test('blank gate threshold blocks a decision',()=>{
  const p=page(); p.el('#gth').value='';
  p.run('readEditor()'); assert.throws(()=>p.run('currentRequest()'),/threshold/i);
});
test('remote usage strings render as text',()=>{
  const p=page(); const result=p.run("legendRow('remote','Remote',{usage:{input_tokens:'<img src=x onerror=alert(1)>'}})");
  assert.ok(!result.includes('<img')); assert.ok(result.includes('&lt;img'));
});
test('scenario load clears errors in both editor modes',()=>{
  const p=page(); p.el('#inErr2').textContent='old error'; p.run('loadPreset(1)');
  assert.equal(p.el('#inErr2').textContent,'');
});
test('tutorial requires both resume examples before it opens',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run("PRESETS.push({name:'Backend engineer, fintech',req:{state:'test',questions:{q:{type:'noul'}}}})");
  await p.run('startTour()');
  assert.equal(p.el('#tour').hidden,true);
  assert.match(p.el('#examplesErr').textContent,/resume/i);
});
test('tutorial revisiting a step cannot let the old step take focus',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run("globalThis.finished=[]; TOUR[0].go=()=>new Promise(resolve=>finished.push(resolve)); $('#tour').hidden=false");
  const old=p.run('showStep(0)'); const recent=p.run('showStep(0)');
  p.run('finished[0]()'); await old;
  assert.equal(p.el('#tourNext').disabled,true);
  p.run('finished[1]()'); await recent;
  assert.equal(p.el('#tourNext').disabled,false);
});
test('model label and tutorial limits come from health',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run("api=async()=>({local:{model:'alternate',device:'cuda',max_state_tokens:1022,max_options:6,threshold:null},remote:{model:'other'},key:{set:false}})");
  await p.run('refreshHealth()');
  assert.match(p.el('#target [data-t=local] small').textContent,/alternate.*cuda/);
  p.run("$('#tour').hidden=false"); await p.run('showStep(1)');
  assert.match(p.el('#tourText').innerHTML,/1022/);
});
test('gate keys inherited from Object.prototype are not treated as configured gates',()=>{
  const p=page(); p.run("mode='json'");
  p.el('#json').value='{"questions":{"toString":{"type":"noul"}}}';
  assert.equal(p.run('currentRequest().questions.toString.type'),'noul');
});
test('changing target discards a pending response',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run('api=()=>new Promise(resolve=>globalThis.finish=resolve)');
  const pending=p.run('run()'); p.run("setTarget('remote')");
  p.run("finish({local:{answers:{safe_to_run:{noul:0.8}}}})"); await pending;
  assert.equal(p.el('#out').innerHTML,'EMPTY');
});
test('adding a question invalidates pending results',async()=>{
  const p=page(); await p.run('examplesReady');
  p.run('api=()=>new Promise(resolve=>globalThis.finish=resolve)'); const pending=p.run('run()');
  p.el('#qtabs').onclick({target:{closest:()=>({id:'addQ'})}});
  p.run("finish({local:{answers:{safe_to_run:{noul:0.8}}}})"); await pending;
  assert.equal(p.el('#out').innerHTML,'EMPTY');
});
test('all shipped examples render real responses with consistent header scores',async()=>{
  const fixtures=JSON.parse(fs.readFileSync(fixture('examples-results.json'),'utf8'));
  const p=page(fixtures); await p.run('examplesReady'); let matches=0,questions=0;
  for (const item of fixtures) {
    p.run(`loadPreset(presetIdx(${JSON.stringify(item.name)}))`); await p.run('run()');
    const html=p.el('#out').innerHTML;
    const count=html.match(/class="score">(\d+)\/(\d+) expected/);
    assert.ok(count,item.name); matches+=Number(count[1]); questions+=Number(count[2]);
    assert.equal(p.el('#inErr').textContent,'',item.name);
  }
  assert.equal(matches,16); assert.equal(questions,38);
});
test('all 12 tutorial steps execute and closing a pending step stays closed',async()=>{
  const fixtures=JSON.parse(fs.readFileSync(fixture('examples-results.json'),'utf8'));
  const p=page(fixtures); await p.run('startTour()');
  for(let i=0;i<12;i++) {
    await p.run(`showStep(${i})`);
    assert.equal(p.el('#tourN').textContent,`${i+1} / 12`);
    assert.equal(p.el('#tourNext').disabled,false);
    assert.ok(p.el('#tourTitle').textContent);
  }
  p.run('api=()=>new Promise(resolve=>globalThis.finish=resolve)'); const pending=p.run('showStep(6)');
  p.run('endTour()'); p.run('finish({local:{answers:{}}})'); await pending;
  assert.equal(p.el('#tour').hidden,true); assert.equal(p.run('tourStep'),-1);
});
test('compare all renders three models, flags disagreement and the shared review signal',()=>{
  const p=page(); p.run('loadPreset(0); lastGates=clone(gates); lastReq=clone(doc); lastExpected={}; lastTarget="all"');
  p.run(`lastOut={local:{answers:{safe_to_run:{type:'noul',noul:0.9}}},
    vega:{answers:{safe_to_run:{type:'noul',noul:0.2,uncertainty:{abstain:true,out_of_scope:false,prediction_set:['no'],needs_review:true}}}},
    remote:{answers:{safe_to_run:{type:'noul',noul:0.95}}}}; renderOut()`);
  const h=p.el('#out').innerHTML;
  assert.ok(h.includes('track vega')); assert.ok(h.includes('track remote'));
  assert.ok(h.includes('disagree')); assert.ok(h.includes('needs review: abstains'));
});
test('compare all without a key notes the skipped remote model',()=>{
  const p=page(); p.run('loadPreset(0); lastGates=clone(gates); lastReq=clone(doc); lastExpected={}; lastTarget="all"');
  p.run(`lastOut={local:{answers:{safe_to_run:{type:'noul',noul:0.9}}},vega:{answers:{safe_to_run:{type:'noul',noul:0.8,uncertainty:{abstain:false,out_of_scope:false,prediction_set:['yes'],needs_review:false}}}}}; renderOut()`);
  const h=p.el('#out').innerHTML;
  assert.ok(h.includes('all agree')===false && h.includes('agree')); assert.match(h,/not run: add a TypeSafe key/);
  assert.ok(h.includes('stands behind it'));
});
test('Vega target does not demand an API key',async()=>{
  const p=page(); await p.run('examplesReady'); p.run('globalThis.calls=0; api=async(path)=>{if(path==="/api/decide") calls++; return {vega:{answers:{}}}}; setTarget("vega")');
  await p.run('run()'); assert.equal(p.el('#keyDlg').open,undefined); assert.equal(p.run('calls'),1);
});
