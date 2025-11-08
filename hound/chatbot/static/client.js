let pc;
let dc;
let remoteAudio;
let chatEl;
let connDot, connText;
let micStream;
let audioTransceiver;
let lastAssistantBubble = null;
let lastUserBubble = null;
let assistantTranscript = '';
let userTranscript = '';
let activityES = null;
let pendingEvt = null;
let nowInvTimer = null;
let codeViewEl = null;
let hoverCardEl = null;
let hypoDetailEl = null;
let avatarHoldTimer = null;

// Inuzumi avatar asset rules
const INUZUMI_BASE = '/static/inuzumi';
// Exactly six allowed emotions (match existing asset names)
// Use hyphenated tokens where applicable
const EMOTIONS = new Set(['neutral','happy','shy','intense-focus','sad-slightly','crying']);
function inuzumiCandidates(emotion){
  // Return an ordered list of exact .jpg files to try, matching the six assets
  const e = (emotion||'neutral').toLowerCase();
  // Map legacy/alias inputs into allowed set
  let tok = e;
  if (tok==='concentrated' || tok==='focused' || tok==='focus') tok = 'intense-focus';
  if (tok==='sad' || tok==='sad-subtle' || tok==='sad-weak') tok = 'sad-slightly';
  if (tok==='happy-strong' || tok==='amused') tok = 'happy';
  if (!EMOTIONS.has(tok)) tok = 'neutral';

  const f = [];
  switch (tok){
    case 'happy':
      f.push(`${INUZUMI_BASE}/Inuzumi-happy.jpg`);
      break;
    case 'neutral':
      f.push(`${INUZUMI_BASE}/Inuzumi-neutral.jpg`);
      break;
    case 'shy':
      f.push(`${INUZUMI_BASE}/inuzumi-shy.jpg`);
      break;
    case 'intense-focus':
      f.push(`${INUZUMI_BASE}/inuzumi-intense-focus.jpg`);
      f.push(`${INUZUMI_BASE}/Inuzumi-intense-focus.jpg`);
      break;
    case 'sad-slightly':
      f.push(`${INUZUMI_BASE}/inuzumi-sad-slightly.jpg`);
      f.push(`${INUZUMI_BASE}/Inuzumi-sad-slightly.jpg`);
      break;
    case 'crying':
      f.push(`${INUZUMI_BASE}/inuzumi-crying.jpg`);
      f.push(`${INUZUMI_BASE}/Inuzumi-crying.jpg`);
      break;
  }
  // Final default
  f.push(`${INUZUMI_BASE}/Inuzumi-neutral.jpg`);
  return f;
}

function setConn(s) {
  if (!connDot || !connText) return;
  connDot.classList.toggle('online', s);
  connDot.classList.toggle('offline', !s);
  connText.textContent = s ? 'Connected' : 'Disconnected';
}
function chatAdd(role, text) {
  const d = document.createElement('div');
  d.className = `bubble ${role}`;
  d.textContent = text;
  if (!chatEl) { chatEl = document.getElementById('chat'); }
  const container = chatEl || document.getElementById('chat');
  if (container) {
    container.appendChild(d);
    container.scrollTop = container.scrollHeight;
  }
  if (role === 'assistant') lastAssistantBubble = d;
  return d;
}
function chatAppendDelta(role, delta) {
  if (role === 'assistant' && lastAssistantBubble) {
    lastAssistantBubble.textContent += delta;
    const container = chatEl || document.getElementById('chat');
    if (container) container.scrollTop = container.scrollHeight;
  } else {
    chatAdd(role, delta);
  }
}
function esc(s){ return String(s||'').replace(/[&<>]/g, c=> ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function highlightSolidity(src){
  // Very lightweight highlighter: order matters (comments/strings first)
  let s = esc(src);
  // block comments
  s = s.replace(/\/\*[\s\S]*?\*\//g, m=>`<span class="com">${m}</span>`);
  // line comments
  s = s.replace(/(^|\n)\s*\/\/.*?(?=\n|$)/g, m=>`<span class=\"com\">${m}</span>`);
  // strings
  s = s.replace(/"([^"\\]|\\.)*"|'([^'\\]|\\.)*'/g, m=>`<span class=\"str\">${m}</span>`);
  // numbers
  s = s.replace(/\b\d+\b/g, m=>`<span class=\"num\">${m}</span>`);
  // keywords/types
  const kw = ['contract','function','returns','public','external','internal','private','view','pure','payable','if','else','for','while','require','revert','emit','return','modifier','struct','enum','mapping','event','library','using','new','delete','override','virtual','immutable','constant'];
  const types = ['address','bool','string','bytes','uint','uint256','int','int256','byte','fixed','ufixed'];
  const kwRe = new RegExp('\\b(' + kw.join('|') + ')\\b','g');
  const tyRe = new RegExp('\\b(' + types.join('|') + ')\\b','g');
  s = s.replace(kwRe, m=>`<span class=\"kw\">${m}</span>`);
  s = s.replace(tyRe, m=>`<span class=\"type\">${m}</span>`);
  // simple function names (identifier followed by '(')
  s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()/g, m=>`<span class=\"fn\">${m}</span>`);
  return s;
}
function renderCode(content, relpath){
  const ext = (relpath||'').toLowerCase();
  let html;
  if (ext.endsWith('.sol')) html = highlightSolidity(content||'');
  else html = esc(content||'');
  return `<code>${html}</code>`;
}
function showCode(content, relpath){
  if (!codeViewEl) codeViewEl = document.getElementById('codeView');
  if (!codeViewEl) return;
  codeViewEl.innerHTML = renderCode(content||'', relpath||'');
  try { codeViewEl.scrollTop = 0; } catch(_){ }
}
function setAvatarLegacy(name){
  const emo = String(name||'neutral').toLowerCase();
  let mapped = emo;
  if (mapped==='concentrated' || mapped==='focused' || mapped==='focus') mapped = 'intense-focus';
  if (mapped==='amused') mapped = 'happy';
  setAvatarState({ emotion: mapped });
}
function setAvatarState({ emotion='neutral', intensity='subtle', look='center', holdMs=0 }={}){
  const img = document.getElementById('avatarImg');
  if (!img) return;
  const candidates = inuzumiCandidates(emotion, intensity, look);
  let idx = 0;
  const tryNext = ()=>{
    if (idx >= candidates.length){ try{ img.onerror = null; }catch(_){ } return; }
    const src = candidates[idx++];
    try{ img.onerror = tryNext; img.src = src; }catch(_){ tryNext(); }
  };
  tryNext();
  // No emotion label under avatar
  // Hold timer to revert to neutral-subtle unless refreshed
  if (avatarHoldTimer){ try{ clearTimeout(avatarHoldTimer); }catch(_){ } avatarHoldTimer=null; }
  if (holdMs && holdMs>0){
    avatarHoldTimer = setTimeout(()=>{ setAvatarState({ emotion:'neutral', intensity:'subtle', look:'center' }); }, Math.min(holdMs, 15000));
  }
}
// No status pill under avatar anymore; keep a no-op for callers
function setPill(_text) { /* intentionally empty */ }
function classifyEmotion(text) {
  const t = (text||'').toLowerCase();
  if (/haha|lol|fun|great|nice|cool|awesome|!/.test(t)) return 'happy';
  if (/(hmm+|let me think|consider|thinking|pause|processing|hold on)/.test(t)) return 'intense-focus';
  if (/(focus|focused|investigating|analy[sz]|digging)/.test(t)) return 'intense-focus';
  if (/(oops|sorry|embarrassed|shy)/.test(t)) return 'shy';
  if (/(unfortunately|regret|sad|:\(|not good|bad news|cry)/.test(t)) return 'sad-slightly';
  return 'neutral';
}

async function connect() {
  remoteAudio = new Audio();
  remoteAudio.autoplay = true;
  try { remoteAudio.playbackRate = 1.15; } catch(_) {}
  connDot = document.getElementById('connDot');
  connText = document.getElementById('connText');
  chatEl = document.getElementById('chat');

  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
  pc.ontrack = (e)=>{ remoteAudio.srcObject = e.streams[0]; try{remoteAudio.play()}catch(_){} };
  pc.onconnectionstatechange = ()=>{
    const connected = (pc.connectionState==='connected');
    setConn(connected);
    try{
      const btn = document.getElementById('connectBtn');
      if (btn){
        // Disable while connected or connecting; allow click otherwise
        btn.disabled = connected || (pc.connectionState==='connecting');
      }
    }catch(_){ /* ignore */ }
  };
  pc.oniceconnectionstatechange = ()=>{
    const s = pc.iceConnectionState;
    if (s==='failed' || s==='disconnected' || s==='closed'){
      try{ const btn = document.getElementById('connectBtn'); if (btn) btn.disabled = false; }catch(_){ }
      setConn(false);
    }
  };
  // Pre-create audio transceiver
  try { audioTransceiver = pc.addTransceiver('audio', { direction: 'sendrecv' }); } catch (_) {}

  // Create data channel before initial negotiation so SDP includes it
  dc = pc.createDataChannel('oai-events');
  dc.onopen = ()=>{
    setConn(true);
    try{ const btn = document.getElementById('connectBtn'); if (btn) btn.disabled = true; }catch(_){ }
    sessionConfigure();
  };
  dc.onmessage = (ev)=>{ try{ handleEvent(JSON.parse(ev.data)); }catch(_){} };
  // Single negotiation with both audio + data
  await renegotiate();
}

function sendEvent(evt){ if (dc && dc.readyState==='open'){ dc.send(JSON.stringify(evt)); return true;} return false; }

async function enableMic(){
  if (micStream) return;
  micStream = await navigator.mediaDevices.getUserMedia({audio:true});
  const t = micStream.getAudioTracks()[0];
  if (audioTransceiver && audioTransceiver.sender) { try{ await audioTransceiver.sender.replaceTrack(t);}catch(_){} }
  setPill('Listening'); setAvatarState({ emotion:'focused', intensity:'subtle' });
  // Toggle UI to allow disabling
  try{
    const micBtn = document.getElementById('micBtn');
    if (micBtn){
      micBtn.textContent = 'Disable Mic';
      micBtn.onclick = disableMic;
    }
  }catch(_){ }
}

async function disableMic(){
  try{
    if (micStream){
      for (const tr of micStream.getTracks()){
        try{ tr.stop(); }catch(_){ }
      }
    }
    // Stop sending microphone upstream
    if (audioTransceiver && audioTransceiver.sender){
      try{ await audioTransceiver.sender.replaceTrack(null); }catch(_){ }
    }
  }finally{
    micStream = null;
    setPill('Idle'); setAvatarState({ emotion:'neutral', intensity:'subtle' });
    try{
      const micBtn = document.getElementById('micBtn');
      if (micBtn){
        micBtn.textContent = 'Enable Mic';
        micBtn.onclick = enableMic;
      }
    }catch(_){ }
  }
}

async function sessionConfigure(){
  // Configure tools and instructions (no nested audio config here)
  const tools = [
    {
      type: 'function',
      name: 'set_emotion',
      description: 'Deprecated: use set_avatar_state instead. Sets legacy UI emotion label.',
      parameters: {
        type: 'object',
        properties: {
          value: { type: 'string', enum: ['neutral','concentrated','amused','shy'] }
        },
        required: ['value']
      }
    },
    {
      type: 'function',
      name: 'set_avatar_state',
      description: 'Set Inuzumi avatar state (not spoken).',
      parameters: {
        type: 'object',
        properties: {
          emotion: { type:'string', enum: ['neutral','happy','shy','intense-focus','sad-slightly','crying'] },
          holdMs: { type:'number', minimum: 0 }
        },
        required: ['emotion']
      }
    },
    {
      type: 'function',
      name: 'get_hound_status',
      description: 'Return a short status of Hound audits.',
      parameters: { type: 'object', properties: {}, required: [] }
    },
    { type: 'function', name: 'human_status', description: 'Human-friendly audit status summary (no raw counts).', parameters: { type:'object', properties:{}, required: [] } },
    {
      type: 'function',
      name: 'enqueue_steering',
      description: 'Queue a steering note for the strategist.',
      parameters: { type: 'object', properties: { text: { type:'string' } }, required: ['text'] }
    },
    { type: 'function', name: 'list_plan', description: 'List last planned investigations with done status.', parameters: { type: 'object', properties: {}, required: [] } },
    { type: 'function', name: 'get_current_activity', description: 'What is being investigated right now.', parameters: { type: 'object', properties: {}, required: [] } },
    { type: 'function', name: 'list_hypotheses', description: 'Top hypotheses (filterable)', parameters: { type:'object', properties:{ limit:{type:'number'}, status:{type:'string'} }, required: [] } },
    { type: 'function', name: 'get_system_overview', description: 'SystemOverview graph stats and top nodes by degree.', parameters: { type:'object', properties:{ limit:{type:'number'} }, required: [] } },
    { type: 'function', name: 'search_graph_nodes', description: 'Search nodes by label/type substring in SystemOverview.', parameters: { type:'object', properties:{ query:{type:'string'}, limit:{type:'number'} }, required: [] } },
    { type: 'function', name: 'get_node_details', description: 'Get compact details + incident edges for a node.', parameters: { type:'object', properties:{ node_id:{type:'string'}, edge_limit:{type:'number'} }, required: ['node_id'] } },
    { type: 'function', name: 'get_file_snippet', description: 'Fetch a small code snippet for a relpath or card_id (uses card store).', parameters: { type:'object', properties:{ relpath:{type:'string'}, card_id:{type:'string'}, max_bytes:{type:'number'} }, required: [] } }
    ,{ type: 'function', name: 'get_hypothesis_details', description: 'Fetch full hypothesis details by id (node_refs, files, reasoning).', parameters: { type:'object', properties:{ id:{type:'string'} }, required: ['id'] } }
    ,{ type: 'function', name: 'get_top_hypothesis', description: 'Fetch the highest-confidence hypothesis with enriched details.', parameters: { type:'object', properties:{}, required: [] } }
    ,{ type: 'function', name: 'list_nodes', description: 'List nodes from the SystemOverview graph.', parameters: { type:'object', properties:{ limit:{type:'number'} }, required: [] } }
    ,{ type: 'function', name: 'list_files', description: 'List known file relpaths (optionally filter by substring).', parameters: { type:'object', properties:{ limit:{type:'number'}, contains:{type:'string'} }, required: [] } }
    ,{ type: 'function', name: 'search_repo', description: 'Search repository files for a query and return top file hits with snippets.', parameters: { type:'object', properties:{ query:{type:'string'}, max_files:{type:'number'}, context:{type:'number'}, case_insensitive:{type:'boolean'} }, required: ['query'] } }
  ];
  // Pull active project for persona
  let pid = '';
  try { const r = await fetch('/api/context'); const j = await r.json(); pid = (j && j.project_id) ? j.project_id : ''; } catch(_){}
  const persona = `You are a junior security auditor (the Scout) assisting on the Hound audit for project: ${pid || '(unset)'}. Speak ONLY in first-person singular (“I”). Never use “we”, “we're”, “our”, or “us” — even when referring to the audit team; rephrase as “I”. Keep replies brief (1–2 short sentences), direct, and professional — no flattery or filler. Ground answers in the latest plan, investigations, session status, and the SystemOverview graph. Prefer calling functions to retrieve facts instead of guessing. When the user gives audit instructions (e.g., “investigate X”, “check Y next”, “remember Z”), immediately call enqueue_steering with the exact text, then acknowledge briefly. Answer progress questions flexibly (you may call human_status, but avoid numeric coverage). When asked for more about the current/most promising issue, call get_top_hypothesis (or get_hypothesis_details if an id is given), then display relevant files via get_artifact in the Artifact Viewer. IMPORTANT: Do NOT paste large code blocks in replies. When you load code, say “Loaded in Artifact Viewer” and summarize only the relevant lines.`;
  // Energetic but succinct delivery; emotion via function, never spoken
  const meta = 'Before speaking, set the avatar with set_avatar_state({emotion}) choosing ONLY from: neutral, happy, shy, intense-focus, sad-slightly, crying. Do NOT include any [EMO: ...] tags in speech and never speak the emotion label aloud. Deliver answers in a clear, confident, brisk style (≈10–15% faster), concise and on-topic.';
  const voiceSelEl = document.getElementById('voiceSel');
  const v = voiceSelEl ? voiceSelEl.value : undefined;
  const sessionCfg = { tools, tool_choice: 'auto', instructions: persona + ' ' + meta };
  if (v) sessionCfg.voice = v;
  sendEvent({ type: 'session.update', session: sessionCfg });
}

async function renegotiate(){
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitForIceGatheringComplete(pc);
  const model = 'gpt-4o-realtime-preview-2024-12-17';
  const voiceSel = document.getElementById('voiceSel');
  const voice = voiceSel ? (voiceSel.value || 'shimmer') : 'shimmer';
  // Add client-side timeout so we don't block the UI indefinitely
  const ctrl = new AbortController();
  const to = setTimeout(()=>{ try{ ctrl.abort(); }catch(_){ } }, 8000);
  let resp, sdp;
  try{
    resp = await fetch(`/webrtc/offer?model=${encodeURIComponent(model)}&voice=${encodeURIComponent(voice)}`,
      { method:'POST', headers:{'Content-Type':'application/sdp'}, body: pc.localDescription.sdp, signal: ctrl.signal });
    sdp = await resp.text();
  } finally { clearTimeout(to); }
  if (!resp || !resp.ok || !sdp || !sdp.startsWith('v=')) throw new Error('SDP negotiation failed');
  await pc.setRemoteDescription({type:'answer', sdp});
}

function waitForIceGatheringComplete(pc, timeoutMs=4000){
  if (pc.iceGatheringState==='complete') return Promise.resolve();
  return new Promise(res=>{
    function chk(){ if (pc.iceGatheringState==='complete'){ try{ pc.removeEventListener('icegatheringstatechange', chk); }catch(_){ } clearTimeout(to); res(); } }
    const to = setTimeout(()=>{ try{ pc.removeEventListener('icegatheringstatechange', chk); }catch(_){ } res(); }, timeoutMs);
    pc.addEventListener('icegatheringstatechange', chk);
  });
}

// Handle Realtime events
let emoCaptured = false;
function handleEvent(evt){
  const t = evt.type;
  // Show transcription of the user's voice input in chat as 'user'
  if (t==='input_audio_transcription.delta' || t==='input_audio_transcript.delta'){
    const delta = evt.delta || '';
    if (delta){
      if (!lastUserBubble) lastUserBubble = chatAdd('user','');
      // Append delta to the latest user bubble
      lastUserBubble.textContent += delta;
      userTranscript += delta;
      const container = chatEl || document.getElementById('chat');
      if (container) container.scrollTop = container.scrollHeight;
    }
    return; // handled
  }
  if (t==='input_audio_transcription.completed' || t==='input_audio_transcript.completed'){
    // Finalize the current user bubble and trigger a model turn automatically
    lastUserBubble = null;
    userTranscript = '';
    // Auto-turn: request response after user finishes speaking
    try{ sendEvent({ type:'response.create', response:{ modalities:['audio','text'] } }); }catch(_){ }
    return;
  }
  if (t==='response.text.delta'){
    let delta = evt.delta || '';
    if (!emoCaptured && delta){
      // Attempt to capture [EMO: ...] at the start
      const bubbleText = (lastAssistantBubble ? lastAssistantBubble.textContent : '');
      const combined = (bubbleText + delta);
      const m = combined.match(/^\s*\[\s*EMO:\s*([A-Za-z]+)\s*\]\s*(.*)$/i);
      if (m){
        const emo = m[1].toLowerCase();
        setAvatarLegacy(emo);
        emoCaptured = true;
        // Replace content after removing tag
        if (lastAssistantBubble) lastAssistantBubble.textContent = '';
        delta = m[2] || '';
      }
    }
    if (delta) chatAppendDelta('assistant', delta);
  }
  if (t==='response.audio_transcript.delta'){
    let delta = evt.delta || '';
    assistantTranscript += delta;
    // Ensure there is a bubble, then append transcript words live
    if (!lastAssistantBubble) { lastAssistantBubble = chatAdd('assistant',''); }
    if (!emoCaptured && delta){
      const bubbleText = (lastAssistantBubble ? lastAssistantBubble.textContent : '');
      const combined = (bubbleText + delta);
      const m = combined.match(/^\s*\[\s*EMO:\s*([A-Za-z]+)\s*\]\s*(.*)$/i);
      if (m){
        const emo = m[1].toLowerCase();
        setAvatarLegacy(emo);
        emoCaptured = true;
        if (lastAssistantBubble) lastAssistantBubble.textContent = '';
        delta = m[2] || '';
      }
    }
    if (delta) chatAppendDelta('assistant', delta);
  }
  if (t==='response.audio.start' || t==='response.output_audio.begin'){ setPill('Speaking'); lastAssistantBubble = chatAdd('assistant',''); assistantTranscript=''; }
  if (t==='response.done' || t==='response.completed'){ if (!emoCaptured){ const emo = classifyEmotion(assistantTranscript); setAvatarState({ emotion: emo }); } setPill('Idle'); lastAssistantBubble=null; assistantTranscript=''; emoCaptured=false; }
  // Function calling
  if (t==='response.function_call_arguments.delta'){ /* streaming args; wait for done */ }
  if (t==='response.done' && evt.response && evt.response.output && evt.response.output[0] && evt.response.output[0].type==='function_call'){
    const item = evt.response.output[0];
    const name = item.name; const callId = item.call_id; let args={};
    try { args = JSON.parse(item.arguments||'{}'); } catch(_){ args={}; }
    // Handle set_emotion locally for immediate UI feedback
    if (name === 'set_emotion' && args && args.value){ setAvatarLegacy(String(args.value)); }
    if (name === 'set_avatar_state'){
      try{
        setAvatarState({ emotion: String(args.emotion||'neutral'), intensity: String(args.intensity||'subtle'), look: String(args.look||'center'), holdMs: Number(args.holdMs||0) });
      }catch(_){ /* ignore */ }
    }
    // Skip server roundtrip for local-only UI tools
    const localOnly = (name === 'set_emotion' || name === 'set_avatar_state');
    (localOnly ? Promise.resolve({ ok:true }) : invokeTool(name, args)).then((out)=>{
      try{
        if (name==='get_file_snippet' && out && out.snippet){ showCode(out.snippet, out.relpath||''); }
        if (name==='get_artifact' && out && Array.isArray(out.artifacts) && out.artifacts.length){ showCode(out.artifacts[0].content||'', out.artifacts[0].relpath||''); }
        if ((name==='get_hypothesis_details' || name==='get_top_hypothesis') && out && out.hypothesis && Array.isArray(out.hypothesis.files) && out.hypothesis.files.length){
          const rel = out.hypothesis.files[0];
          invokeTool('get_artifact', { relpath: rel, max_bytes: 200000 }).then((o2)=>{
            if (o2 && Array.isArray(o2.artifacts) && o2.artifacts.length){ showCode(o2.artifacts[0].content||'', o2.artifacts[0].relpath||''); }
          });
        }
      }catch(_){ /* ignore UI errors */ }
      // Sanitize tool output sent back to the model to prevent it from pasting large code into replies.
      let outForModel = out;
      try{
        if (name === 'get_artifact' && out && Array.isArray(out.artifacts)){
          outForModel = { ok: out.ok, artifacts: (out.artifacts||[]).map(a=>({ relpath: a.relpath, card_id: a.card_id, size_bytes: a.size_bytes, truncated: a.truncated, displayed_in_viewer: true })) };
        }
        if (name === 'get_file_snippet' && out){
          outForModel = { ok: out.ok, relpath: out.relpath, card_id: out.card_id, displayed_in_viewer: true };
        }
        if (name === 'search_repo' && out && Array.isArray(out.results)){
          outForModel = { ok: out.ok, results: (out.results||[]).map(r=>({ relpath: r.relpath, lineno: r.lineno, snippet: '[redacted in reply; opened in viewer if requested]' })) };
        }
      }catch(_){ outForModel = out; }
      sendEvent({ type:'conversation.item.create', item: { type:'function_call_output', call_id: callId, output: JSON.stringify(outForModel) } });
      sendEvent({ type:'response.create', response:{ modalities:['audio','text'] } });
    });
  }
}

async function invokeTool(name, args){
  try{
    const r = await fetch(`/api/tool/${encodeURIComponent(name)}`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(args||{}) });
    return await r.json();
  }catch(e){ return { ok:false, error: String(e) }; }
}

window.addEventListener('DOMContentLoaded', ()=>{
  // Ensure avatar has a graceful fallback if Inuzumi asset not present
  try{
    const img = document.getElementById('avatarImg');
    if (img){
      // Only JPGs: if initial fails, try common default
      const onDefault = ()=>{ try{ img.onerror=null; img.src='/static/inuzumi/Inuzumi-neutral.jpg'; }catch(_){ } };
      img.onerror = onDefault;
    }
  }catch(_){ }
  const connectBtn = document.getElementById('connectBtn');
  const micBtn = document.getElementById('micBtn');
  const pttBtn = document.getElementById('pttBtn');
  const textForm = document.getElementById('textForm');
  const textInput = document.getElementById('textInput');
  const actStartBtn = document.getElementById('actStartBtn');
  const actStopBtn = document.getElementById('actStopBtn');
  const projectInput = document.getElementById('projectInput');
  const activity = document.getElementById('activity');
  const planView = document.getElementById('planView');
  const tabActivity = document.getElementById('tabActivity');
  const tabPlan = document.getElementById('tabPlan');
  const tabHypo = document.getElementById('tabHypo');
  const hypoView = document.getElementById('hypoView');
  codeViewEl = document.getElementById('codeView');
  // Ensure chat container is cached even before connecting
  if (!chatEl) chatEl = document.getElementById('chat');
  const nowInv = document.getElementById('nowInvestigating');

  // Track whether we attached to an instance via Start
  let isAttached = false;
  let attachedProjectId = '';
  let attachedInstanceId = '';
  // Expose project id for detail helpers
  try { window.attachedProjectId = attachedProjectId; } catch(_) {}

  // Populate instances into a datalist-like UX (basic)
  fetch('/api/instances').then(r=>r.json()).then(j=>{
    if (!j || !Array.isArray(j.instances)) return;
    const items = j.instances || [];
    // Build a simple list of project_id (pid)
    const list = items.map(x=>`${x.project_id || 'unknown'} | ${x.id}`);
    projectInput.setAttribute('list','instancesList');
    let dl = document.getElementById('instancesList');
    if (!dl) { dl = document.createElement('datalist'); dl.id='instancesList'; document.body.appendChild(dl); }
    dl.innerHTML = list.map(v=>`<option value=\"${v}\"></option>`).join('');
    // If current input value does not correspond to any returned project, default to most recent alive (or most recent overall)
    try{
      const current = (projectInput.value||'').trim();
      const match = items.find(x=> String(x.project_id||'').trim() === current);
      const hasExact = !!match;
      const isMatchAlive = !!(match && match.alive);
      if (!current || !hasExact || !isMatchAlive){
        // Prefer alive instances first
        let prefer = items.filter(x=> x.alive);
        if (prefer.length === 0) prefer = items.slice();
        if (prefer.length > 0){
          const chosen = prefer[0]; // server sorts by started_at desc
          projectInput.value = String(chosen.project_id||'');
          // Persist active project on server for tool calls
          fetch('/api/context', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ project_id: projectInput.value }) }).catch(()=>{});
        }
      }
    }catch(_){ /* ignore */ }
  }).catch(()=>{});

  connectBtn.addEventListener('click', async ()=>{
    connectBtn.disabled = true;
    try{ await connect(); micBtn.disabled=false; pttBtn.disabled=false; }catch(e){ connectBtn.disabled=false; }
  });
  const voiceSelect = document.getElementById('voiceSel');
  if (voiceSelect) voiceSelect.addEventListener('change', ()=>{ renegotiate().catch(()=>{}); });
  // No style selector anymore; default mic action is enable until toggled
  micBtn.addEventListener('click', enableMic);

  // PTT: basic toggle (no VAD override here)
  pttBtn.addEventListener('mousedown', ()=>{ setPill('Listening'); setAvatarState({ emotion:'intense-focus' }); });
  pttBtn.addEventListener('mouseup', ()=>{ setPill('Thinking'); sendEvent({ type:'response.create', response:{ modalities:['audio','text'] } }); const hint=document.getElementById('pttStatus'); if(hint){ hint.textContent='Sent'; hint.hidden=false; setTimeout(()=>hint.hidden=true, 800);} });

  textForm.addEventListener('submit', (ev)=>{
    ev.preventDefault();
    const text = (textInput.value||'').trim(); if(!text) return;
    chatAdd('you', text); textInput.value='';
    sendEvent({ type:'conversation.item.create', item:{ type:'message', role:'user', content:[{ type:'input_text', text }] } });
    sendEvent({ type:'response.create', response:{ modalities:['audio','text'] } });
  });

  // Activity stream controls
  actStartBtn.addEventListener('click', async ()=>{
    const raw = (projectInput.value||'').trim();
    if (!raw) { return; }
    // Do NOT block on realtime connect; try in background for chat, but start telemetry immediately
    (async ()=>{
      try{
        const already = (dc && dc.readyState==='open') || (pc && pc.connectionState==='connected');
        if (!already){
          setPill('Connecting');
          await connect();
          try{ micBtn.disabled=false; pttBtn.disabled=false; }catch(_){ }
          setPill('Idle');
        }
      }catch(_){ /* ignore; chat is optional for telemetry */ }
    })();
    // Allow either "project_id | instance_id" or plain project id fallback
    let proj = raw; let instId = '';
    const m = raw.split('|');
    if (m.length >= 2) { proj = m[0].trim(); instId = m[1].trim(); }
    // Persist active project on server for tool calls
    fetch('/api/context', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ project_id: proj }) }).catch(()=>{});
    // Resolve telemetry instance id when not provided
    try{
      if (!instId){ instId = await resolveInstanceId(proj); }
    }catch(_){ /* ignore */ }
    if (!instId){ appendEvt({ time:'', label:'Status', cls:'tag', text:'No telemetry instance (start agent with --telemetry)' }); return; }
    startActivity(proj, instId);
    // Mark UI as attached
    isAttached = true; attachedProjectId = proj; attachedInstanceId = instId;
    try { window.attachedProjectId = attachedProjectId; } catch(_) {}
    actStartBtn.disabled = true; actStopBtn.disabled = false;
  });
  actStopBtn.addEventListener('click', ()=>{
    stopActivity();
    isAttached = false; attachedProjectId = ''; attachedInstanceId = '';
    try { window.attachedProjectId = attachedProjectId; } catch(_) {}
    actStartBtn.disabled = false; actStopBtn.disabled = true;
  });

  // Steering submit
  const steerForm = document.getElementById('steerForm');
  const steerInput = document.getElementById('steerInput');
  const steerStatus = document.getElementById('steerStatus');
  if (steerForm) steerForm.addEventListener('submit', async (ev)=>{
    ev.preventDefault();
    const text = (steerInput?.value||'').trim(); if (!text) return;
    // Persist active project and send steering
    const raw = (projectInput?.value||'').trim();
    let proj = raw; if (!proj){ try{ const cx=await fetch('/api/context').then(r=>r.json()); proj=cx.project_id||''; }catch(_){} }
    if (proj) { fetch('/api/context', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ project_id: proj }) }).catch(()=>{}); }
    try{
      const r = await fetch('/api/tool/enqueue_steering', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) });
      const j = await r.json();
      if (j && j.ok){
        if (steerStatus){ steerStatus.textContent='Queued'; steerStatus.hidden=false; setTimeout(()=> steerStatus.hidden=true, 1200); }
        const ts = new Date().toTimeString().split(' ')[0];
        appendEvt({ time:`[${ts}]`, label:'Status', cls:'tag', text:`Steering queued: ${text}` });
        steerInput.value='';
      } else {
        if (steerStatus){ steerStatus.textContent='Failed'; steerStatus.hidden=false; setTimeout(()=> steerStatus.hidden=true, 2000); }
      }
    }catch(_){ if (steerStatus){ steerStatus.textContent='Failed'; steerStatus.hidden=false; setTimeout(()=> steerStatus.hidden=true, 2000); } }
  });

  // ----- Plan tab handling -----
  let planTimer = null;
  let hypoTimer = null;
  function showTab(which){
    if (which === 'plan'){
      tabPlan.classList.add('active'); tabActivity.classList.remove('active');
      tabHypo.classList.remove('active');
      planView.classList.remove('hidden'); activity.classList.add('hidden'); hypoView.classList.add('hidden');
      // Fetch immediately and then periodically (only when attached)
      if (!isAttached){ planView.innerHTML = '<div class="plan-item"><div class="plan-body">No active instance. Click Start to attach.</div></div>'; return; }
      refreshPlan();
      if (planTimer) { clearInterval(planTimer); planTimer = null; }
      planTimer = setInterval(()=>{ refreshPlan(); }, 4000);
      if (hypoTimer){ clearInterval(hypoTimer); hypoTimer=null; }
    } else if (which === 'hypo'){
      tabHypo.classList.add('active'); tabActivity.classList.remove('active'); tabPlan.classList.remove('active');
      hypoView.classList.remove('hidden'); activity.classList.add('hidden'); planView.classList.add('hidden');
      if (!isAttached){ hypoView.innerHTML = '<div class="hypo-item"><div class="hypo-body">No active instance. Click Start to attach.</div></div>'; return; }
      refreshHypotheses();
      if (hypoTimer) { clearInterval(hypoTimer); hypoTimer = null; }
      hypoTimer = setInterval(()=>{ refreshHypotheses(); }, 5000);
      if (planTimer){ clearInterval(planTimer); planTimer=null; }
    } else {
      tabActivity.classList.add('active'); tabPlan.classList.remove('active');
      tabHypo.classList.remove('active');
      activity.classList.remove('hidden'); planView.classList.add('hidden'); hypoView.classList.add('hidden');
      if (planTimer) { clearInterval(planTimer); planTimer = null; }
      if (hypoTimer) { clearInterval(hypoTimer); hypoTimer = null; }
    }
  }
  async function refreshPlan(){
    try{
      if (!isAttached || !attachedProjectId){ planView.innerHTML = '<div class="plan-item"><div class="plan-body">No active instance. Click Start to attach.</div></div>'; return; }
      const j = await invokeTool('list_plan', { project_id: attachedProjectId });
      if (!j || !j.ok){ planView.innerHTML = '<div class="plan-item"><div class="plan-body">No plan available yet.</div></div>'; return; }
      renderPlan(j.plan||[]);
    }catch(_){ planView.innerHTML = '<div class="plan-item"><div class="plan-body">Failed to load plan.</div></div>'; }
  }
  function renderPlan(items){
    if (!planView) return;
    if (!Array.isArray(items) || items.length===0){ planView.innerHTML = '<div class="plan-item"><div class="plan-body">No plan items yet</div></div>'; return; }
    let html = '';
    for (const it of items){
      const status = String(it.status||'PENDING').toUpperCase();
      let mark = '•'; let cls = 'style="color:#7fd5ff"';
      if (status==='DONE'){ mark='✓'; cls='style="color:#7ee198"'; }
      if (status==='ACTIVE'){ mark='▶'; cls='style="color:#ffd479"'; }
      const pr = (it.priority!=null) ? `Priority: ${it.priority}` : '';
      const imp = it.impact ? `Impact: ${it.impact}` : '';
      const cat = it.category ? `Category: ${it.category}` : '';
      const fa = (Array.isArray(it.focus_areas) && it.focus_areas.length) ? `Focus: ${it.focus_areas.slice(0,3).join(', ')}` : '';
      const meta = [pr, imp, cat, fa].filter(Boolean).join(' | ');
      html += `<div class="plan-item"><div class="plan-mark" ${cls}>${mark}</div><div class="plan-body"><div class="plan-goal">${esc(it.goal||'Unknown')}</div>${meta?`<div class="plan-meta">${esc(meta)}</div>`:''}</div></div>`;
    }
    planView.innerHTML = html;
  }
  async function refreshHypotheses(){
    try{
      if (!isAttached || !attachedProjectId){ hypoView.innerHTML = '<div class="hypo-item"><div class="hypo-body">No active instance. Click Start to attach.</div></div>'; return; }
      const j = await invokeTool('list_hypotheses', { project_id: attachedProjectId, limit: 50 });
      if (!j || !j.ok){ hypoView.innerHTML = '<div class="hypo-item"><div class="hypo-body">No findings yet.</div></div>'; return; }
      renderHypotheses(j.hypotheses||[]);
    }catch(_){ hypoView.innerHTML = '<div class="hypo-item"><div class="hypo-body">Failed to load findings.</div></div>'; }
  }
  // Expose for detail helpers
  try { window.refreshHypotheses = refreshHypotheses; } catch(_) {}
  function renderHypotheses(items){
    if (!hypoView) return;
    if (!Array.isArray(items) || items.length===0){ hypoView.innerHTML = '<div class="hypo-item"><div class="hypo-body">No findings yet.</div></div>'; return; }
    let html = '';
    for (const h of items){
      const conf = (typeof h.confidence==='number') ? Math.round(h.confidence*100) : (h.confidence||0);
      const status = (h.status||'proposed');
      const title = esc(h.title||'(untitled)');
      // Remove native title tooltips; use rich hover cards
      const rejectedCls = (String(status).toLowerCase()==='rejected') ? ' rejected' : '';
      html += `<div class="hypo-item${rejectedCls}">
        <div class="hypo-mark">${conf}%</div>
        <div class="hypo-body">
          <div class="hypo-title" data-hid="${esc(h.id||'')}" data-desc="${esc(h.description||'')}" data-type="${esc(h.vulnerability_type||'')}" data-status="${esc(status)}" data-conf="${String(conf)}">${title}</div>
          <div class="hypo-meta">${esc(status)}</div>
        </div>
        <div class="hypo-actions">
          <button class="confirm" data-hid="${esc(h.id||'')}" title="Mark confirmed (100%)">Confirm</button>
          <button class="reject" data-hid="${esc(h.id||'')}" title="Mark rejected (0%)">Reject</button>
          <button class="details" data-hid="${esc(h.id||'')}">Details</button>
        </div>
      </div>`;
    }
    hypoView.innerHTML = html;
    // Wire action buttons
    for (const btn of hypoView.querySelectorAll('button.confirm')){
      btn.addEventListener('click', async (ev)=>{
        const hid = ev.currentTarget.getAttribute('data-hid')||'';
        if (!hid) return;
        try{ await invokeTool('set_hypothesis_status', { id: hid, status: 'confirmed', project_id: attachedProjectId }); }catch(_){ }
        refreshHypotheses();
      });
    }
    for (const btn of hypoView.querySelectorAll('button.reject')){
      btn.addEventListener('click', async (ev)=>{
        const hid = ev.currentTarget.getAttribute('data-hid')||'';
        if (!hid) return;
        try{ await invokeTool('set_hypothesis_status', { id: hid, status: 'rejected', project_id: attachedProjectId }); }catch(_){ }
        refreshHypotheses();
      });
    }
    for (const btn of hypoView.querySelectorAll('button.details')){
      btn.addEventListener('click', async (ev)=>{
        const hid = ev.currentTarget.getAttribute('data-hid')||'';
        if (!hid) return;
        await showHypoDetails(hid);
      });
    }
    // Hover cards on titles
    for (const t of hypoView.querySelectorAll('.hypo-title')){
      t.addEventListener('mouseenter', (ev)=>{
        const el = ev.currentTarget;
        const info = {
          title: el.textContent||'',
          type: el.getAttribute('data-type')||'',
          status: el.getAttribute('data-status')||'',
          conf: el.getAttribute('data-conf')||'0',
          desc: el.getAttribute('data-desc')||''
        };
        showHoverCard(el, info);
      });
      t.addEventListener('mouseleave', ()=>{ hideHoverCardSoon(); });
      t.addEventListener('click', async (ev)=>{
        const hid = ev.currentTarget.getAttribute('data-hid')||'';
        if (!hid) return;
        await showHypoDetails(hid);
      });
    }
  }
  if (tabActivity && tabPlan){
    tabActivity.addEventListener('click', ()=> showTab('activity'));
    tabPlan.addEventListener('click', ()=> showTab('plan'));
    if (tabHypo){ tabHypo.addEventListener('click', ()=> showTab('hypo')); }
  }

  function friendlyTag(action){
    const a = (action||'').toLowerCase();
    if (a==='load_nodes' || a==='load_node' || a==='fetch_code') return {label:'Fetch code', cls:'fetch'};
    if (a==='update_node') return {label:'Memorize fact', cls:'memo'};
    if (a==='add_edge') return {label:'Relate facts', cls:'graph'};
    if (a==='query_graph' || a==='focus' || a==='summarize') return {label:'Explore graph', cls:'graph'};
    if (a==='propose_hypothesis') return {label:'Finding', cls:'hyp'};
    if (a==='update_hypothesis') return {label:'Finding update', cls:'hyp'};
    if (a==='deep_think') return {label:'Strategist', cls:'think'};
    if (a==='status') return {label:'Status', cls:'tag'};
    return {label: (action||'Act'), cls:'tag'};
  }
  function stripAnsi(s){ return String(s||'').replace(/\x1b\[[0-9;]*m/g,''); }
  function appendEvt(obj){
    if (!activity) return;
    const d = document.createElement('div'); d.className='evt';
    const time = document.createElement('div'); time.className='time'; time.textContent = obj.time || '';
    const tag = document.createElement('div'); tag.className = 'tag ' + (obj.cls||''); tag.textContent = obj.label || 'Act';
    const msg = document.createElement('div'); msg.className='msg'; msg.textContent = obj.text || '';
    d.appendChild(time); d.appendChild(tag); d.appendChild(msg);
    activity.appendChild(d);
    activity.scrollTop = activity.scrollHeight;
  }
let lastIter = 0;
const _seenEvt = new Set();
function _dedupeKey(j){ return `${j.type||''}|${j.action||''}|${j.iteration||''}|${(j.message||'').slice(0,80)}|${(j.reasoning||'').slice(0,80)}`; }
  function handleActivityData(data){
    // Try JSON first (telemetry SSE)
    try{
      const j = JSON.parse(data);
      const ts = j.ts ? new Date(j.ts*1000) : new Date();
      const tstr = ts.toTimeString().split(' ')[0];
      // Only show substantive agent events; drop heartbeats/status/steer noise
      const showTypes = new Set(['decision','result','executing','analyzing','complete','generating_report','strategist','usage']);
      if (!showTypes.has(j.type)) return;
      // Dedupe initial replays
      const key = _dedupeKey(j);
      if (_seenEvt.has(key)) return;
      _seenEvt.add(key);
      if (_seenEvt.size > 200) { // keep small
        const it = _seenEvt.values().next();
        _seenEvt.delete(it.value);
      }
      const action = j.action || '';
      const tag = (j.type === 'decision' && action) ? friendlyTag(action) : friendlyTag(j.type||'');
      let text = '';
      if (j.type === 'decision'){
        const thought = j.reasoning || '';
        text = `Action: ${tag.label}` + (thought ? `\nThought: ${thought}` : '');
      } else if (j.type === 'result' || j.type === 'executing' || j.type === 'analyzing' || j.type === 'complete' || j.type === 'generating_report'){
        text = j.message || '';
      } else {
        text = j.message || j.reasoning || '';
      }
      lastIter = j.iteration || lastIter;
      appendEvt({ time: `[${tstr}] #${lastIter||'-'}`, label: tag.label, cls: tag.cls, text });
      // Do not mirror activity stream into chat; keep chat for user ↔ assistant messages only
      return;
    }catch(_){ /* not JSON */ }
    // Fallback: parse CLI log lines
    const line = stripAnsi(String(data||''));
    let m;
    if ((m = line.match(/Model Decision.*Iteration\s+(\d+)/i))){ lastIter = parseInt(m[1]||'0',10); pendingEvt=null; return; }
    if ((m = line.match(/\bAction:\s*([A-Za-z_]+)/i))){ const a=m[1]; const tag=friendlyTag(a); pendingEvt = { label: tag.label, cls: tag.cls, text: `Action: ${tag.label}` }; return; }
    if ((m = line.match(/\bThought:\s*(.*)$/i))){ const txt=m[1]; if (pendingEvt){ pendingEvt.text += `\nThought: ${txt}`; appendEvt({ time:'', ...pendingEvt }); pendingEvt=null; } else { appendEvt({ time:'', label:'Thought', cls:'think', text: txt }); } return; }
    if ((m = line.match(/\bResult:\s*(.*)$/i))){ const txt=m[1]; appendEvt({ time:'', label:'Result', cls:'tag', text: txt }); return; }
    // Show any other non-empty lines as info in log-tail mode
    const txt = line.trim(); if (txt) { appendEvt({ time:'', label:'Info', cls:'tag', text: txt }); }
  }
  function startActivity(proj, instId){
    stopActivity();
    const url = `/api/instance/status?id=${encodeURIComponent(instId)}`;
    activityES = new EventSource(url);
    activityES.onmessage = (e)=>{ handleActivityData(e.data); };
    activityES.onerror = ()=>{ appendEvt({ time:'', label:'warn', cls:'tag', text:'stream error' }); };
    // Start pinned "Now Investigating" updater
    if (nowInvTimer) { clearInterval(nowInvTimer); nowInvTimer = null; }
    nowInvTimer = setInterval(async ()=>{
      try{
        const r = await fetch('/api/tool/get_recent_activity', { method:'POST', headers:{'Content-Type':'application/json'}, body: '{}' });
        const j = await r.json();
        if (j && j.ok){
          const s = j.current_goal || j.summary || '';
          if (nowInv) nowInv.textContent = s ? `Now investigating: ${s}` : 'Now investigating: —';
        }
      }catch(_){ /* ignore */ }
    }, 2000);
    // Prefill stream with recent events so the UI isn't empty on connect
    fetch(`/api/instance/recent?id=${encodeURIComponent(instId)}&limit=60`).then(r=>r.json()).then(j=>{
      if (!j || !Array.isArray(j.events)) return;
      for (const ev of j.events){
        try { handleActivityData(JSON.stringify(ev)); } catch(_){}
      }
    }).catch(()=>{});
  }
  function stopActivity(){ if (activityES){ activityES.close(); activityES=null; } }
  // Clear pinned updater when leaving page
  window.addEventListener('beforeunload', ()=>{ if (nowInvTimer){ clearInterval(nowInvTimer); nowInvTimer=null; } });

  async function resolveInstanceId(proj){
    try{
      const r = await fetch('/api/instances');
      if (!r.ok) return '';
      const j = await r.json();
      const items = (j && j.instances) ? j.instances : [];
      // Exact match on project_id; if multiple, choose the latest started_at
      const matches = items.filter(x=> (String(x.project_id||'').trim() === proj.trim()));
      if (matches.length === 0){
        // Also try to match suffix if user pasted a path and registry stored a short name, or vice versa
        const tail = proj.split('/').slice(-1)[0];
        const alt = items.filter(x=> String(x.project_id||'').endsWith('/'+tail) || String(x.project_id||'')===tail);
        if (alt.length === 0) return '';
        alt.sort((a,b)=> (b.started_at||0) - (a.started_at||0));
        return alt[0].id;
      }
      matches.sort((a,b)=> (b.started_at||0) - (a.started_at||0));
      return matches[0].id;
    }catch{ return ''; }
  }

  // Initialize project input from server context if any
  fetch('/api/context').then(r=>r.json()).then(j=>{
    if (j && j.project_id && !projectInput.value) projectInput.value = j.project_id;
  }).catch(()=>{});
});

// --- Finding detail/hover helpers ---
function ensureHoverCard(){
  if (hoverCardEl) return hoverCardEl;
  const d = document.createElement('div');
  d.className = 'hovercard';
  d.style.display = 'none';
  document.body.appendChild(d);
  hoverCardEl = d;
  return d;
}
let hoverHideTimer = null;
function showHoverCard(anchorEl, info){
  const d = ensureHoverCard();
  if (hoverHideTimer){ clearTimeout(hoverHideTimer); hoverHideTimer=null; }
  const conf = parseInt(info.conf||'0',10);
  d.innerHTML = `
    <div class="title">${esc(info.title||'')}</div>
    <div class="meta">${esc(info.type||'')}${info.type?' • ':''}${esc(info.status||'')} • ${conf}%</div>
    <div class="text">${esc((info.desc||'').slice(0,280))}${(info.desc||'').length>280?'…':''}</div>
  `;
  const r = anchorEl.getBoundingClientRect();
  const pad = 8;
  d.style.display = 'block';
  const h = d.offsetHeight || 0;
  const top = window.scrollY + r.top - h - pad;
  const left = window.scrollX + r.left;
  d.style.left = Math.max(12, left) + 'px';
  d.style.top = Math.max(12, top) + 'px';
}
function hideHoverCardSoon(){
  if (!hoverCardEl) return;
  if (hoverHideTimer){ clearTimeout(hoverHideTimer); }
  hoverHideTimer = setTimeout(()=>{ if (hoverCardEl) hoverCardEl.style.display='none'; }, 120);
}

async function showHypoDetails(hid){
  if (!hypoDetailEl) hypoDetailEl = document.getElementById('hypoDetail');
  if (!hypoDetailEl) return;
  try{
    const j = await invokeTool('get_hypothesis_details', { id: hid, project_id: window.attachedProjectId });
    if (!j || !j.ok || !j.hypothesis){
      hypoDetailEl.innerHTML = '<div class="section">Failed to load details.</div>';
      hypoDetailEl.classList.remove('hidden');
      return;
    }
    renderHypoDetail(j.hypothesis);
  }catch(_){
    hypoDetailEl.innerHTML = '<div class="section">Failed to load details.</div>';
    hypoDetailEl.classList.remove('hidden');
  }
}

function sevBadge(sev){
  const s = String(sev||'').toLowerCase();
  if (s.startsWith('h')) return '<span class="badge sev-high">High</span>';
  if (s.startsWith('m')) return '<span class="badge sev-med">Medium</span>';
  if (s.startsWith('l')) return '<span class="badge sev-low">Low</span>';
  return '';
}
function statusBadge(st){
  const s = String(st||'').toLowerCase();
  if (s==='confirmed') return '<span class="badge status-confirmed">Confirmed</span>';
  if (s==='rejected') return '<span class="badge status-rejected">Rejected</span>';
  return '<span class="badge status-proposed">Proposed</span>';
}

function renderHypoDetail(h){
  if (!hypoDetailEl) return;
  const conf = Math.round((typeof h.confidence==='number'?h.confidence:parseFloat(h.confidence||'0'))*100) || (h.confidence||0);
  const badges = [statusBadge(h.status), sevBadge(h.severity), h.vulnerability_type?`<span class="badge">${esc(h.vulnerability_type)}</span>`:''].filter(Boolean).join(' ');
  const files = Array.isArray(h.files)?h.files:[];
  const evid = Array.isArray(h.evidence)?h.evidence:[];
  hypoDetailEl.innerHTML = `
    <div class="hd-head">
      <div>
        <div class="hd-title">${esc(h.title||'(untitled finding)')}</div>
        <div class="hd-meta">${badges}</div>
      </div>
      <button class="hd-close" title="Close">Close</button>
    </div>
    <div class="row" aria-hidden="true">
      <div style="min-width:120px;">Confidence</div>
      <div class="progress" style="flex:1"><div class="bar" style="width:${conf}%"></div></div>
      <div style="min-width:40px; text-align:right;">${conf}%</div>
    </div>
    <div class="hd-body">
      <div class="section">
        <h4>Description</h4>
        <div class="desc">${esc(h.description||'No description')}</div>
      </div>
      <div class="section">
        <h4>Related Files</h4>
        <div class="files">${files.length? files.map(r=>`<div><a href="#" data-rel="${esc(r)}">${esc(r)}</a></div>`).join('') : '<div>—</div>'}</div>
        <div class="actions">
          <button class="confirm" data-hid="${esc(h.id||'')}">Confirm</button>
          <button class="reject" data-hid="${esc(h.id||'')}">Reject</button>
        </div>
      </div>
      <div class="section" style="grid-column: 1 / span 2;">
        <h4>Evidence</h4>
        <div class="evidence">${evid.length? evid.map(e=>`• ${esc(e)}`).join('<br/>') : '—'}</div>
      </div>
    </div>
  `;
  hypoDetailEl.classList.remove('hidden');
  // Wire close
  const closeBtn = hypoDetailEl.querySelector('.hd-close');
  if (closeBtn) closeBtn.addEventListener('click', ()=>{ hypoDetailEl.classList.add('hidden'); });
  // Wire confirm/reject
  const cBtn = hypoDetailEl.querySelector('button.confirm');
  if (cBtn) cBtn.addEventListener('click', async ()=>{
    try{ await invokeTool('set_hypothesis_status', { id: h.id, status: 'confirmed', project_id: window.attachedProjectId }); }catch(_){ }
    try{ await window.refreshHypotheses(); }catch(_){ }
  });
  const rBtn = hypoDetailEl.querySelector('button.reject');
  if (rBtn) rBtn.addEventListener('click', async ()=>{
    try{ await invokeTool('set_hypothesis_status', { id: h.id, status: 'rejected', project_id: window.attachedProjectId }); }catch(_){ }
    try{ await window.refreshHypotheses(); }catch(_){ }
  });
  // Wire file links: load first artifact into viewer
  for (const a of hypoDetailEl.querySelectorAll('.files a[data-rel]')){
    a.addEventListener('click', async (ev)=>{
      ev.preventDefault();
      const rel = ev.currentTarget.getAttribute('data-rel')||'';
      if (!rel) return;
      try{
        const j = await invokeTool('get_artifact', { relpath: rel, max_bytes: 200000, project_id: window.attachedProjectId });
        if (j && j.ok && j.artifacts && j.artifacts.length){
          showCode(j.artifacts[0].content||'', j.artifacts[0].relpath||'');
        }
      }catch(_){ /* ignore */ }
    });
  }
}
