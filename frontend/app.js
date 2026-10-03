const panel=document.getElementById('chat-panel'), launcher=document.getElementById('chat-launcher');
const question=document.getElementById('question'), messages=document.getElementById('messages');
const send=document.getElementById('send'), mode=document.getElementById('search-type'), newChat=document.getElementById('clear-chat');
let busy=false, conversationId=null;
try {conversationId=localStorage.getItem('fieldnotes-conversation');} catch (_) {}
newChat.textContent='New chat';
function remember(id){conversationId=id;try{if(id)localStorage.setItem('fieldnotes-conversation',id);else localStorage.removeItem('fieldnotes-conversation');}catch(_){}}
function openChat(){panel.hidden=false;launcher.setAttribute('aria-expanded','true');question.focus();}
function closeChat(){panel.hidden=true;launcher.setAttribute('aria-expanded','false');launcher.focus();}
function scroll(){messages.scrollTop=messages.scrollHeight;}
function bubble(text,kind){const el=document.createElement('div');el.className='message '+kind;el.textContent=text;messages.append(el);scroll();return el;}
function setBusy(value){busy=value;send.disabled=value;mode.disabled=value;newChat.disabled=value;}
function localURL(value){const url=new URL(value,location.origin);if(url.origin!==location.origin||!url.pathname.startsWith('/api/media/'))throw new Error('Invalid evidence link');return url.href;}
async function request(url,options){const response=await fetch(url,options),data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Unable to complete the request.');return data;}
function renderAnswer(data){const el=bubble(data.answer,'assistant');for(const s of data.sources||[]){const a=document.createElement('a');a.className='source-link';a.textContent='['+s.label+'] '+s.citation+' ↗';a.href=localURL(s.page_url);a.target='_blank';a.rel='noopener';el.append(a);}for(const item of data.images||[]){const a=document.createElement('a');a.href=localURL(item.url);a.target='_blank';a.rel='noopener';const image=document.createElement('img');image.className='result-image';image.src=a.href;image.alt='Retrieved illustration — '+item.citation;image.addEventListener('load',scroll);a.append(image);el.append(a);const label=document.createElement('span');label.className='image-label';label.textContent=item.citation;el.append(label);}scroll();}
async function restoreHistory(){if(!conversationId)return;setBusy(true);try{const data=await request('/api/conversations/'+encodeURIComponent(conversationId));if(data.messages.length)messages.replaceChildren();for(const m of data.messages){if(m.role==='user')bubble(m.content,'user');else if(m.status==='completed')renderAnswer({answer:m.content,sources:m.sources,images:m.images});else bubble(m.content||'This answer did not finish. Please ask again.','error');}}catch(error){bubble('Could not load saved chat: '+error.message+' Refresh to retry or start a new chat.','error');}finally{setBusy(false);}}
async function ask(text,type){if(busy||!text.trim())return;openChat();setBusy(true);const waiting=document.createElement('div');waiting.className='loading';try{if(!conversationId){const c=await request('/api/conversations',{method:'POST'});remember(c.id);}messages.querySelector('.welcome')?.remove();bubble(text,'user');question.value='';waiting.textContent='Finding evidence and preparing your answer…';messages.append(waiting);scroll();const data=await request('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text,content_type:type||null,conversation_id:conversationId})});waiting.remove();remember(data.conversation_id);renderAnswer(data);}catch(error){waiting.remove();bubble(error.message||'Unable to connect. Please try again.','error');}finally{setBusy(false);question.focus();}}
launcher.addEventListener('click',()=>panel.hidden?openChat():closeChat());
document.getElementById('close-chat').addEventListener('click',closeChat);
document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',openChat));
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!panel.hidden)closeChat();});
document.getElementById('chat-form').addEventListener('submit',e=>{e.preventDefault();ask(question.value.trim(),mode.value);});
question.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask(question.value.trim(),mode.value);}});
document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>{if(!busy){mode.value=b.dataset.type||'';ask(b.dataset.question,mode.value);}}));
newChat.addEventListener('click',()=>{if(!busy){remember(null);messages.replaceChildren();bubble('Ready for a new conversation. Previous chats remain saved.','assistant');}});
request('/api/document').then(d=>{document.getElementById('document-name').textContent=d.filename;document.getElementById('document-meta').textContent=d.pages+' pages · Ready to explore';}).catch(()=>{document.getElementById('document-meta').textContent='Document unavailable — check server setup';document.querySelector('.connected').textContent='○ Unavailable';});
restoreHistory();
