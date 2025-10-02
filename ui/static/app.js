async function send(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  add('me', q);
  const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query:q})});
  const data = await r.json();
  add('bot', data.answer);
  (data.locations||[]).forEach((loc,i)=>{
    const btn = document.createElement('button');
    btn.className = 'cite-btn';
    btn.innerText = `Xem trích dẫn ${i+1}`;
    btn.onclick = ()=> window.open(`/view?file=${encodeURIComponent(loc.file)}&page=${loc.page}`,'_blank');
    document.getElementById('messages').appendChild(btn);
  })
}
function add(cls, text){
  const d=document.createElement('div'); d.className='msg '+cls; d.textContent=text; document.getElementById('messages').appendChild(d);
}