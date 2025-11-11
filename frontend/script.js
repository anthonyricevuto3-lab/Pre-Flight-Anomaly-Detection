// Demo frontend script for Pre-Flight Anomaly Detection
// Update FUNCTION_URL if your function is hosted elsewhere
const FUNCTION_URL = '/api/detect_anomalies';

function el(id){return document.getElementById(id)}

async function sendPayload(payload){
  const respEl = el('response');
  respEl.textContent = 'Sending...';
  try{
    const res = await fetch(FUNCTION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const text = await res.text();
    // try to pretty-print JSON
    try{
      const json = JSON.parse(text);
      respEl.textContent = JSON.stringify(json, null, 2);
    }catch(e){
      respEl.textContent = text;
    }
  }catch(err){
    respEl.textContent = 'Error: '+err.message;
  }
}

el('send').addEventListener('click', ()=>{
  const raw = el('payload').value;
  try{
    const payload = JSON.parse(raw);
    sendPayload(payload);
  }catch(e){
    el('response').textContent = 'Invalid JSON: '+e.message;
  }
});

el('send-rand').addEventListener('click', ()=>{
  // generate small random sample around normal ranges
  const sample = {
    rpm: Math.round(1000 + Math.random()*1200),
    temperature: Math.round(58 + Math.random()*40),
    pressure: Math.round(2200 + Math.random()*2000),
    voltage: +(22 + Math.random()*13).toFixed(2)
  };
  el('payload').value = JSON.stringify(sample, null, 2);
  sendPayload(sample);
});
