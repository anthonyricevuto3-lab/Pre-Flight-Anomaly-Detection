// Demo frontend script for Pre-Flight Anomaly Detection
// Update FUNCTION_URL if your function is hosted elsewhere
const FUNCTION_URL = 'https://pre-fligt-anomaly-detection-fxenbdg2ced7hrg8.westus3-01.azurewebsites.net/api/detect_anomalies';

function el(id){return document.getElementById(id)}

async function sendPayload(payload){
  const respEl = el('response');
  respEl.textContent = 'Sending request to Azure Function...';
  try{
    const res = await fetch(FUNCTION_URL, {
      method: 'POST',
      mode: 'cors',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const text = await res.text();
    // try to pretty-print JSON
    try{
      const json = JSON.parse(text);
      respEl.textContent = JSON.stringify(json, null, 2);
    }catch(e){
      respEl.textContent = text;
    }
  }catch(err){
    respEl.textContent = `Error: ${err.message}\n\nTroubleshooting:\n- Check if Azure Function is running\n- Verify CORS headers are set\n- Check browser console for details\n\nFunction URL: ${FUNCTION_URL}`;
    console.error('Fetch error:', err);
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
