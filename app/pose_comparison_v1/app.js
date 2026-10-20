const $ = id => document.getElementById(id);
const LIBS = ['mediapipe','openpose','mmpose','deepercut'];
const NAMES = {mediapipe:'MediaPipe',openpose:'OpenPose',mmpose:'MMPose',deepercut:'DeeperCut'};
const COLORS = {mediapipe:'#267d78',openpose:'#4776a2',mmpose:'#a17b36',deepercut:'#975961'};
const COMMON_EDGES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const STATUS = {queued:'Sırada',running:'İşleniyor',completed:'Tamamlandı',partial:'Kısmen tamamlandı',error:'İşlem hatası',cancelled:'Durduruldu',interrupted:'Yarıda kaldı',POSE_RETURNED:'Koordinatlar döndü',NO_POSE:'Pose bulunamadı',ERROR:'İşlem hatası',CANCELLED:'Durduruldu'};
const RESUMABLE = new Set(['interrupted','cancelled','error','partial']);
const WORKSPACE_KEY = 'poz-analizi.workspace.v1';
const BASE_NOTE = 'Nokta sayısı, doğruluk oranı değildir.';
const state = {token:'',videos:[],selected:new Set(),focus:null,result:null,resultId:null,index:0,tab:'frame',jobs:[],schemes:{},playing:false,timer:null,imageCache:new Map(),drawVersion:0,loadVersion:0,followJobs:false,queuePending:false,cancelPending:false,pollPending:false,batch:null,batchDirty:true,batchPending:false,batchError:'',resumePending:new Set(),inspection:null,inspectionVersion:0,zoom:1,persistenceReady:false};
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format = (n,digits=1) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toLocaleString('tr-TR',{maximumFractionDigits:digits});
const pct = n => n == null ? '—' : `%${format(n)}`;
const time = seconds => seconds == null ? '—' : `${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(2).padStart(5,'0')}`;
const libCell = lib => `<span class="library-cell"><i style="background:${COLORS[lib]||'#75766f'}"></i>${esc(NAMES[lib]||lib)}</span>`;
const isFailed = model => model && ['ERROR','CANCELLED'].includes(model.detection_status);
const activeJob = videoId => [...state.jobs].reverse().find(j=>j.video_id===videoId && ['running','queued'].includes(j.status));

async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Pose-Token':state.token},body:JSON.stringify(body)};
  const response=await fetch(path,options),data=await response.json();
  if(!response.ok)throw new Error(data.error||'İstek tamamlanamadı.');return data;
}
function toast(text){$('toast').textContent=text;$('toast').classList.add('visible');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('visible'),5500);}
function stopPlay(){clearTimeout(state.timer);state.playing=false;$('play-button').textContent='Oynat';$('play-button').setAttribute('aria-label','Örneklenmiş kareleri oynat');}
function readWorkspace(){
  try{const data=JSON.parse(localStorage.getItem(WORKSPACE_KEY));return data?.version===1?data:null;}catch{return null;}
}
function saveWorkspace(){
  if(!state.persistenceReady)return;
  try{localStorage.setItem(WORKSPACE_KEY,JSON.stringify({version:1,selected:[...state.selected],focus:state.focus,resultId:state.resultId,index:state.index,tab:state.tab,folder:$('folder-input').value,frameStep:Number($('sample-step').value),sampleLimit:Number($('sample-limit').value),pointsMode:$('points-mode').value,pointsViewVersion:2,showPoints:$('show-points').checked,showIds:$('show-ids').checked,showLowScore:$('show-low-score').checked,libraries:[...document.querySelectorAll('input[name="library"]:checked')].map(c=>c.value)}));}catch{/* Storage may be disabled; the application remains usable. */}
}
function restoreWorkspace(saved){
  const videoIds=new Set(state.videos.map(v=>v.video_id)),latest=items=>[...items].reverse().find(j=>j.result_path&&videoIds.has(j.video_id));
  let resultId=null;
  if(saved){
    if(typeof saved.folder==='string')$('folder-input').value=saved.folder;
    for(const [key,id,min] of [['frameStep','sample-step',1],['sampleLimit','sample-limit',0]])if(Number.isSafeInteger(saved[key])&&saved[key]>=min)$(id).value=saved[key];
    $('points-mode').value=saved.pointsViewVersion===2&&['common','native'].includes(saved.pointsMode)?saved.pointsMode:'native';
    for(const [key,id] of [['showPoints','show-points'],['showIds','show-ids'],['showLowScore','show-low-score']])if(typeof saved[key]==='boolean')$(id).checked=saved[key];
    if(Array.isArray(saved.libraries))for(const checkbox of document.querySelectorAll('input[name="library"]'))checkbox.checked=!checkbox.disabled&&saved.libraries.includes(checkbox.value);
    if(['frame','summary','batch','jobs'].includes(saved.tab))state.tab=saved.tab;
    if(saved.resultId==='pilot'&&saved.focus==null){resultId='pilot';state.focus=null;}
    else if(videoIds.has(saved.focus)){
      state.focus=saved.focus;
      const stored=state.jobs.find(j=>j.id===saved.resultId&&j.video_id===saved.focus&&j.result_path);
      resultId=stored?.id||(saved.resultId==null?null:latest(state.jobs.filter(j=>j.video_id===saved.focus))?.id)||null;
    }else{
      const stored=state.jobs.find(j=>j.id===saved.resultId&&j.result_path&&videoIds.has(j.video_id)),fallback=stored||latest(state.jobs);
      state.focus=fallback?.video_id||state.videos[0]?.video_id||null;resultId=fallback?.id||null;
    }
  }else{const fallback=latest(state.jobs);state.focus=fallback?.video_id||state.videos[0]?.video_id||null;resultId=fallback?.id||null;}
  if(Array.isArray(saved?.selected))state.selected=new Set(saved.selected.filter(id=>videoIds.has(id)));
  else if(state.focus)state.selected.add(state.focus);
  state.persistenceReady=true;return resultId;
}
async function loadResult(id){
  stopPlay();const version=++state.loadVersion,result=await api(`/api/results/${encodeURIComponent(id)}`);
  if(version!==state.loadVersion)return false;state.result=result;state.resultId=id;state.index=0;renderView();renderTable();renderDownloads();return true;
}
function renderVideos(){
  saveWorkspace();
  const container=$('video-list'),scroll=container.scrollTop;container.replaceChildren();
  const history=document.createElement('button');history.className='history-item';history.textContent='Önceki 10 resimlik çalışma';
  history.addEventListener('click',async()=>{try{state.focus=null;state.followJobs=false;await loadResult('pilot');renderVideos();}catch(error){toast(error.message);}});container.append(history);
  for(const video of state.videos){
    const job=activeJob(video.video_id)||[...state.jobs].reverse().find(j=>j.video_id===video.video_id),row=document.createElement('div');
    row.className=`video-row ${state.focus===video.video_id?'active':''}`;row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label',`${video.name} videosunu aç`);
    row.innerHTML=`<label class="queue-check"><input type="checkbox" aria-label="${esc(video.name)} kuyruğa seç" ${state.selected.has(video.video_id)?'checked':''}></label><img class="video-thumb" src="${esc(video.thumbnail_url)}" alt="" loading="lazy"><div class="video-row-details"><strong title="${esc(video.name)}">${esc(video.name)}</strong><span>${time(video.duration)} · ${format(video.fps,2)} FPS</span><small class="file-status ${esc(job?.status||'')}">${esc(job?STATUS[job.status]||job.status:'İşlenmedi')}</small></div>`;
    row.querySelector('label').addEventListener('click',e=>e.stopPropagation());
    row.querySelector('input').addEventListener('change',e=>{if(e.target.checked)state.selected.add(video.video_id);else state.selected.delete(video.video_id);renderQueueCount();updateSamplingHint();});
    const choose=async()=>{try{state.focus=video.video_id;state.followJobs=false;state.loadVersion++;stopPlay();const completed=[...state.jobs].reverse().find(j=>j.video_id===video.video_id&&j.result_path);if(completed)await loadResult(completed.id);else{state.result=null;state.resultId=null;state.index=0;renderView();renderTable();renderDownloads();}renderVideos();updateSamplingHint();}catch(error){toast(error.message);}};
    row.addEventListener('click',choose);row.addEventListener('keydown',e=>{if(e.target===row&&['Enter',' '].includes(e.key)){e.preventDefault();choose();}});container.append(row);
  }
  container.scrollTop=scroll;renderQueueCount();
}
function renderQueueCount(){
  $('queue-count').textContent=`${state.videos.length} video · ${state.selected.size} seçili`;
  const eligible=[...state.selected].filter(id=>!activeJob(id)),running=state.jobs.some(j=>['running','queued'].includes(j.status));
  $('run-button').disabled=state.queuePending||!eligible.length;$('run-button').textContent=state.queuePending?'Ekleniyor…':running?'Kuyruğa ekle':'Analizi başlat';
  $('run-button').title=!eligible.length&&state.selected.size?'Seçili videolar zaten işlem kuyruğunda.':'';$('cancel-button').disabled=state.cancelPending||!running;
  const all=$('select-all-videos');all.checked=state.videos.length>0&&state.selected.size===state.videos.length;all.indeterminate=state.selected.size>0&&state.selected.size<state.videos.length;all.disabled=!state.videos.length;
}
function updateSamplingHint(){const video=state.videos.find(v=>v.video_id===state.focus)||state.videos.find(v=>state.selected.has(v.video_id)),step=Number($('sample-step').value);$('sample-preview').textContent=video&&step>0?`${format(video.fps,2)} FPS kaynakta yaklaşık ${format(video.fps/step,2)} örnek/sn. İlk örnek 0. karedir.`:'Kare aralığı kaynak videonun FPS değeri değildir.';saveWorkspace();}
function pointStats(lib,model){
  if(!model)return {nativeIn:null,commonIn:null,outside:null,commonOutside:null,commonLow:null};
  const scheme=state.schemes[lib],video=state.result?.video||state.videos.find(v=>v.video_id===state.focus)||{},valid=new Map((model.points||[]).filter(p=>p.coordinate_available===true&&Number.isFinite(p.x_px)&&Number.isFinite(p.y_px)&&Number.isInteger(p.native_id)&&p.native_id>=0&&p.native_id<scheme.expected).map(p=>[p.native_id,p]));
  const common=new Set(scheme.common_ids),outside=[...valid.values()].filter(p=>p.x_px<0||p.y_px<0||p.x_px>=video.width||p.y_px>=video.height),outsideIds=new Set(model.out_of_image_native_ids||outside.map(p=>p.native_id)),commonOut=[...outsideIds].filter(id=>common.has(id)).length;
  return {nativeIn:model.native_in_image??model.in_image_available??((model.native_available??valid.size)-outsideIds.size),commonIn:model.common_in_image??model.common_in_image_available??((model.common_available??[...valid.keys()].filter(id=>common.has(id)).length)-commonOut),outside:model.native_out_of_image??model.out_of_image_count??outsideIds.size,commonOutside:model.common_out_of_image??commonOut,commonLow:model.low_score_count==null?null:(model.low_score_native_ids||[]).filter(id=>common.has(id)).length};
}
function modelNotes(lib,model){
  if(!model)return ['Analiz bekleniyor'];if(isFailed(model))return [STATUS[model.detection_status]||model.detection_status];
  const counts=pointStats(lib,model),common=$('points-mode').value==='common',outside=common?counts.commonOutside:counts.outside,low=common?counts.commonLow:model.low_score_count,notes=[];if(outside)notes.push(`${outside} görüntü dışında`);if(model.candidate_count>1)notes.push(`${model.candidate_count} ${lib==='deepercut'?'gövde kümesi':'kişi adayı'}`);if(low)notes.push(`${low} düşük skor`);return notes;
}
function modelPanel(lib,model){
  const common=$('points-mode').value==='common',expected=common?12:state.schemes[lib].expected,available=model?.[common?'common_available':'native_available'],failed=isFailed(model),stats=pointStats(lib,model),notes=modelNotes(lib,model),card=document.createElement('article');card.className=`model-panel ${failed?'has-error':''}`;
  const count=model&&!failed?`${format(available,0)} <span>/ ${expected}</span>`:'—',review=failed||(common?stats.commonOutside:stats.outside)>0||model?.candidate_count>1||(common?stats.commonLow:model?.low_score_count)>0;
  card.innerHTML=`<header class="model-panel-header"><span class="model-name"><i style="background:${COLORS[lib]}"></i>${NAMES[lib]}</span><button class="expand-model icon-button" type="button" aria-label="${NAMES[lib]} görüntüsünü büyüt" title="Görüntüyü büyüt"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4"/></svg></button></header><div class="canvas-holder"><canvas aria-label="${NAMES[lib]} nokta çizimi"></canvas>${failed?`<div class="canvas-message"><strong>${esc(notes[0])}</strong><span>Bu işlem bulunma oranına katılmadı.</span></div>`:''}</div><footer class="model-panel-footer"><div><small>${common?'Ortak 12 · dönen / ':'Dönen / '}beklenen</small><strong class="count">${count}</strong></div><span class="panel-note ${review?'needs-review':''}">${esc(notes.join(' · '))}${review?' · Kontrol':''}</span></footer>`;
  card.querySelector('.expand-model').addEventListener('click',()=>openInspection(lib));return card;
}
async function getImage(url){
  if(state.imageCache.has(url)){const cached=state.imageCache.get(url);state.imageCache.delete(url);state.imageCache.set(url,cached);return cached;}
  const promise=new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Görüntü yüklenemedi.'));image.src=url;});state.imageCache.set(url,promise);
  while(state.imageCache.size>8)state.imageCache.delete(state.imageCache.keys().next().value);
  try{return await promise;}catch(error){if(state.imageCache.get(url)===promise)state.imageCache.delete(url);throw error;}
}
function drawModel(canvas,image,lib,model,{raw=false,large=false}={}){
  const width=image.naturalWidth,height=image.naturalHeight,max=large?2560:1280,ratio=Math.min(1,max/Math.max(width,height));canvas.width=Math.max(1,Math.round(width*ratio));canvas.height=Math.max(1,Math.round(height*ratio));
  const ctx=canvas.getContext('2d');ctx.scale(canvas.width/width,canvas.height/height);ctx.drawImage(image,0,0,width,height);if(raw||(!large&&!$('show-points').checked)||!model||isFailed(model))return;
  const scheme=state.schemes[lib],mode=$('points-mode').value,valid=new Map((model.points||[]).filter(p=>p.coordinate_available===true&&Number.isFinite(p.x_px)&&Number.isFinite(p.y_px)&&Number.isInteger(p.native_id)&&p.native_id>=0&&p.native_id<scheme.expected&&p.x_px>=0&&p.y_px>=0&&p.x_px<width&&p.y_px<height&&(mode!=='common'||scheme.common_ids.includes(p.native_id))).map(p=>[p.native_id,p]));
  const edges=mode==='common'?COMMON_EDGES.map(([a,b])=>[scheme.common_map[a],scheme.common_map[b]]):scheme.edges,scale=Math.max(.05,Math.min(canvas.clientWidth/width,canvas.clientHeight/height)),unit=1/scale;ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=1.25*unit;ctx.strokeStyle=COLORS[lib];
  for(const [a,b] of edges){if(valid.has(a)&&valid.has(b)){ctx.beginPath();ctx.moveTo(valid.get(a).x_px,valid.get(a).y_px);ctx.lineTo(valid.get(b).x_px,valid.get(b).y_px);ctx.stroke();}}
  for(const [id,p] of valid){const low=$('show-low-score').checked&&Number.isFinite(p.score)&&scheme.low_score_level!=null&&p.score<scheme.low_score_level;ctx.fillStyle=low?'#e8a23c':COLORS[lib];ctx.strokeStyle='#fff';ctx.lineWidth=.7*unit;ctx.beginPath();ctx.arc(p.x_px,p.y_px,2.5*unit,0,2*Math.PI);ctx.fill();ctx.stroke();if($('show-ids').checked){ctx.font=`${8.5*unit}px -apple-system,BlinkMacSystemFont,sans-serif`;ctx.lineWidth=2.2*unit;ctx.strokeStyle='rgba(255,255,255,.88)';ctx.fillStyle='#262a2c';ctx.strokeText(id,p.x_px+3.5*unit,p.y_px-3*unit);ctx.fillText(id,p.x_px+3.5*unit,p.y_px-3*unit);}}
}
function notices(result,job,sample){
  const messages=[],add=value=>{if(value)messages.push(typeof value==='string'?value:JSON.stringify(value));};add(job?.error);add(result?.render_error);add(result?.sampling_warning);
  if(result?.status&&['PARTIAL','ERROR','CANCELLED'].includes(result.status))add(`Sonuç durumu: ${STATUS[result.status.toLowerCase()]||result.status}. Eksik ve hatalı işlemler sonuç tablolarında ayrıca gösterilir.`);
  for(const lib of LIBS){const m=sample?.models[lib];if(m?.error)add(`${NAMES[lib]}: ${m.error}`);}
  const box=$('analysis-notices');box.replaceChildren();box.hidden=!messages.length;for(const message of new Set(messages)){const p=document.createElement('p');p.textContent=message;box.append(p);}
}
async function renderView(){
  saveWorkspace();
  const version=++state.drawVersion,result=state.result,sample=result?.samples[state.index],focused=state.videos.find(v=>v.video_id===state.focus),pending=activeJob(state.focus),job=state.jobs.find(j=>j.id===state.resultId)||[...state.jobs].reverse().find(j=>j.video_id===state.focus);
  const latestJob=[...state.jobs].reverse().find(j=>j.video_id===state.focus);
  const previousResult=Boolean(result&&!result.historical&&latestJob&&state.resultId!==latestJob.id);
  $('active-title').textContent=result?.historical?'Önceki 10 resimlik çalışma':result?.video.name||focused?.name||'Video karşılaştırması';$('active-subtitle').textContent=result?.historical?'Resim pilotu · dört kütüphanenin kayıtlı çıktıları':focused?`${focused.width} × ${focused.height} · ${format(focused.fps,2)} FPS · ${time(focused.duration)}`:'Aynı kare, dört farklı pose modeli';
  $('view-label').textContent=previousResult?'Önceki sonuç':pending?STATUS[pending.status]:result?.historical?'Önceki çalışma':result?'Analiz sonucu':job?STATUS[job.status]||job.status:'İşlenmedi';
  $('source-note').textContent=result?.historical?'Bu görüntüler önceki 10 resimlik çalışmadan. Yeni bir video analizi değildir.':result?`${previousResult?'Önceki sonuç: ':''}${result.samples.length} kare · ${result.frame_step} karede bir${result.limited?' · sınırlı deneme':''}${$('points-mode').value==='common'?' · ortak 12 eklem':''}.${previousResult?` Son işlem: ${latestJob.frame_step} karede bir · ${(STATUS[latestJob.status]||latestJob.status).toLocaleLowerCase('tr-TR')}.`:''}`:pending?`${STATUS[pending.status]}. Sonuçlar işlem tamamlandığında açılacak; ilerleme alt çubukta gösteriliyor.`:job&&RESUMABLE.has(job.status)?'İşlem tamamlanmadı. Ayrıntıyı işlem geçmişinde görebilir ve uygun işlemi sürdürebilirsiniz.':'Video henüz işlenmedi. Kütüphaneleri ve kare aralığını seçip analizi başlatın.';
  notices(result,job,sample);const grid=$('model-grid');grid.querySelectorAll('canvas').forEach(c=>{c.width=1;c.height=1;});grid.replaceChildren();let image;const url=sample?.image_url||focused?.thumbnail_url;try{if(url)image=await getImage(url);}catch(error){toast(error.message);}if(version!==state.drawVersion)return;
  for(const lib of LIBS){const model=sample?.models[lib],card=modelPanel(lib,model);grid.append(card);if(image)drawModel(card.querySelector('canvas'),image,lib,model);else card.querySelector('.expand-model').disabled=true;if(!model&&result&&!result.historical)card.querySelector('.panel-note').textContent='Bu çalışmada seçilmedi';}
  const count=result?.samples.length||0;$('frame-slider').max=Math.max(0,count-1);$('frame-slider').value=state.index;$('frame-slider').disabled=count<2;$('frame-position').textContent=result?.historical?`Görsel ${state.index+1} / ${count}`:sample?`Kare ${sample.frame_index} · Örnek ${state.index+1} / ${count}`:'Kare —';$('time-position').textContent=result?.historical?'Video zamanı yok':sample?`${time(sample.timestamp_sec)} / ${time(result.video.duration)}`:'00:00';$('previous-frame').disabled=state.index===0||count<2;$('next-frame').disabled=state.index>=count-1||count<2;$('play-button').disabled=count<2;if($('model-dialog').open)renderInspection();
}
async function openInspection(lib){stopPlay();state.inspection=lib;state.zoom=1;$('inspect-original').checked=false;$('model-dialog').showModal();await renderInspection();}
async function renderInspection(){
  if(!$('model-dialog').open||!state.inspection)return;const version=++state.inspectionVersion,lib=state.inspection,sample=state.result?.samples[state.index],model=sample?.models[lib],focused=state.videos.find(v=>v.video_id===state.focus),url=sample?.image_url||focused?.thumbnail_url;if(!url)return;
  try{const image=await getImage(url);if(version!==state.inspectionVersion||!$('model-dialog').open)return;$('model-dialog-title').textContent=`${NAMES[lib]} · ${$('points-mode').value==='common'?'ortak 12 eklem':'kendi noktaları'}`;$('model-dialog-detail').textContent=`${sample?state.result.historical?`Görsel ${state.index+1}`:`Kare ${sample.frame_index} · ${time(sample.timestamp_sec)}`:'Kaynak önizlemesi'} · ${image.naturalWidth} × ${image.naturalHeight}`;$('inspection-warning').textContent=model?modelNotes(lib,model).join(' · ')||BASE_NOTE:'Bu kare için model sonucu henüz yok.';const viewport=$('inspection-viewport'),fit=Math.min((viewport.clientWidth-32)/image.naturalWidth,(viewport.clientHeight-32)/image.naturalHeight),stage=$('inspection-stage');stage.style.width=`${Math.max(1,Math.round(image.naturalWidth*fit*state.zoom))}px`;stage.style.height=`${Math.max(1,Math.round(image.naturalHeight*fit*state.zoom))}px`;drawModel($('inspection-canvas'),image,lib,model,{raw:$('inspect-original').checked,large:true});$('zoom-value').textContent=`${Math.round(state.zoom*100)}%`;$('zoom-out').disabled=state.zoom<=1;$('zoom-in').disabled=state.zoom>=6;}catch(error){toast(error.message);}
}
function table(headers,rows){$('table-head').innerHTML=`<tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr>`;$('table-body').innerHTML=rows.map(row=>`<tr>${row.map((v,i)=>`<td${i===0?' class="first-cell"':''}>${v}</td>`).join('')}</tr>`).join('')||`<tr><td colspan="${headers.length}" class="table-empty">Henüz sonuç bulunmuyor.</td></tr>`;}
function summaryCells(row){
  const prefix=$('points-mode').value==='common'?'common':'native';
  const returned=row[prefix+'_available_total'],expected=row[prefix+'_expected_total'];
  return [libCell(row.library),row.processed_frames?`${format(returned,0)} / ${format(expected,0)} <small>(${pct(row[prefix+'_coverage_pct'])})</small>`:'—',`${format(row.processed_frames,0)} / ${format(row.sampled_frames,0)}`,`${format(row.error_frames,0)} / ${format(row.cancelled_frames??0,0)}`];
}
function renderTable(){
  saveWorkspace();
  document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===state.tab);b.setAttribute('aria-selected',String(b.dataset.tab===state.tab));b.tabIndex=b.dataset.tab===state.tab?0:-1;});$('measurement-note').textContent=(state.tab==='batch'?'Farklı analiz ayarları ayrı tutulur. ':'')+BASE_NOTE;const result=state.result,sample=result?.samples[state.index];
  if(state.tab==='jobs'){
    table(['Video','Durum','Örnekleme','Kütüphaneler','Sonuç / devam','Açıklama'],state.jobs.map(job=>[esc(job.name),`<span class="job-status ${esc(job.status)}">${esc(STATUS[job.status]||job.status)}</span>`,`Her ${format(job.frame_step,0)}. kare${job.sample_limit?` · ilk ${format(job.sample_limit,0)}`:' · tüm video'}`,(job.libraries||[]).map(lib=>NAMES[lib]||lib).map(esc).join(', '),`${job.result_path?`<button class="table-link" data-result="${esc(job.id)}">Sonucu aç</button>`:''}${RESUMABLE.has(job.status)?` <button class="table-link resume-link" data-resume="${esc(job.id)}" ${state.resumePending.has(job.id)?'disabled':''}>${state.resumePending.has(job.id)?'Hazırlanıyor…':'Devam et / yeniden dene'}</button>`:''}${!job.result_path&&!RESUMABLE.has(job.status)?'—':''}`,`<span class="job-error" title="${esc(job.error||'')}">${esc(job.error||'—')}</span>`]));
    document.querySelectorAll('[data-result]').forEach(b=>b.addEventListener('click',async()=>{try{const job=state.jobs.find(j=>j.id===b.dataset.result);state.focus=job.video_id;state.followJobs=false;await loadResult(job.id);state.tab='frame';renderTable();renderVideos();}catch(error){toast(error.message);}}));document.querySelectorAll('[data-resume]').forEach(b=>b.addEventListener('click',()=>resumeJob(b.dataset.resume)));return;
  }
  if(state.tab==='batch'){
    if(!state.batch){table(['Toplu sonuç'],[[`<span class="${state.batchError?'needs-review':''}">${esc(state.batchError||'Toplu sonuçlar yükleniyor…')}</span>`]]);return;}
    table(['Kütüphane','Örnekleme','Video','Dönen / beklenen','İşlenen kare','Hata / durdu'],state.batch.rows.map(row=>{
      const cells=summaryCells(row);
      return [cells[0],esc(row.protocol_label||`Her ${row.frame_step}. kare`),`${format(row.videos,0)}${row.failed_videos||row.partial_videos?` <small class="needs-review">(${format(row.failed_videos||0,0)} hata · ${format(row.partial_videos||0,0)} eksik)</small>`:''}`,cells[1],cells[2],cells[3]];
    }));return;
  }
  if(state.tab==='summary'){
    table(['Kütüphane','Dönen / beklenen','İşlenen kare','Hata / durdu'],(result?.summary||[]).map(summaryCells));return;
  }
  table(['Kütüphane','Dönen / beklenen','Görüntü dışı','Düşük skor','Durum'],LIBS.map(lib=>{
    const m=sample?.models[lib],okay=m&&['POSE_RETURNED','NO_POSE'].includes(m.detection_status),stats=pointStats(lib,m);
    const common=$('points-mode').value==='common',available=m?.[common?'common_available':'native_available'],expected=common?12:state.schemes[lib].expected;
    const outside=common?stats.commonOutside:stats.outside,low=common?stats.commonLow:m?.low_score_count;
    return [libCell(lib),okay?`${format(available,0)} / ${expected}`:'—',okay?format(outside,0):'—',okay?format(low,0):'—',`<span class="${m&&(!okay||outside||m.candidate_count>1)?'needs-review':''}" title="${esc(m?.error||'')}">${esc(m?STATUS[m.detection_status]||m.detection_status:'İşlenmedi')}</span>`];
  }));
}
function renderDownloads(){
  const batch=state.tab==='batch';
  const downloads=state.tab==='jobs'?{}:batch?state.batch?.downloads||{}:state.result?.downloads||{};
  const csvKey=batch||state.tab==='summary'?'summary_csv':'frame_counts_csv';
  for(const [id,key] of [['download-csv',csvKey],['download-video','comparison_video']]){
    const button=$(id),url=downloads[key];
    button.href=url||'#';button.setAttribute('aria-disabled',String(!url));button.classList.toggle('disabled',!url);
    button.onclick=e=>{if(!url){e.preventDefault();toast('Bu görünüm için indirilebilir analiz dosyası yok.');}};
  }
  $('download-video').title='Her modelin kendi noktalarını içeren video';
  $('download-csv').title=batch?'Toplu sonuç CSV dosyasını indir':state.tab==='summary'?'Video özeti CSV dosyasını indir':'Kare sayımları CSV dosyasını indir';
}
async function loadBatch(){if(state.batchPending||(!state.batchDirty&&state.batch))return;state.batchPending=true;state.batchError='';try{state.batch=await api('/api/batch-summary');state.batchDirty=false;}catch(error){state.batchError=error.message;toast(`Toplu sonuç: ${error.message}`);}finally{state.batchPending=false;if(state.tab==='batch'){renderTable();renderDownloads();}}}
function renderProgress(){const active=state.jobs.find(j=>j.status==='running'),queued=state.jobs.filter(j=>j.status==='queued').length;if(!active){$('job-progress').innerHTML=queued?`<span>${queued} video sırada</span>`:'<span class="idle-dot"></span><span>Çalışmaya hazır</span>';return;}const p=active.progress||{},fraction=p.total?Math.min(100,100*p.completed/p.total):0,stage=p.library?`${NAMES[p.library]} · ${p.completed||0}/${p.total||'—'} kare`:p.stage==='sampling'?'Kareler hazırlanıyor':p.message||'Başlatılıyor';$('job-progress').innerHTML=`<div class="progress-caption"><span>${esc(active.name)}</span><strong>${esc(stage)}</strong></div><div class="progress-track"><i style="width:${fraction}%"></i></div><small>${queued} video sırada</small>`;}
async function poll(){
  if(state.pollPending)return;state.pollPending=true;
  try{const jobs=await api('/api/jobs'),previous=new Map(state.jobs.map(j=>[j.id,j.status])),signature=items=>JSON.stringify(items.map(j=>[j.id,j.status,j.result_path,j.error])),changed=signature(jobs)!==signature(state.jobs);state.jobs=jobs;renderProgress();renderQueueCount();let loaded=false;for(const job of jobs){if(state.followJobs&&job.result_path&&previous.get(job.id)!==job.status&&['completed','partial'].includes(job.status)){state.focus=job.video_id;loaded=await loadResult(job.id);toast(`${job.name}: ${STATUS[job.status].toLocaleLowerCase('tr-TR')}.`);}}if(changed){renderVideos();if(!loaded)renderView();state.batchDirty=true;if(state.tab==='batch')await loadBatch();if(state.tab==='jobs'||state.tab==='batch')renderTable();}}catch(error){console.warn('Yerel durum güncellenemedi:',error.message);}finally{state.pollPending=false;}
}
async function startQueue(){
  if(state.queuePending)return;state.queuePending=true;renderQueueCount();
  try{const libraries=[...document.querySelectorAll('input[name="library"]:checked')].map(c=>c.value),ids=[...state.selected].filter(id=>!activeJob(id)),step=Number($('sample-step').value),limit=Number($('sample-limit').value);if(!libraries.length)throw new Error('En az bir kütüphane seçin.');if(!ids.length)throw new Error('Seçili videolar zaten kuyrukta.');if(!Number.isInteger(step)||step<1)throw new Error('Kare aralığı 1 veya daha büyük tam sayı olmalı.');if(!Number.isInteger(limit)||limit<0)throw new Error('En fazla örnek 0 veya daha büyük tam sayı olmalı.');const data=await api('/api/queue',{video_ids:ids,libraries,frame_step:step,sample_limit:limit});state.jobs=data.jobs;state.followJobs=true;state.focus=ids[0];state.result=null;state.resultId=null;state.loadVersion++;stopPlay();state.batchDirty=true;renderVideos();renderView();renderTable();renderDownloads();renderProgress();if(state.tab==='batch')loadBatch();toast(`${ids.length} video analiz kuyruğuna eklendi.`);}catch(error){toast(error.message);}finally{state.queuePending=false;renderQueueCount();}
}
async function resumeJob(id){
  if(state.resumePending.has(id))return;state.resumePending.add(id);renderTable();
  try{
    const original=state.jobs.find(j=>j.id===id),data=await api('/api/resume',{job_id:id});
    if(data.jobs)state.jobs=data.jobs;state.followJobs=true;state.batchDirty=true;state.focus=original?.video_id||state.focus;
    if(state.resultId!==id){state.loadVersion++;stopPlay();if(original?.result_path)await loadResult(id);else{state.result=null;state.resultId=null;state.index=0;}}
    renderView();renderDownloads();toast(data.message||'İşlem devam kuyruğuna alındı.');await poll();renderVideos();renderProgress();
  }catch(error){toast(error.message);}finally{state.resumePending.delete(id);renderTable();}
}
function updateCapabilities(capabilities){for(const lib of LIBS){const checkbox=document.querySelector(`input[name="library"][value="${lib}"]`),cap=capabilities[lib];if(!cap)continue;const previouslyDisabled=checkbox.disabled;checkbox.disabled=!cap.ready;if(!cap.ready)checkbox.checked=false;else if(previouslyDisabled)checkbox.checked=true;const label=checkbox.closest('label');label.classList.toggle('unavailable',!cap.ready);label.querySelector('.library-state').textContent=cap.ready?'Hazır':cap.message;label.title=cap.message||'Model dosyaları ve çalışma ortamı mevcut; çıkarım sonucu ayrıca kontrol edilir.';}}
async function initialize(){
  try{
    const saved=readWorkspace(),data=await api('/api/bootstrap');state.token=data.token;state.schemes=data.schemes;state.videos=data.videos;state.jobs=data.jobs;
    $('folder-input').value=data.default_folder;updateCapabilities(data.capabilities);const resultId=restoreWorkspace(saved);
    if(resultId){
      try{await loadResult(resultId);if(Number.isSafeInteger(saved?.index)&&saved.index>=0){state.index=Math.min(saved.index,Math.max(0,state.result.samples.length-1));renderView();renderTable();}}
      catch(error){toast(`Önceki görünüm açılamadı: ${error.message}`);state.result=null;state.resultId=null;renderView();renderTable();renderDownloads();}
    }else{renderView();renderTable();renderDownloads();}
    renderVideos();updateSamplingHint();renderProgress();if(state.tab==='batch')loadBatch();setInterval(poll,1800);
  }catch(error){toast(error.message);$('source-note').textContent='Uygulamaya bağlanılamadı. Yerel sunucunun çalıştığını kontrol edin.';}
}
$('scan-button').addEventListener('click',async()=>{try{$('scan-button').disabled=true;const data=await api('/api/scan',{folder:$('folder-input').value});state.videos=data.videos;state.selected=new Set(data.videos.map(v=>v.video_id));state.focus=null;renderVideos();updateSamplingHint();toast(`${data.videos.length} video bulundu.${data.errors.length?` ${data.errors.length} dosya açılamadı.`:''}`);}catch(error){toast(error.message);}finally{$('scan-button').disabled=false;}});
$('folder-input').addEventListener('keydown',e=>{if(e.key==='Enter')$('scan-button').click();});
$('select-all-videos').addEventListener('change',e=>{state.selected=e.target.checked?new Set(state.videos.map(v=>v.video_id)):new Set();renderVideos();updateSamplingHint();});
$('upload-trigger').addEventListener('click',()=>$('upload-input').click());
$('upload-input').addEventListener('change',async e=>{try{$('upload-trigger').disabled=true;for(const file of e.target.files){toast(`${file.name} yerel uygulamaya aktarılıyor…`);const response=await fetch('/api/upload',{method:'POST',headers:{'X-Pose-Token':state.token,'X-File-Name':encodeURIComponent(file.name)},body:file}),data=await response.json();if(!response.ok)throw new Error(data.error);for(const video of data.videos){if(!state.videos.some(v=>v.video_id===video.video_id))state.videos.push(video);state.selected.add(video.video_id);}}renderVideos();updateSamplingHint();toast('Videolar listeye eklendi.');}catch(error){toast(error.message);}finally{e.target.value='';$('upload-trigger').disabled=false;}});
$('run-button').addEventListener('click',startQueue);
$('cancel-button').addEventListener('click',async()=>{if(state.cancelPending)return;state.cancelPending=true;renderQueueCount();try{const data=await api('/api/cancel',{});toast(data.message);await poll();}catch(error){toast(error.message);}finally{state.cancelPending=false;renderQueueCount();}});
$('refresh-capabilities').addEventListener('click',async()=>{const button=$('refresh-capabilities');button.disabled=true;try{const data=await api('/api/capabilities');updateCapabilities(data.capabilities||data);saveWorkspace();toast('Model hazırlığı yeniden kontrol edildi.');}catch(error){toast(error.message);}finally{button.disabled=false;}});
$('sample-step').addEventListener('input',updateSamplingHint);
$('sample-limit').addEventListener('input',saveWorkspace);
document.querySelectorAll('input[name="library"]').forEach(c=>c.addEventListener('change',saveWorkspace));
for(const id of ['points-mode','show-points','show-ids','show-low-score'])$(id).addEventListener('change',()=>{renderView();renderTable();});
for(const [id,delta] of [['previous-frame',-1],['next-frame',1]])$(id).addEventListener('click',()=>{stopPlay();state.index=Math.max(0,Math.min((state.result?.samples.length||1)-1,state.index+delta));renderView();renderTable();});
$('frame-slider').addEventListener('input',e=>{stopPlay();state.index=Number(e.target.value);renderView();renderTable();});
$('play-button').addEventListener('click',()=>{if(state.playing){stopPlay();return;}state.playing=true;$('play-button').textContent='Duraklat';$('play-button').setAttribute('aria-label','Oynatmayı duraklat');const tick=async()=>{if(!state.playing||!state.result?.samples.length)return;state.index=(state.index+1)%state.result.samples.length;await renderView();renderTable();const delay=state.result.historical?900:Math.max(80,1000*state.result.frame_step/state.result.video.fps);if(state.playing)state.timer=setTimeout(tick,delay);};state.timer=setTimeout(tick,300);});
document.querySelectorAll('[data-tab]').forEach(b=>{b.addEventListener('click',()=>{state.tab=b.dataset.tab;renderTable();renderDownloads();if(state.tab==='batch')loadBatch();});b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();e.stopPropagation();const tabs=[...document.querySelectorAll('[data-tab]')],i=tabs.indexOf(b),target=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[target].click();tabs[target].focus();});});
$('references-button').addEventListener('click',()=>$('references-dialog').showModal());$('references-close').addEventListener('click',()=>$('references-dialog').close());
$('model-dialog-close').addEventListener('click',()=>$('model-dialog').close());$('model-dialog').addEventListener('close',()=>{state.inspectionVersion++;state.inspection=null;const canvas=$('inspection-canvas');canvas.width=1;canvas.height=1;});
$('zoom-in').addEventListener('click',()=>{state.zoom=Math.min(6,state.zoom+0.5);renderInspection();});$('zoom-out').addEventListener('click',()=>{state.zoom=Math.max(1,state.zoom-0.5);renderInspection();});$('zoom-reset').addEventListener('click',()=>{state.zoom=1;$('inspection-viewport').scrollTo(0,0);renderInspection();});$('inspect-original').addEventListener('change',renderInspection);
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderView,120);});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||document.querySelector('dialog[open]'))return;if(e.code==='ArrowRight'){$('next-frame').click();e.preventDefault();}if(e.code==='ArrowLeft'){$('previous-frame').click();e.preventDefault();}});
const THEME_KEY='poz-analizi.theme';
function setTheme(theme){
  const dark=theme==='dark';document.documentElement.dataset.theme=dark?'dark':'light';
  const button=$('theme-toggle');button.textContent=dark?'Açık tema':'Koyu tema';
  button.setAttribute('aria-label',dark?'Açık temaya geç':'Koyu temaya geç');button.setAttribute('aria-pressed',String(dark));
  try{localStorage.setItem(THEME_KEY,dark?'dark':'light');}catch{}
}
let initialTheme='light';
try{initialTheme=localStorage.getItem(THEME_KEY)||'light';}catch{}
setTheme(initialTheme);
$('theme-toggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
initialize();
