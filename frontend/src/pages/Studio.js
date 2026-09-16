import { useState, useRef } from 'react';

const API = window.location.hostname === 'localhost'
  ? 'http://localhost:8002'
  : 'https://eva-ai-1-9wfc.onrender.com';

const TOKEN_KEYS = ['aria_token', 'token', 'access_token', 'authToken'];
const getToken = () => TOKEN_KEYS.map((k) => localStorage.getItem(k)).find(Boolean) || '';

const IMG_STYLES = [
  { id: 'photorealistic', label: 'photorealistic', suffix: ', photorealistic, hyperdetailed, 8K' },
  { id: 'anime',          label: 'anime',          suffix: ', anime style, studio ghibli' },
  { id: 'cinematic',      label: 'cinematic',      suffix: ', cinematic lighting, movie still, anamorphic lens' },
  { id: 'oil_painting',   label: 'oil painting',   suffix: ', oil painting, textured brushstrokes, classical art' },
  { id: 'watercolor',     label: 'watercolor',     suffix: ', watercolor painting, soft edges, pastel tones' },
  { id: 'digital_art',    label: 'digital art',    suffix: ', digital art, concept art, ArtStation' },
  { id: '3d_render',      label: '3D render',      suffix: ', 3D render, octane render, physically based lighting' },
  { id: 'sketch',         label: 'sketch',         suffix: ', pencil sketch, hand drawn, detailed linework' },
];

const IMG_SUGGESTIONS = [
  { icon: '✨', label: 'Enhance quality', prompt: 'ultra high quality, masterpiece, ' },
  { icon: '🌅', label: 'Golden hour', prompt: 'golden hour lighting, warm sunlight, ' },
  { icon: '🌙', label: 'Night scene', prompt: 'night scene, moonlight, dark atmosphere, ' },
  { icon: '🎭', label: 'Dramatic', prompt: 'dramatic lighting, high contrast, cinematic, ' },
  { icon: '🌸', label: 'Soft & dreamy', prompt: 'soft light, dreamy, pastel colors, ethereal, ' },
  { icon: '🏙️', label: 'Cyberpunk city', prompt: 'cyberpunk city, neon lights, futuristic, rain, ' },
  { icon: '🦋', label: 'Fantasy', prompt: 'fantasy world, magical, mythical creatures, enchanted, ' },
  { icon: '📸', label: 'Portrait', prompt: 'professional portrait, studio lighting, sharp focus, ' },
];

const VID_SUGGESTIONS = [
  { icon: '🌊', label: 'Ocean waves', prompt: 'Calm ocean waves at sunset, gentle motion' },
  { icon: '🔥', label: 'Fire effect', prompt: 'Flickering fire flames, dynamic motion, warm glow' },
  { icon: '🌌', label: 'Space', prompt: 'Stars moving through space, nebula, cosmic journey' },
  { icon: '🌿', label: 'Nature', prompt: 'Leaves swaying in wind, peaceful forest, sunlight' },
  { icon: '🏙️', label: 'City timelapse', prompt: 'City streets at night, car lights, urban energy' },
  { icon: '💨', label: 'Clouds', prompt: 'Clouds moving across blue sky, time lapse style' },
];

const downloadFile = (dataUrl, filename) => {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
};

export default function Studio() {
  const [tab, setTab]             = useState('image');
  const [prompt, setPrompt]       = useState('');
  const [style, setStyle]         = useState('photorealistic');
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');
  const [imgResult, setImgResult] = useState(null);
  const [vidResult, setVidResult] = useState(null);
  const [videoMode, setVideoMode] = useState('text');
  const [imgFiles, setImgFiles]   = useState([]);
  const [previews, setPreviews]   = useState([]);
  const [gallery, setGallery]     = useState([]);
  const [, setRefImage]           = useState(null);   // reference image for img gen
  const [refPreview, setRefPreview] = useState(null);
  const [liked, setLiked]         = useState(null);   // true=like, false=dislike
  const fileRef    = useRef();
  const refFileRef = useRef();
  const textareaRef = useRef();

  // ── Generate Image ──────────────────────────────────────
  const generateImage = async (customPrompt) => {
    const p = customPrompt || prompt;
    if (!p.trim()) { setError('Enter a prompt first'); return; }
    setLoading(true); setError(''); setImgResult(null); setLiked(null);
    try {
      const suffix = IMG_STYLES.find(s => s.id === style)?.suffix || '';
      const token = getToken();
      const res = await fetch(`${API}/image/generate-v2`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
          ...(token && token !== 'offline' ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          prompt: p + suffix,
          width: 512,
          height: 512,
          style: '',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
      if (data.image_url) {
        setImgResult(data);
        setGallery(g => [data, ...g]);
      } else {
        throw new Error(data.message || 'No image returned');
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  // ── Generate Video ──────────────────────────────────────
  const generateVideo = async (customPrompt) => {
    const p = customPrompt || prompt;
    setLoading(true); setError(''); setVidResult(null);
    try {
      if (videoMode === 'text') {
        if (!p.trim()) { setError('Enter a prompt'); setLoading(false); return; }
        const res = await fetch(`${API}/video/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true',
          },
          body: JSON.stringify({
            prompt: p,
            negative_prompt: 'blurry, low quality',
            num_frames: 16,
            fps: 8,
            width: 256,
            height: 256,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
        setVidResult(data);
      } else {
        if (!imgFiles.length) { setError('Upload at least 2 images'); setLoading(false); return; }
        const fd = new FormData();
        imgFiles.forEach(f => fd.append('files', f));
        const res = await fetch(`${API}/video/from-images?fps=4&transition=fade`, {
          method: 'POST',
          headers: { 'ngrok-skip-browser-warning': 'true' },
          body: fd,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
        setVidResult(data);
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const handleFiles = (e) => {
    const files = Array.from(e.target.files);
    setImgFiles(files);
    setPreviews(files.map(f => URL.createObjectURL(f)));
  };

  const handleRefImage = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setRefImage(file);
    setRefPreview(URL.createObjectURL(file));
  };

  const handleSuggestion = (s) => {
    if (tab === 'image') {
      setPrompt(prev => s.prompt + prev);
      generateImage(s.prompt + prompt);
    } else {
      setPrompt(s.prompt);
      generateVideo(s.prompt);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      tab === 'image' ? generateImage() : generateVideo();
    }
  };

  const suggestions = tab === 'image' ? IMG_SUGGESTIONS : VID_SUGGESTIONS;
  return (
    <div style={{ minHeight: '100vh', background: '#e3e3f0', color: '#0d0f11', display: 'flex', flexDirection: 'column' }}>

      {/* ── Header ── */}
      <div style={{ padding: '20px 24px 0', maxWidth: 760, margin: '0 auto', width: '100%' }}>
        <div style={{ fontFamily: 'Georgia, serif', fontSize: 22, fontWeight: 700, marginBottom: 2, display: 'flex', alignItems: 'center', gap: 8 }}>
          🎨 AI Studio
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>Generate images and videos with AI</div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
          {[['image', '🖼️', 'Image Generation']].map(([id, icon, label]) => (
            <button key={id}
              onClick={() => { setTab(id); setError(''); setLiked(null); }}
              style={{
                flex: 1, padding: '10px 0', borderRadius: 10,
                border: `1px solid ${tab === id ? 'transparent' : '#1e2d45'}`,
                background: tab === id ? 'linear-gradient(135deg,#6366f1,#8b5cf6)' : 'transparent',
                color: tab === id ? '#fff' : '#64748b',
                fontWeight: 600, fontSize: 14, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              }}>
              {icon} {label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main Content ── */}
      <div style={{ flex: 1, maxWidth: 760, margin: '0 auto', width: '100%', padding: '0 24px' }}>

        {/* ── IMAGE TAB ── */}
        {tab === 'image' && (
          <>
            {/* Style chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, marginBottom: 16 }}>
              {IMG_STYLES.map(s => (
                <button key={s.id}
                  onClick={() => setStyle(s.id)}
                  style={{
                    padding: '5px 13px', borderRadius: 20,
                    border: `1px solid ${style === s.id ? '#6366f1' : '#1e2d45'}`,
                    background: style === s.id ? 'rgba(99,102,241,.2)' : 'transparent',
                    color: style === s.id ? '#a5b4fc' : '#64748b',
                    fontSize: 12, cursor: 'pointer',
                  }}>{s.label}</button>
              ))}
            </div>

            {/* Reference image strip */}
            {refPreview && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, background: '#141c2e', borderRadius: 10, padding: '8px 12px', border: '1px solid #1e2d45' }}>
                <img src={refPreview} alt="ref" style={{ height: 48, width: 48, objectFit: 'cover', borderRadius: 6 }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Reference image attached</div>
                  <div style={{ fontSize: 12, color: '#94a3b8' }}>Used as style/composition reference</div>
                </div>
                <button onClick={() => { setRefImage(null); setRefPreview(null); }}
                  style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 18 }}>×</button>
              </div>
            )}

            {/* Result */}
            {imgResult?.image_url && (
              <div style={{ borderRadius: 16, overflow: 'hidden', border: '1px solid #1e2d45', marginBottom: 16, background: '#d1d7e3' }}>
                <img src={imgResult.image_url} alt={imgResult.prompt} style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain' }} />

                {/* Action bar */}
                <div style={{ padding: '12px 16px', background: '#c8cedc', display: 'flex', alignItems: 'center', gap: 10 }}>
                  {/* Like/Dislike */}
                  <button onClick={() => setLiked(true)}
                    style={{ background: liked === true ? 'rgba(99,241,132,.15)' : 'none', border: `1px solid ${liked === true ? '#4ade80' : '#1e2d45'}`, borderRadius: 8, padding: '6px 12px', color: liked === true ? '#4ade80' : '#64748b', cursor: 'pointer', fontSize: 16 }}>👍</button>
                  <button onClick={() => setLiked(false)}
                    style={{ background: liked === false ? 'rgba(241,99,99,.15)' : 'none', border: `1px solid ${liked === false ? '#f87171' : '#1e2d45'}`, borderRadius: 8, padding: '6px 12px', color: liked === false ? '#f87171' : '#64748b', cursor: 'pointer', fontSize: 16 }}>👎</button>

                  {/* Regenerate */}
                  <button onClick={() => generateImage()}
                    style={{ background: 'none', border: '1px solid #1e2d45', borderRadius: 8, padding: '6px 12px', color: '#94a3b8', cursor: 'pointer', fontSize: 14 }}>🔄</button>

                  <div style={{ flex: 1, fontSize: 11, color: '#475569', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {imgResult.prompt?.slice(0, 80)}
                  </div>

                  {/* Save */}
                  <button onClick={() => downloadFile(imgResult.image_url, imgResult.filename || 'generated.png')}
                    style={{ background: 'none', border: '1px solid #1e2d45', borderRadius: 8, padding: '6px 14px', color: '#94a3b8', cursor: 'pointer', fontSize: 13 }}>↓ Save</button>
                </div>
              </div>
            )}

            {/* Gallery */}
            {gallery.length > 1 && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, color: '#475569', marginBottom: 8 }}>Session gallery · {gallery.length} images</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(100px,1fr))', gap: 6 }}>
                  {gallery.map((g, i) => (
                    <img key={i} src={g.image_url} alt="" onClick={() => { setImgResult(g); setLiked(null); }}
                      style={{ width: '100%', aspectRatio: 1, objectFit: 'cover', borderRadius: 8, cursor: 'pointer', border: '1px solid #1e2d45', opacity: imgResult === g ? 1 : 0.7 }} />
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* ── VIDEO TAB ── */}
        {tab === 'video' && (
          <>
            {/* Mode toggle */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
              {[['text', '📝 Text to Video'], ['images', '🖼️ Image to Video']].map(([m, label]) => (
                <button key={m} onClick={() => setVideoMode(m)}
                  style={{
                    flex: 1, padding: '9px 0', borderRadius: 9,
                    border: `1px solid ${videoMode === m ? '#8b5cf6' : '#1e2d45'}`,
                    background: videoMode === m ? 'rgba(139,92,246,.18)' : 'transparent',
                    color: videoMode === m ? '#c4b5fd' : '#64748b',
                    fontSize: 13, cursor: 'pointer', fontWeight: videoMode === m ? 600 : 400,
                  }}>{label}</button>
              ))}
            </div>

            {/* Image upload for image-to-video */}
            {videoMode === 'images' && (
              <>
                <div onClick={() => fileRef.current.click()}
                  style={{ border: '2px dashed #1e2d45', borderRadius: 10, padding: 20, textAlign: 'center', cursor: 'pointer', marginBottom: 12, background: '#d1d9eb' }}>
                  <div style={{ fontSize: 28, marginBottom: 6 }}>📤</div>
                  <div style={{ fontSize: 13, color: '#64748b' }}>Click to upload images (PNG, JPG, WebP)</div>
                </div>
                <input ref={fileRef} type="file" accept="image/*" multiple style={{ display: 'none' }} onChange={handleFiles} />
                {previews.length > 0 && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                    {previews.map((p, i) => (
                      <img key={i} src={p} alt="" style={{ height: 64, borderRadius: 7, objectFit: 'cover', border: '1px solid #1e2d45' }} />
                    ))}
                  </div>
                )}
              </>
            )}

            {/* Video result */}
            {vidResult && (
              <div style={{ borderRadius: 16, overflow: 'hidden', border: '1px solid #1e2d45', marginBottom: 16, background: '#e4e8ef' }}>
                {vidResult.video_url?.startsWith('data:video') ? (
                  <video src={vidResult.video_url} controls autoPlay loop style={{ width: '100%', display: 'block', maxHeight: 480 }} />
                ) : vidResult.video_url?.startsWith('data:image/gif') ? (
                  <img src={vidResult.video_url} alt="Generated GIF" style={{ width: '100%', display: 'block' }} />
                ) : null}

                <div style={{ padding: '12px 16px', background: '#d5dae6', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <button onClick={() => generateVideo()}
                    style={{ background: 'none', border: '1px solid #1e2d45', borderRadius: 8, padding: '6px 12px', color: '#94a3b8', cursor: 'pointer', fontSize: 14 }}>🔄</button>
                  <div style={{ flex: 1, fontSize: 11, color: '#475569' }}>{vidResult.method || 'AI Generated'}</div>
                  <button onClick={() => downloadFile(vidResult.video_url, vidResult.filename || 'video.mp4')}
                    style={{ background: 'none', border: '1px solid #1e2d45', borderRadius: 8, padding: '6px 14px', color: '#94a3b8', cursor: 'pointer', fontSize: 13 }}>↓ Save</button>
                </div>
              </div>
            )}
          </>
        )}

        {/* Error */}
        {error && (
          <div style={{ background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 10, padding: '10px 14px', fontSize: 13, color: '#f87171', marginBottom: 14, display: 'flex', gap: 8 }}>
            <span>⚠️</span><span>{error}</span>
          </div>
        )}

        {/* Suggestions */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {suggestions.map((s, i) => (
            <button key={i} onClick={() => handleSuggestion(s)}
              disabled={loading}
              style={{
                padding: '7px 14px', borderRadius: 20,
                border: '1px solid #1e2d45',
                background: 'rgba(255,255,255,.03)',
                color: '#94a3b8', fontSize: 12, cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', gap: 5,
                transition: '.15s',
              }}>
              {s.icon} {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Bottom Input Bar (Gemini style) ── */}
      <div style={{ position: 'sticky', bottom: 0, background: '#dde0ea', borderTop: '1px solid #1e2d45', padding: '14px 24px 20px' }}>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          <div style={{ background: '#e1e6f1', border: '1px solid #1e2d45', borderRadius: 16, padding: '10px 14px', display: 'flex', alignItems: 'flex-end', gap: 10 }}>

            {/* Attach reference image (image tab only) */}
            {tab === 'image' && (
              <>
                <button onClick={() => refFileRef.current.click()}
                  title="Attach reference image"
                  style={{ background: 'none', border: 'none', color: refPreview ? '#6366f1' : '#475569', cursor: 'pointer', fontSize: 20, padding: '4px', flexShrink: 0, lineHeight: 1 }}>
                  📎
                </button>
                <input ref={refFileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleRefImage} />
              </>
            )}

            {/* Textarea */}
            <textarea
              ref={textareaRef}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={tab === 'image'
                ? 'Describe the image… (Enter to generate, Shift+Enter for new line)'
                : 'Describe the video… (Enter to generate)'}
              rows={1}
              style={{
                flex: 1, background: 'transparent', border: 'none', outline: 'none',
                color: '#e2e8f0', fontSize: 14, resize: 'none', fontFamily: 'inherit',
                lineHeight: 1.5, maxHeight: 120, overflowY: 'auto',
              }}
            />

            {/* Send button */}
            <button
              onClick={() => tab === 'image' ? generateImage() : generateVideo()}
              disabled={loading}
              style={{
                background: loading ? '#1e2d45' : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
                border: 'none', borderRadius: 10, padding: '8px 16px',
                color: loading ? '#475569' : '#fff',
                fontWeight: 700, fontSize: 14, cursor: loading ? 'not-allowed' : 'pointer',
                flexShrink: 0, display: 'flex', alignItems: 'center', gap: 6,
                transition: '.18s',
              }}>
              {loading
                ? <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>⏳ <span style={{ fontSize: 12 }}>Generating…</span></span>
                : tab === 'image' ? '✨ Generate' : '🎬 Generate'}
            </button>
          </div>

          <div style={{ fontSize: 11, color: '#334155', textAlign: 'center', marginTop: 8 }}>
            {tab === 'image' ? 'Powered by FLUX.1-schnell · 30–60s generation time' : 'Powered by FAL AnimateDiff · 1–3 min generation time'}
          </div>
        </div>
      </div>

    </div>
  );
}
