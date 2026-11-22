with open('app.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Fix 1 - wrap ml_engine import
old1 = 'from ml_engine import router as ml_router'
new1 = '''try:
    from ml_engine import router as ml_router
    HAS_ML = True
except Exception as e:
    print(f"ml_engine not loaded: {e}")
    HAS_ML = False
    ml_router = None'''

# Fix 2 - wrap image_generator import
old2 = 'from image_generator import router as image_router'
new2 = '''try:
    from image_generator import router as image_router
    HAS_IMAGE = True
except Exception as e:
    print(f"image_generator not loaded: {e}")
    HAS_IMAGE = False
    image_router = None'''

# Fix 3 - wrap video_generator import
old3 = 'from video_generator import router as video_router'
new3 = '''try:
    from video_generator import router as video_router
    HAS_VIDEO = True
except Exception as e:
    print(f"video_generator not loaded: {e}")
    HAS_VIDEO = False
    video_router = None'''

# Fix 4 - wrap include_router calls
old4 = 'app.include_router(ml_router, prefix="/ml")'
new4 = 'if HAS_ML and ml_router: app.include_router(ml_router, prefix="/ml")'

old5 = 'app.include_router(image_router, prefix="/image")'
new5 = 'if HAS_IMAGE and image_router: app.include_router(image_router, prefix="/image")'

old6 = 'app.include_router(video_router, prefix="/video")'
new6 = 'if HAS_VIDEO and video_router: app.include_router(video_router, prefix="/video")'

changes = [
    (old1, new1, 'ml_engine import'),
    (old2, new2, 'image_generator import'),
    (old3, new3, 'video_generator import'),
    (old4, new4, 'ml_router include'),
    (old5, new5, 'image_router include'),
    (old6, new6, 'video_router include'),
]

for old, new, name in changes:
    if old in content:
        content = content.replace(old, new)
        print(f'✅ {name}')
    else:
        print(f'⚠️  {name} — not found (may already be fixed)')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('\nDone! Push to GitHub to redeploy on Render.')
