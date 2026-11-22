"""
EVA: default response language env var (EVA_DEFAULT_LANGUAGE) patch.
बाइट-पातळीवर बदल करतो (PowerShell प्रमाणे encoding बिघडवत नाही). मूळ फाइलची .bak प्रत बनवतो.

चालवा:
    cd D:\\Eva_Ai_brain
    python patch_default_language.py
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\Eva_Ai_brain")
OLD = b'language: str = "en"'
NEW = b'language: str = os.getenv("EVA_DEFAULT_LANGUAGE", "en")'

TARGETS = [
    ROOT / "src" / "app.py",
    ROOT / "eva-ml-service" / "rag_routes_v2.py",
]

for path in TARGETS:
    print("\n==", path)
    if not path.exists():
        print("   फाइल सापडली नाही — वगळले")
        continue
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    print("   BOM:", data[:3] == b"\xef\xbb\xbf", "| CRLF:", b"\r\n" in data)
    print("   दूषित-अक्षर (mojibake) खुणा 'à¤':", text.count("\u00e0\u00a4"),
          "| 'Ã':", text.count("\u00c3"))

    if NEW in data:
        print("   आधीच पॅच केलेले — वगळले")
        continue
    n = data.count(OLD)
    if n != 1:
        print(f"   अपेक्षित 1 जुळणी, सापडल्या {n} — बदल केला नाही")
        continue
    if b"import os" not in data:
        print("   इशारा: 'import os' सापडले नाही. फाइलच्या सुरुवातीला 'import os' जोडा.")
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(data)
        print("   backup:", bak)
    path.write_bytes(data.replace(OLD, NEW))
    print("   पॅच केले")

print("\nपुढे: .env मध्ये EVA_DEFAULT_LANGUAGE=mr टाका आणि app + ML service दोन्ही restart करा.")
