from pathlib import Path
import hashlib
import sys

app = Path("src/app.py")
backup = Path(r"D:\EVA_SECURITY_STAGE2_20261103_142243\src_app.py.working")

if not app.is_file() or not backup.is_file():
    sys.exit("STOP: source or verified backup missing")

current = app.read_bytes()
saved = backup.read_bytes()

if hashlib.sha256(current).digest() != hashlib.sha256(saved).digest():
    sys.exit("STOP: src/app.py differs from Stage 2 backup; no changes made")

text = current.decode("utf-8")

def replace_once(old, new, label):
    global text
    count = text.count(old)
    if count != 1:
        sys.exit(f"STOP: {label}: expected 1 exact match, found {count}")
    text = text.replace(old, new, 1)
    print("PATCH READY:", label)

replace_once(
'''ADMIN_USERNAME    = os.getenv("ADMIN_USERNAME", "admin@eva.ai")
ADMIN_PASSWORD    = os.getenv("ADMIN_PASSWORD", "")  # MUST be set via environment variable ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â never hardcode

pwd_context  = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# Pre-hash the admin password at startup
ADMIN_PASSWORD_HASH = ADMIN_PASSWORD  # plain compare at startup
''',
'''ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin@eva.ai").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# Prefer a pre-generated bcrypt hash; otherwise hash the environment password once.
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
if ADMIN_PASSWORD_HASH:
    if pwd_context.identify(ADMIN_PASSWORD_HASH) != "bcrypt":
        raise RuntimeError("ADMIN_PASSWORD_HASH must contain a valid bcrypt hash")
elif ADMIN_PASSWORD:
    if len(ADMIN_PASSWORD.encode("utf-8")) > 72:
        raise RuntimeError("ADMIN_PASSWORD exceeds bcrypt's 72-byte limit")
    ADMIN_PASSWORD_HASH = pwd_context.hash(ADMIN_PASSWORD)

# Do not retain the plaintext admin password in the application global.
ADMIN_PASSWORD = ""
''',
"admin password configuration"
)

replace_once(
'''def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
''',
'''def hash_password(plain: str) -> str:
    if len(plain.encode("utf-8")) > 72:
        raise ValueError("Password must not exceed 72 UTF-8 bytes")
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    if len(plain.encode("utf-8")) > 72:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except (ValueError, TypeError):
        return False
''',
"password helper functions"
)

start = text.index("USERS = {")
end = text.index("\n\nclass RegisterRequest", start)

users_block = '''ADMIN_USER_EMAIL = (
    ADMIN_USERNAME if "@" in ADMIN_USERNAME else "admin@eva.ai"
)

USERS = {}
if ADMIN_PASSWORD_HASH:
    _admin_user = {
        "name": "Admin",
        "email": ADMIN_USER_EMAIL,
        "username": ADMIN_USERNAME,
        "password_hash": ADMIN_PASSWORD_HASH,
        "role": "administrator",
        "provider": "email",
    }
    USERS[ADMIN_USERNAME] = _admin_user
    if ADMIN_USERNAME == "admin":
        USERS["admin@eva.ai"] = _admin_user
'''
text = text[:start] + users_block + text[end:]
print("PATCH READY: hardcoded admin users removed")

register_start = text.index('@app.post("/auth/register")')
login_start = text.index('@app.post("/auth/login")', register_start)

register_block = '''@app.post("/auth/register")
def register_user(data: RegisterRequest):
    email = data.email.lower().strip()
    name = data.name or data.full_name or email.split("@")[0]

    if len(data.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Password exceeds 72 UTF-8 bytes")

    if email == ADMIN_USER_EMAIL or email in USERS:
        raise HTTPException(status_code=400, detail="User already exists")

    USERS[email] = {
        "name": name,
        "email": email,
        "username": (data.username or email).strip(),
        "password_hash": hash_password(data.password),
        "role": "user",
        "provider": "email",
    }

    token = create_token(USERS[email])
    return {
        "access_token": token,
        "token": token,
        "user": public_user(USERS[email]),
    }


'''
text = text[:register_start] + register_block + text[login_start:]
print("PATCH READY: registration password hashing")

login_start = text.index('@app.post("/auth/login")')
google_start = text.index('@app.post("/auth/google")', login_start)

login_block = '''@app.post("/auth/login")
def login_user(data: LoginRequest):
    username = (data.email or data.username or "").lower().strip()

    if ADMIN_USERNAME == "admin" and username == "admin@eva.ai":
        username = ADMIN_USERNAME

    user = USERS.get(username)
    if not user or not verify_password(
        data.password, user.get("password_hash", "")
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(user)
    return {
        "access_token": token,
        "token": token,
        "user": public_user(user),
        "name": user.get("name"),
        "role": user.get("role"),
    }


'''
text = text[:login_start] + login_block + text[google_start:]
print("PATCH READY: login hash verification and debug-log removal")

# Verify key invariants before writing.
if '"password": "admin123"' in text:
    sys.exit("STOP: hardcoded admin password remains")
if 'print("LOGIN DATA:", data)' in text or 'print("USERS:", USERS)' in text:
    sys.exit("STOP: credential debug logging remains")
if text.count('def login_user(') != 1 or text.count('def register_user(') != 1:
    sys.exit("STOP: unexpected auth endpoint definition count")

# Write only the working app.py; preserve the staged index and all other files.
app.write_text(text, encoding="utf-8", newline="")
print("PATCH APPLIED: src/app.py only")
print("SHA256:", hashlib.sha256(app.read_bytes()).hexdigest())
