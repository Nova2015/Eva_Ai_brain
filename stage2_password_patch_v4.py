from pathlib import Path
import hashlib
import sys

app = Path("src/app.py")
backup = Path(r"D:\EVA_SECURITY_STAGE2_20261103_142243\src_app.py.working")

if not app.is_file() or not backup.is_file():
    sys.exit("STOP: source or backup missing")

raw = app.read_bytes()
if hashlib.sha256(raw).digest() != hashlib.sha256(backup.read_bytes()).digest():
    sys.exit("STOP: backup mismatch; no changes made")

text = raw.decode("utf-8").replace("\r\n", "\n")

# 1. Admin password configuration
anchor = 'SECRET_KEY        = _require_secret("SECRET_KEY")'
if text.count(anchor) != 1:
    sys.exit("STOP: SECRET_KEY anchor mismatch")

config_start = text.index("ADMIN_USERNAME", text.index(anchor))
config_end = text.index("# _______ Rate limiter", config_start)

new_config = '''ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin@eva.ai").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
if ADMIN_PASSWORD_HASH:
    if pwd_context.identify(ADMIN_PASSWORD_HASH) != "bcrypt":
        raise RuntimeError("ADMIN_PASSWORD_HASH must contain a valid bcrypt hash")
elif ADMIN_PASSWORD:
    if len(ADMIN_PASSWORD.encode("utf-8")) > 72:
        raise RuntimeError("ADMIN_PASSWORD exceeds bcrypt's 72-byte limit")
    ADMIN_PASSWORD_HASH = pwd_context.hash(ADMIN_PASSWORD)

ADMIN_PASSWORD = ""

'''
text = text[:config_start] + new_config + text[config_end:]
print("READY: admin configuration")

# 2. Password helpers
helper_start = text.index("def verify_password(")
helper_end = text.index("def create_access_token(", helper_start)

new_helpers = '''def hash_password(plain: str) -> str:
    if len(plain.encode("utf-8")) > 72:
        raise ValueError("Password must not exceed 72 UTF-8 bytes")
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed or len(plain.encode("utf-8")) > 72:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except (ValueError, TypeError):
        return False


'''
text = text[:helper_start] + new_helpers + text[helper_end:]
print("READY: password helpers")

# 3. Admin users
users_start = text.index("USERS = {")
users_end = text.index("class RegisterRequest", users_start)

new_users = '''ADMIN_USER_EMAIL = (
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
text = text[:users_start] + new_users + text[users_end:]
print("READY: admin users")

# 4. Registration
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
print("READY: registration")

# 5. Login; Google route is preserved
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
print("READY: login")

# 6. Validate all changes before writing
checks = {
    "single login endpoint": text.count("def login_user(") == 1,
    "single register endpoint": text.count("def register_user(") == 1,
    "single password verifier": text.count("def verify_password(") == 1,
    "single hash helper": text.count("def hash_password(") == 1,
    "no hardcoded admin123": '"password": "admin123"' not in text,
    "no credential debug logs": (
        'print("LOGIN DATA:", data)' not in text
        and 'print("USERS:", USERS)' not in text
    ),
    "registration hashes password": (
        '"password_hash": hash_password(data.password)' in text
    ),
    "login verifies password hash": (
        'user.get("password_hash", "")' in text
    ),
    "Google route preserved": '@app.post("/auth/google")' in text,
}

for name, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ": " + name)

if not all(checks.values()):
    sys.exit("STOP: validation failed; source unchanged")

# Preserve Windows CRLF
updated = text.replace("\n", "\r\n").encode("utf-8")

# Last-moment backup integrity check
if hashlib.sha256(app.read_bytes()).digest() != hashlib.sha256(backup.read_bytes()).digest():
    sys.exit("STOP: source changed during patch preparation")

app.write_bytes(updated)

print("PATCH APPLIED: src/app.py only")
print("CRLF RESTORED:", b"\r\n" in updated)
print("NEW SHA256:", hashlib.sha256(updated).hexdigest())
