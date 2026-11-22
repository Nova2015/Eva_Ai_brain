from pathlib import Path
import hashlib
import sys

app = Path("src/app.py")
backup = Path(r"D:\EVA_SECURITY_STAGE2_20261103_142243\src_app.py.working")

if not app.is_file() or not backup.is_file():
    sys.exit("STOP: source or backup missing")

original = app.read_bytes()
if hashlib.sha256(original).digest() != hashlib.sha256(backup.read_bytes()).digest():
    sys.exit("STOP: source differs from verified backup")

text = original.decode("utf-8")

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

'''
text = text[:config_start] + new_config + text[config_end:]
print("READY: admin password configuration")

# 2. Password helper functions
old_helper = '''def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)'''

new_helper = '''def hash_password(plain: str) -> str:
    if len(plain.encode("utf-8")) > 72:
        raise ValueError("Password must not exceed 72 UTF-8 bytes")
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed or len(plain.encode("utf-8")) > 72:
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except (ValueError, TypeError):
        return False'''

if text.count(old_helper) != 1:
    sys.exit(f"STOP: password helper exact match count = {text.count(old_helper)}")
text = text.replace(old_helper, new_helper, 1)
print("READY: password helper functions")

# 3. Remove hardcoded admin credentials
if text.count("USERS = {") != 1:
    sys.exit("STOP: USERS block marker mismatch")

users_start = text.index("USERS = {")
users_end = text.index("\n\nclass RegisterRequest", users_start)

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
print("READY: hardcoded admin users removed")

# 4. Registration stores only password hashes
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
print("READY: registration password hashing")

# 5. Login verifies bcrypt hashes; Google/phone routes remain untouched
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
print("READY: login hash verification")

# 6. Final pre-write checks
checks = {
    "one login endpoint": text.count("def login_user(") == 1,
    "one register endpoint": text.count("def register_user(") == 1,
    "one password verifier": text.count("def verify_password(") == 1,
    "password hash helper exists": text.count("def hash_password(") == 1,
    "no hardcoded admin123": '"password": "admin123"' not in text,
    "no credential debug prints": 'print("LOGIN DATA:", data)' not in text and 'print("USERS:", USERS)' not in text,
    "registration stores hash": '"password_hash": hash_password(data.password)' in text,
    "login verifies hash": 'user.get("password_hash", "")' in text,
}

for label, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ":", label)

if not all(checks.values()):
    sys.exit("STOP: pre-write validation failed; source unchanged")

# Write only src/app.py. Existing staged index remains untouched.
app.write_text(text, encoding="utf-8", newline="")
print("PATCH APPLIED: src/app.py only")
print("NEW_SHA256:", hashlib.sha256(app.read_bytes()).hexdigest())
