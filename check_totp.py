import re
from src.config import get_env

s = get_env("ANGEL_TOTP_SECRET")
print("Length:", len(s))
print("Looks like the 6-digit code:", s.isdigit() and len(s) <= 8)
print("Has spaces:", " " in s)
print("Has quotes:", '"' in s or "'" in s)
print("Only valid characters (A-Z, 2-7):", bool(re.fullmatch(r"[A-Za-z2-7 ]+=*", s)))