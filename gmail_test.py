print("🔥 Script started")

import smtplib

EMAIL = "shreelakshmi112.k@gmail.com"
PASS = "bivhdztfumfpbzuu"   # NO SPACES

print("🔄 Connecting to Gmail SMTP...")

server = smtplib.SMTP("smtp.gmail.com", 587)
server.set_debuglevel(1)
server.starttls()

print("🔐 Logging in...")
server.login(EMAIL, PASS)

print("📨 Sending email...")
server.sendmail(
    EMAIL,
    EMAIL,
    "Subject: SMTP TEST\n\nIf you got this, SMTP works."
)

server.quit()
print("✅ DONE — Email sent successfully")
