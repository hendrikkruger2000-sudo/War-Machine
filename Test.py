# main.py
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import qrcode
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
import sqlite3, json, os


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs("static/waybills", exist_ok=True)
os.makedirs("static/qrcodes", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

def init_db():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()

    try:
        # Create tables if they don't exist
        cursor.execute("""CREATE TABLE IF NOT EXISTS updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ops TEXT,
            hmj TEXT,
            haz TEXT,
            company TEXT,
            date TEXT,
            time TEXT,
            "update" TEXT
        );""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS completed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ops TEXT,
            company TEXT,
            delivery_date TEXT,
            time TEXT,
            signed_by TEXT,
            document TEXT,
            pod TEXT
        );""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_number TEXT,
            service_type TEXT,
            collection_company TEXT,
            collection_address TEXT,
            collection_person TEXT,
            collection_number TEXT,
            delivery_company TEXT,
            delivery_address TEXT,
            delivery_person TEXT,
            delivery_number TEXT,
            client_reference TEXT,
            pickup_date TEXT,
            inco_terms TEXT,
            client_notes TEXT,
            pdf_path TEXT,
            timestamp TEXT,
            assigned_driver TEXT,
            status TEXT
        );""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_number TEXT,
            driver_id TEXT,
            timestamp TEXT
        );""")

        print("✅ Tables created")

        # Check if requests table is empty
        cursor.execute("SELECT COUNT(*) FROM requests")
        request_count = cursor.fetchone()[0]

        if request_count == 0:
            print("🔁 Restoring database from JSON backups...")

            def restore_table(json_path, table_name):
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                        for row in data:
                            cursor.execute(f"""
                                INSERT INTO {table_name} ({','.join(row.keys())})
                                VALUES ({','.join(['?'] * len(row))})
                            """, list(row.values()))
                    print(f"✅ Restored {table_name} from {json_path}")

            restore_table("static/backups/requests.json", "requests")
            restore_table("static/backups/updates.json", "updates")
            restore_table("static/backups/completed.json", "completed")

        else:
            print("✅ hazmat.db already contains data — no restore needed")

        conn.commit()
    except Exception as e:
        print("❌ init_db() failed:", e)
    finally:
        conn.close()
init_db()

def backup_database():
    os.makedirs("static/backups", exist_ok=True)
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    try:
        backup_database()
    except Exception as e:
        print("❌ Backup failed:", e)

    def dump_table(table_name, filename):
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        with open(f"static/backups/{filename}", "w") as f:
            json.dump([dict(zip(columns, row)) for row in rows], f, indent=2)

    dump_table("requests", "requests.json")
    dump_table("updates", "updates.json")
    dump_table("completed", "completed.json")
    conn.close()
    print("✅ Database backed up to JSON")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>Hazmat Collection System</title>
        <link rel="icon" href="/icon.png" type="image/png">
        <style>
          body { font-family:Segoe UI; text-align:center; padding:2rem; background:#ECEFF1; }
          h1 { color:#D32F2F; }
          p { color:#455A64; }
          button {
            margin:1rem; padding:0.75rem 1.5rem;
            background-color:#388E3C; color:white;
            border:none; border-radius:4px;
            font-size:1rem;
          }
        </style>
      </head>
      <body>
        <img src="/logo.png" alt="Hazmat Logo" style="width:150px; margin-bottom:1rem;">
        <h1>Welcome to Hazmat Collection System</h1>
        <p>Choose an action below:</p>
        <button onclick="window.location.href='/submit'">Book a Collection</button>
        <button onclick="window.location.href='/track'">Track a Collection</button>
      </body>
    </html>
    """

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/icon.png")

@app.get("/ping")
def ping():
    return {"status": "awake"}

@app.get("/ops/unassigned")
def get_unassigned():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reference_number, collection_company, collection_address, pickup_date
        FROM requests WHERE assigned_driver IS NULL AND status IS NOT 'Delivered'
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"hazjnb_ref": r[0], "company": r[1], "address": r[2], "pickup_date": r[3]} for r in rows]

@app.get("/ops/collections")
def get_available_collections():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reference_number, collection_company, collection_address, pickup_date
        FROM requests
        WHERE assigned_driver IS NULL AND status IS NOT 'Delivered'
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{
        "hazjnb_ref": r[0],
        "company": r[1],
        "address": r[2],
        "pickup_date": r[3]
    } for r in rows]

@app.get("/submit", response_class=HTMLResponse)
def submit_form():
    return """
    <html>
      <head>
        <title>Book a Hazmat Collection</title>
        <link rel="icon" href="/icon.png" type="image/png">
        <style>
          body { font-family:Segoe UI; padding:2rem; background:#ECEFF1; }
          h1 { color:#D32F2F; text-align:center; }
          form { max-width:600px; margin:auto; background:white; padding:2rem; border-radius:8px; }
          label { display:block; margin-top:1rem; font-weight:bold; }
          input, textarea, select {
            width:100%; padding:0.5rem; margin-top:0.5rem;
            border:1px solid #B0BEC5; border-radius:4px;
          }
          button {
            margin-top:2rem; padding:0.75rem 1.5rem;
            background-color:#388E3C; color:white;
            border:none; border-radius:4px;
            font-size:1rem;
          }
        </style>
      </head>
      <body>
        <h1>Book a Hazmat Collection</h1>
        <form action="/submit" method="post" enctype="multipart/form-data">
          <label>Service Type</label>
          <select name="serviceType">
            <option value="local">Local</option>
            <option value="export">Export</option>
            <option value="import">Import</option>
          </select>

          <label>Collection Company</label>
          <input type="text" name="collection_company_local">

          <label>Collection Address</label>
          <input type="text" name="collection_address_local">

          <label>Pickup Date</label>
          <input type="date" name="pickup_date_local">

          <label>Delivery Company</label>
          <input type="text" name="delivery_company_local">

          <label>Delivery Address</label>
          <input type="text" name="delivery_address_local">

          <label>Client Reference</label>
          <input type="text" name="client_reference_local">

          <label>Client Notes</label>
          <textarea name="client_notes_local"></textarea>

          <label>Shipment Documents</label>
          <input type="file" name="shipment_docs" multiple>

          <button type="submit">Submit Collection Request</button>
        </form>
      </body>
    </html>
    """



@app.get("/ops/assigned")
def get_assigned():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reference_number, collection_company, collection_address, pickup_date, assigned_driver, status
        FROM requests WHERE assigned_driver IS NOT NULL AND status IS NOT 'Delivered'
        ORDER BY timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"hazjnb_ref": r[0], "company": r[1], "address": r[2], "pickup_date": r[3], "driver": r[4], "status": r[5]} for r in rows]

@app.get("/driver/{code}")
def get_driver_jobs(code: str):
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT reference_number, collection_company, collection_address, pickup_date
        FROM requests WHERE assigned_driver = ?
    """, (code,))
    rows = cursor.fetchall()
    conn.close()
    return [{"hazjnb_ref": r[0], "company": r[1], "address": r[2], "pickup_date": r[3]} for r in rows]


@app.post("/scan_qr")
def scan_qr(payload: dict):
    ref = payload.get("ref")
    driver_id = payload.get("driver_id")
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scan_log (reference_number, driver_id, timestamp) VALUES (?, ?, ?)
    """, (ref, driver_id, timestamp))
    cursor.execute("""
        UPDATE requests SET status = 'Collected' WHERE reference_number = ?
    """, (ref,))
    conn.commit()
    conn.close()
    return {"status": "collected", "ref": ref, "driver": driver_id}

@app.post("/ops/updates")
def submit_update(payload: dict):
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO updates (ops, hmj, haz, company, date, time, update)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["ops"], payload["hmj"], payload["haz"], payload["company"],
        payload["date"], payload["time"], payload["update"]
    ))
    conn.commit()
    conn.close()
    backup_database()
    return {"status": "update received"}


@app.get("/ops/updates")
def get_updates():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute('SELECT ops, hmj, haz, company, date, time, "update" FROM updates ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return [{"ops": r[0], "hmj": r[1], "haz": r[2], "company": r[3], "date": r[4], "time": r[5], "update": r[6]} for r in rows]

@app.post("/ops/completed")
def submit_completed(payload: dict):
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO completed (ops, company, delivery_date, time, signed_by, document, pod)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        payload["ops"], payload["company"], payload["delivery_date"], payload["time"],
        payload["signed_by"], payload["document"], payload["pod"]
    ))
    cursor.execute("""
        UPDATE requests SET status = 'Delivered' WHERE reference_number = ?
    """, (payload["haz_ref"],))
    conn.commit()
    conn.close()
    backup_database()
    return {"status": "completed"}


@app.get("/ops/completed")
def get_completed():
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ops, company, delivery_date, time, signed_by, document, pod FROM completed ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"ops": r[0], "company": r[1], "delivery_date": r[2], "time": r[3], "signed_by": r[4], "document": r[5], "pod": r[6]} for r in rows]

# Remaining routes: /submit, /pdf/{id}, /confirm/{ref}, /thankyou, generate_pdf — already correct in your version

@app.post("/submit")
async def submit(request: Request):
    form = await request.form()
    files = await request.form()
    uploaded_files = request._form.getlist("shipment_docs")

    service_type = form.get("serviceType")

    def get_field(name):
        return form.get(f"{name}_local") or form.get(f"{name}_export") or form.get(f"{name}_import")

    def get_inco():
        if service_type == "local":
            return "DTD"
        return form.get("inco_terms_export") or form.get("inco_terms_import") or "N/A"

    collection_company = get_field("collection_company")
    collection_address = get_field("collection_address")
    collection_person = get_field("collection_person")
    collection_number = get_field("collection_number")
    delivery_company = get_field("delivery_company")
    delivery_address = get_field("delivery_address")
    delivery_person = get_field("delivery_person")
    delivery_number = get_field("delivery_number")
    client_reference = get_field("client_reference")
    pickup_date = get_field("pickup_date")
    client_notes = get_field("client_notes")
    inco_terms = get_inco()
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    counter_path = "static/backups/ref_counter.txt"

    # Read current counter
    if os.path.exists(counter_path):
        with open(counter_path, "r") as f:
            last_id = int(f.read().strip())
    else:
        last_id = 0

    # Increment and write back
    new_id = last_id + 1
    with open(counter_path, "w") as f:
        f.write(str(new_id))

    reference_number = f"HAZJNB{str(new_id).zfill(4)}"

    cursor.execute("""
        INSERT INTO requests (
            reference_number, service_type, collection_company, collection_address, collection_person, collection_number,
            delivery_company, delivery_address, delivery_person, delivery_number,
            client_reference, pickup_date, inco_terms, client_notes, pdf_path, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reference_number, service_type, collection_company, collection_address, collection_person, collection_number,
        delivery_company, delivery_address, delivery_person, delivery_number,
        client_reference, pickup_date, inco_terms, client_notes, "", timestamp
    ))
    request_id = cursor.lastrowid
    conn.commit()
    conn.close()
    backup_database()

    for file in uploaded_files:
        contents = await file.read()
        with open(f"static/uploads/{reference_number}_{file.filename}", "wb") as f:
            f.write(contents)

    # ✅ QR code now links to HAZJNB reference number
    qr_url = f"https://hazmat-collection.onrender.com/confirm/{reference_number}"
    qr_img = qrcode.make(qr_url)
    qr_path = f"static/qrcodes/qr_{request_id}.png"
    qr_img.save(qr_path)

    pdf_path = f"static/waybills/waybill_{request_id}.pdf"
    generate_pdf({
        "reference_number": reference_number,
        "service_type": service_type,
        "collection_company": collection_company,
        "collection_address": collection_address,
        "collection_person": collection_person,
        "collection_number": collection_number,
        "delivery_company": delivery_company,
        "delivery_address": delivery_address,
        "delivery_person": delivery_person,
        "delivery_number": delivery_number,
        "client_reference": client_reference,
        "pickup_date": pickup_date,
        "inco_terms": inco_terms,
        "client_notes": client_notes
    }, request_id, qr_path, pdf_path)

    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE requests SET pdf_path = ? WHERE id = ?", (pdf_path, request_id))
    conn.commit()
    conn.close()
    backup_database()

    return HTMLResponse(f"""
    <html>
      <head>
        <title>Waybill Generated</title>
        <style>
          body {{
            font-family: Segoe UI;
            background: #ECEFF1;
            text-align: center;
            padding: 2rem;
          }}
          button {{
            margin: 1rem;
            padding: 0.75rem 1.5rem;
            background-color: #388E3C;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
          }}
        </style>
      </head>
      <body>
        <h1 style="color:#D32F2F;">Waybill Generated</h1>
        <p>Your waybill is ready. Click below to view or download it:</p>
        <button onclick="window.open('/pdf/{request_id}', '_blank')">📄 View Waybill PDF</button>
        <button onclick="window.location.href='/thankyou'">✅ Continue</button>
      </body>
    </html>
    """)

@app.get("/pdf/{request_id}")
def serve_pdf(request_id: int):
    path = f"static/waybills/waybill_{request_id}.pdf"
    file = open(path, "rb")
    return StreamingResponse(
        file,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="waybill_{request_id}.pdf"'}
    )

@app.get("/thankyou", response_class=HTMLResponse)
def thank_you():
    return HTMLResponse("""
    <html><body style="font-family:Segoe UI; text-align:center; padding:2rem;">
        <h1 style="color:#388E3C;">Thank you! Your request has been submitted.</h1>
        <button onclick="window.location.href='/'" style="margin:1rem; padding:0.75rem 1.5rem; background-color:#D32F2F; color:white; border:none; border-radius:4px;">Book Another Collection</button>
        <button onclick="window.location.href='/track'" style="margin:1rem; padding:0.75rem 1.5rem; background-color:#455A64; color:white; border:none; border-radius:4px;">Track a Collection</button>
    </body></html>
    """)

# ✅ QR confirmation now uses HAZJNB reference
@app.get("/confirm/{hazjnb_ref}", response_class=HTMLResponse)
def confirm(hazjnb_ref: str):
    return HTMLResponse(f"<h1>Driver confirmed request {hazjnb_ref}</h1>")


@app.post("/assign")
def assign_collection(payload: dict):
    driver_code = payload.get("driver_code")
    hazjnb_ref = payload.get("hazjnb_ref")

    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()

    # Log incoming request
    print(f"🚨 Assigning driver {driver_code} to reference {hazjnb_ref}")

    # Run update
    cursor.execute("""
        UPDATE requests SET assigned_driver = ?, status = 'Assigned' WHERE reference_number = ?
    """, (driver_code, hazjnb_ref))

    conn.commit()
    affected = cursor.rowcount
    conn.close()

    # Log result
    if affected == 0:
        print(f"❌ No matching reference_number found for {hazjnb_ref}")
        return JSONResponse(content={"status": "error", "message": "Reference not found"}, status_code=404)

    print(f"✅ Assignment succeeded for {hazjnb_ref}")
    return {"status": "success", "driver": driver_code, "ref": hazjnb_ref}

@app.get("/driver/{code}")
def get_driver_jobs(code: str):
    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT reference_number, collection_company, collection_address, pickup_date
        FROM requests WHERE assigned_driver = ?
    """, (code,))
    rows = cursor.fetchall()
    conn.close()

    jobs = []
    for row in rows:
        jobs.append({
            "hazjnb_ref": row[0],
            "company": row[1],
            "address": row[2],
            "pickup_date": row[3]
        })

    return jobs
@app.get("/ops/backup")
def trigger_backup():
    backup_database()
    return {"status": "backup complete"}

@app.post("/scan_qr")
def scan_qr(payload: dict):
    ref = payload.get("ref")
    driver_id = payload.get("driver_id")
    timestamp = datetime.now().isoformat()

    conn = sqlite3.connect("hazmat.db")
    cursor = conn.cursor()

    # Log scan
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_number TEXT,
            driver_id TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO scan_log (reference_number, driver_id, timestamp)
        VALUES (?, ?, ?)
    """, (ref, driver_id, timestamp))

    # Update status
    cursor.execute("""
        UPDATE requests SET status = 'Collected' WHERE reference_number = ?
    """, (ref,))

    conn.commit()
    conn.close()

    print(f"✅ QR scan logged and status updated for {ref}")
    return {"status": "collected", "ref": ref, "driver": driver_id}

def generate_pdf(data, request_id, qr_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFillColor(HexColor("#ECEFF1"))
    c.rect(0, 0, width, height, fill=1)

    logo_path = "static/logo.png"
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 20*mm, height - 30*mm, width=40*mm, height=20*mm, preserveAspectRatio=True, mask='auto')

    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(HexColor("#D32F2F"))
    c.drawString(70*mm, height - 25*mm, "Hazmat Collection Waybill")

    def section(title, y):
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(HexColor("#455A64"))
        c.drawString(20*mm, y, title)
        c.setStrokeColor(HexColor("#B0BEC5"))
        c.line(20*mm, y - 2*mm, width - 20*mm, y - 2*mm)
        return y - 10*mm

    def field(label, value, y):
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#212121"))
        c.drawString(25*mm, y, f"{label}:")
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70 * mm, y, value or "—")
        return y - 7 * mm

    y = height - 50 * mm
    y = section("Shipment Details", y)
    y = field("Reference Number", data["reference_number"], y)
    y = field("Service Type", data["service_type"], y)
    y = field("Client Reference", data["client_reference"], y)
    y = field("Pickup Date", data["pickup_date"], y)
    y = field("Inco Terms", data["inco_terms"] or "N/A", y)

    y -= 5 * mm
    y = section("Collection", y)
    y = field("Company", data["collection_company"], y)
    y = field("Address", data["collection_address"], y)
    y = field("Contact Person", data["collection_person"], y)
    y = field("Contact Number", data["collection_number"], y)

    y -= 5 * mm
    y = section("Delivery", y)
    y = field("Company", data["delivery_company"], y)
    y = field("Address", data["delivery_address"], y)
    y = field("Contact Person", data["delivery_person"], y)
    y = field("Contact Number", data["delivery_number"], y)

    y -= 5 * mm
    y = section("Client Notes", y)
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#212121"))
    c.drawString(25 * mm, y, data["client_notes"] or "None")

    # QR Code
    if os.path.exists(qr_path):
        c.drawImage(qr_path, width - 50 * mm, 20 * mm, width=30 * mm, preserveAspectRatio=True, mask='auto')

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#607D8B"))
    c.drawString(20 * mm, 10 * mm, "Generated by Hazmat Global Logistics System")

    c.save()