# streamlit_app.py
import streamlit as st
import sqlite3, os
from datetime import date

# ─── INITIALISATION ────────────────────────────────────────────────────────────

# Dossier pour les reçus
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Connexion et création des tables si nécessaires
conn = sqlite3.connect("fuel.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    nom TEXT,
    role TEXT,
    email TEXT,
    avatar_path TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY,
    name TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS depenses (
    id INTEGER PRIMARY KEY,
    type TEXT,
    date TEXT,
    montant INTEGER,
    kilometrage INTEGER,
    photo_path TEXT,
    user_id INTEGER,
    vehicle_id INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(vehicle_id) REFERENCES vehicles(id)
)
""")
conn.commit()

# Seed initial data si vide
if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
    users = [
        ("Alice Chauffeur", "chauffeur", "alice@ex.com", ""),
        ("Bob ChefMission", "chef", "bob@ex.com", ""),
        ("Céline Gestionnaire", "gestionnaire", "celine@ex.com", "")
    ]
    c.executemany("INSERT INTO users (nom,role,email,avatar_path) VALUES (?,?,?,?)", users)
    vehicles = [("Véhicule A",), ("Véhicule B",), ("Véhicule C",)]
    c.executemany("INSERT INTO vehicles (name) VALUES (?)", vehicles)
    conn.commit()

# Récupération
users = c.execute("SELECT * FROM users").fetchall()
vehicles = c.execute("SELECT * FROM vehicles").fetchall()

# ─── SIDEBAR : CHOIX UTILISATEUR & NAVIGATION ───────────────────────────────────

st.sidebar.title("Connexion & menu")
user_sel = st.sidebar.selectbox(
    "Je suis", 
    {u[1]: u for u in users}.keys()
)
current_user = next(u for u in users if u[1]==user_sel)
is_admin = current_user[2] == "gestionnaire"

page = st.sidebar.radio("Aller à", ["Ajouter une dépense", "Tableau de bord", "Profils"])

# ─── HEADER ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between">
  <h1>Gestion Carburant</h1>
  <div style="text-align:right">
    <p><strong>{current_user[1]}</strong><br/><small>{current_user[2]}</small></p>
  </div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ─── PAGE “Ajouter une dépense” ─────────────────────────────────────────────────

if page == "Ajouter une dépense":
    st.header("Ajouter une dépense")
    with st.form("depense_form"):
        dep_type = st.selectbox("Type de dépense", ["carburant", "autre"])
        d = st.date_input("Date", value=date.today())
        km = st.number_input("Kilométrage (km)", min_value=0, step=1) if dep_type=="carburant" else None
        veh = st.selectbox("Véhicule", {v[1]:v for v in vehicles}.keys())
        montant = st.number_input("Montant (FCFA)", min_value=0, step=100)
        photo = st.file_uploader("Reçu (photo)", type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Enregistrer")
        if submitted:
            # Sauvegarde de la photo
            photo_path = None
            if photo:
                ext = photo.name.split(".")[-1]
                photo_path = os.path.join(UPLOAD_DIR, f"{date.today()}_{os.urandom(4).hex()}.{ext}")
                with open(photo_path,"wb") as f: f.write(photo.getbuffer())
            # Insertion DB
            vid = next(v[0] for v in vehicles if v[1]==veh)
            c.execute("""
                INSERT INTO depenses
                (type,date,montant,kilometrage,photo_path,user_id,vehicle_id)
                VALUES (?,?,?,?,?,?,?)
            """, (dep_type, d.isoformat(), montant, km, photo_path, current_user[0], vid))
            conn.commit()
            st.success("Dépense enregistrée ✅")

# ─── PAGE “Tableau de bord” ───────────────────────────────────────────────────────

elif page == "Tableau de bord":
    if not is_admin:
        st.warning("⚠️ Seul le gestionnaire peut voir le tableau de bord.")
    else:
        st.header("Tableau de bord global")
        df = conn.execute("""
            SELECT u.nom, SUM(d.montant) 
            FROM depenses d JOIN users u ON d.user_id=u.id
            WHERE d.type='carburant'
            GROUP BY u.nom
        """).fetchall()
        noms, totaux = zip(*df) if df else ([],[])
        st.bar_chart(data={n: t for n,t in zip(noms,totaux)})

        st.header("Consommation par véhicule")
        df2 = conn.execute("""
            SELECT v.name, SUM(d.montant)
            FROM depenses d JOIN vehicles v ON d.vehicle_id=v.id
            WHERE d.type='carburant'
            GROUP BY v.name
        """).fetchall()
        vehs, vtots = zip(*df2) if df2 else ([],[])
        st.bar_chart(data={v: t for v,t in zip(vehs,vtots)})

# ─── PAGE “Profils” ──────────────────────────────────────────────────────────────

elif page == "Profils":
    st.header("Profils utilisateurs")
    if is_admin:
        rows = users
    else:
        rows = [current_user]
    for u in rows:
        st.subheader(u[1])
        st.write(f"- Rôle : {u[2]}")
        st.write(f"- Email : {u[3]}")
        # avatar si disponible
        if u[4] and os.path.exists(u[4]):
            st.image(u[4], width=100)

# ─── FIN ────────────────────────────────────────────────────────────────────────
