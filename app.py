# streamlit_app.py
import streamlit as st
import sqlite3, os
from datetime import date

# ─── INIT ────────────────────────────────────────────────────────────────────────

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

conn = sqlite3.connect("fuel.db", check_same_thread=False)
c = conn.cursor()

# Création des tables si nécessaire
c.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    nom TEXT,
    role TEXT,
    email TEXT,
    avatar_path TEXT
);
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY,
    name TEXT
);
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    chef_id INTEGER,
    chauffeur_id INTEGER,
    FOREIGN KEY(chef_id) REFERENCES users(id),
    FOREIGN KEY(chauffeur_id) REFERENCES users(id)
);
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
);
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
    teams = [
        # (chef_id, chauffeur_id)
        (2, 1),  # Bob <-> Alice
    ]
    c.executemany("INSERT INTO teams (chef_id,chauffeur_id) VALUES (?,?)", teams)
    conn.commit()

# Récupérations
users = c.execute("SELECT * FROM users").fetchall()
vehicles = c.execute("SELECT * FROM vehicles").fetchall()
teams = c.execute("SELECT t.id,u1.nom,u2.nom FROM teams t "
                  "JOIN users u1 ON t.chef_id=u1.id "
                  "JOIN users u2 ON t.chauffeur_id=u2.id").fetchall()

# ─── LOGIN ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Gestion Carburant", layout="wide")
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("🔑 Connexion")
    with st.form("login"):
        email = st.selectbox("Votre compte", [u[3] for u in users])
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            user = next(u for u in users if u[3] == email)
            st.session_state.user = user
            st.experimental_rerun()
    st.stop()

current_user = st.session_state.user
is_admin = current_user[2] == "gestionnaire"

# ─── HEADER & NAV ───────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between">
  <h1>Gestion Carburant</h1>
  <div style="text-align:right">
    <p><strong>{current_user[1]}</strong><br/><small>{current_user[2]}</small></p>
  </div>
</div>
<hr/>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Ajouter une dépense", "Tableau de bord", "Profils", "Déconnexion"])

if page == "Déconnexion":
    st.session_state.user = None
    st.experimental_rerun()

# ─── AJOUT DEPENSE ──────────────────────────────────────────────────────────────

if page == "Ajouter une dépense":
    st.header("➕ Ajouter une dépense")
    with st.form("depense_form"):
        dep_type = st.selectbox("Type de dépense", ["carburant", "autre"])
        d = st.date_input("Date", value=date.today())
        km = st.number_input("Kilométrage (km)", min_value=0, step=1) if dep_type=="carburant" else None
        veh = st.selectbox("Véhicule", [v[1] for v in vehicles])
        montant = st.number_input("Montant (FCFA)", min_value=0, step=100)
        photo = st.file_uploader("Reçu (photo)", type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Enregistrer")
        if submitted:
            # Sauvegarde de la photo
            photo_path = None
            if photo:
                ext = photo.name.split(".")[-1]
                filename = f"{date.today()}_{os.urandom(4).hex()}.{ext}"
                photo_path = os.path.join(UPLOAD_DIR, filename)
                with open(photo_path,"wb") as f:
                    f.write(photo.getbuffer())
            vid = next(v[0] for v in vehicles if v[1]==veh)
            c.execute("""
                INSERT INTO depenses
                (type,date,montant,kilometrage,photo_path,user_id,vehicle_id)
                VALUES (?,?,?,?,?,?,?)
            """, (dep_type, d.isoformat(), montant, km, photo_path, current_user[0], vid))
            conn.commit()
            st.success("✅ Dépense enregistrée")

# ─── DASHBOARD ──────────────────────────────────────────────────────────────────

elif page == "Tableau de bord":
    st.header("📊 Tableau de bord")
    if not is_admin:
        st.warning("⚠️ Seul le gestionnaire peut voir le tableau de bord.")
    else:
        # Chauffeurs
        df_ch = conn.execute("""
            SELECT u.nom, SUM(d.montant)
            FROM depenses d
            JOIN users u ON d.user_id=u.id
            WHERE u.role='chauffeur' AND d.type='carburant'
            GROUP BY u.nom
        """).fetchall()
        if df_ch:
            noms, vals = zip(*df_ch)
            st.subheader("Consommation par chauffeur")
            st.bar_chart({n: v for n, v in zip(noms, vals)})

        # Chefs de mission
        df_chef = conn.execute("""
            SELECT u.nom, SUM(d.montant)
            FROM depenses d
            JOIN users u ON d.user_id=u.id
            WHERE u.role='chef' AND d.type='carburant'
            GROUP BY u.nom
        """).fetchall()
        if df_chef:
            noms2, vals2 = zip(*df_chef)
            st.subheader("Consommation par chef de mission")
            st.bar_chart({n: v for n, v in zip(noms2, vals2)})

        # Véhicules
        df_v = conn.execute("""
            SELECT v.name, SUM(d.montant)
            FROM depenses d
            JOIN vehicles v ON d.vehicle_id=v.id
            WHERE d.type='carburant'
            GROUP BY v.name
        """).fetchall()
        if df_v:
            vehs, vtots = zip(*df_v)
            st.subheader("Consommation par véhicule")
            st.bar_chart({v: t for v, t in zip(vehs, vtots)})

        # Équipes
        df_team = []
        for t in teams:
            team_id, chef_nom, ch_nom = t
            total = conn.execute("""
                SELECT SUM(montant) FROM depenses
                WHERE type='carburant' AND (user_id=? OR user_id=?)
            """, (next(u[0] for u in users if u[1]==chef_nom),
                  next(u[0] for u in users if u[1]==ch_nom))
            ).fetchone()[0] or 0
            df_team.append((f"{chef_nom} & {ch_nom}", total))
        if df_team:
            names, sums = zip(*df_team)
            st.subheader("Consommation par équipe")
            st.bar_chart({n: s for n, s in zip(names, sums)})

# ─── PROFILS ────────────────────────────────────────────────────────────────────

elif page == "Profils":
    st.header("👥 Profils utilisateurs")
    if is_admin:
        rows = users
    else:
        rows = [current_user]
    for u in rows:
        st.subheader(u[1])
        st.write(f"- **Rôle** : {u[2]}")
        st.write(f"- **Email** : {u[3]}")
        if u[4] and os.path.exists(u[4]):
            st.image(u[4], width=100)
