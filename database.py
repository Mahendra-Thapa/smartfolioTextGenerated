import psycopg2
from psycopg2.extras import RealDictCursor
from config import Config


def get_connection():
    """Return a new PostgreSQL connection."""
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        cursor_factory=RealDictCursor
    )


def init_db():
    """Create all tables if they do not exist."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id            SERIAL PRIMARY KEY,
            full_name     VARCHAR(150)  NOT NULL,
            bio           TEXT,
            skills        TEXT,
            experience    TEXT,
            education     TEXT,
            projects      TEXT,
            contact_info  TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Tables initialized.")


def save_portfolio(data: dict) -> int:
    """
    Insert a portfolio row and return the new id.

    data keys: full_name, bio, skills, experience,
               education, projects, contact_info
    """
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        INSERT INTO portfolios
            (full_name, bio, skills, experience, education, projects, contact_info)
        VALUES
            (%(full_name)s, %(bio)s, %(skills)s, %(experience)s,
             %(education)s, %(projects)s, %(contact_info)s)
        RETURNING id;
    """, data)

    new_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return new_id


def get_portfolio(portfolio_id: int) -> dict | None:
    """Fetch one portfolio by id."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("SELECT * FROM portfolios WHERE id = %s;", (portfolio_id,))
    row = cur.fetchone()

    cur.close()
    conn.close()
    return dict(row) if row else None


def get_all_portfolios() -> list[dict]:
    """Fetch all portfolios (for training data collection)."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("SELECT * FROM portfolios ORDER BY created_at DESC;")
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return [dict(r) for r in rows]