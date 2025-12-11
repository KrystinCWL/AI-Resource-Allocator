# CIC Scheduler

A small Flask application to manage course scheduling, faculty preferences, and to produce schedule exports with KPIs. This repo contains upload/normalization logic, scheduling algorithms, multi-sheet Excel exports, and an AI-driven analysis/summary feature.

**Project layout**
- `app.py` — Main Flask application and routes.
- `algorithm.py` — Scheduling algorithm and scoring utilities.
- `convert.py` — Conversion helpers used for import/export and scoring normalization.
- `templates/` — Jinja2 HTML templates for the web UI.
- `Static/` — CSS and JS assets (styles at `Static/css/style.css`, client scripts at `Static/js/app.js`).
- `sql/` — SQL files to create schema and seed data.
- `requirements.txt` — Python dependencies (includes `pandas`, `openpyxl`, etc.).


**Key features**
- Upload faculty preferences (supports multiple Excel sheet formats; sheet-selection heuristics included).
- Accept uploads either by `UserEmail` or `UserId`.
- Generate multi-sheet schedule exports (`.xlsx`) including Assignments, Faculty Load, Unassigned courses, and KPI summary.
- AI-powered analysis report (Gemini via `google.genai`) with retry/backoff and sanitized output.
- Basic admin UI for uploading, viewing schedules, and exporting.


**Requirements**
- Python 3.10+ (project has been developed and tested on Python 3.12).
- MySQL server for persistent storage (or adjust `config.py` to use a different DB).
- An account / credentials for Google GenAI if you plan to use the AI report feature.


Setup
-----
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure your database connection in `config.py` (or set equivalent environment variables used by your app). Typical values to set:

- `DB_HOST`
- `DB_USER`
- `DB_PASS`
- `DB_NAME`

If you prefer environment variables, ensure your `config.py` reads them (this repo expects DB settings in `config.py`).

4. Initialize the database schema and seed data (MySQL client example):

```powershell
mysql -u <user> -p < sql\01_schema_structure_create.sql
mysql -u <user> -p < sql\02_seed_data_insert.sql
```

(There are additional SQL files in the `sql/` folder for cleanup and helpers.)


Running the app
---------------
You can run the Flask app directly using `python`:

```powershell
python .\app.py
```

Or use the Flask CLI (PowerShell):

```powershell
$env:FLASK_APP='app.py'; $env:FLASK_ENV='development'; flask run
```

The app will be reachable at `http://127.0.0.1:5000` by default.


AI / Gemini configuration
-------------------------
The AI report uses Google GenAI (`google.genai`). To enable this feature you must provide valid credentials. Two common approaches:

- Set up Google Cloud Application Default Credentials and set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON path.
- Or set an API key / client settings as required by your environment and the `google.genai` client.
- To add failover keys (helps when one key is throttled), set `GEMINI_API_KEYS` to a comma-separated list; the app will try each key in order before showing an overload message. The legacy `GEMINI_API_KEY` still works and is used first if present.

If the model is temporarily unavailable, the app uses exponential backoff and returns a friendly message; the UI also sanitizes model output (removes bold markers and normalizes the date header).


Using the app (quick guide)
---------------------------
- Admin -> Upload: Upload your faculty preference Excel file. The importer attempts to find the correct sheet automatically.
- Admin -> Schedule: View generated schedules; use the `Export` button to download a multi-sheet `.xlsx` file containing Assignments, Faculty Load, Unassigned, and KPI sheets.
- Admin -> Analysis: Generate an AI summary report; use the provided Print or Download-as-Word buttons to export the report.


Troubleshooting
---------------
- CSS not updating after change: hard-refresh the browser (`Ctrl+F5`) or clear cache.
- Upload complaining about missing columns: ensure your file contains columns that map to `CourseId` and `priority_rank` (the app canonicalizes several header variants). If using `UserId`, make sure IDs exist in `systemuser` table.
- Gemini API errors (503): the app retries with exponential backoff; if persistent, verify credentials and quota on the Google Cloud console.


Development notes
-----------------
- Styles are located in `Static/css/style.css` (recent fixes target login layout and button alignment).
- The app contains canonicalization logic for uploaded columns — see `_normkey` and `_CANON_SPEC` in `app.py` to add additional header variants if needed.
- The Excel export uses `pandas` + `openpyxl` to create multi-sheet workbooks on the server.


Contributing
------------
- Fork the repo, create a feature branch, and open a pull request.
- Keep changes minimal and focused; run the app locally to test.


Next steps I can help with
-------------------------
- Run the app locally and verify flows (upload, export, AI report).
- Add a `README` section for environment-specific configuration (example `config.py` snippet).
- Create a `.env.example` or `docker-compose` for easier local setup.


If you'd like, I can now:
- Start the Flask server and walk through a demo flow, or
- Add a `config.example.py` for easier setup, or
- Commit `README.md` to git for you.

