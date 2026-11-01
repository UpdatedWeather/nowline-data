# Nowline data

A free GitHub Actions job that prepares everything the Nowline app shows, so the phone only
downloads small, ready-to-draw files:

| File | What | Size |
|---|---|---|
| `latest.json` | Live manifest: radar frames, satellite frames, lightning | ~3 KB |
| `live/r_<time>.png` | Radar (EUMETNET OPERA), reprojected for the app, value = (dBZ+32)×2 | 30–150 KB |
| `live/s_<time>.jpg` | EUMETSAT GeoColour satellite | ~150 KB |
| `live/lightning.json` | Individual strikes, last 3 h | small |
| `events/index.json` | History feed with thumbnails and peak numbers | ~10 KB |
| `events/<id>/…` | Each event's radar, satellite, lightning and facts | 3–8 MB |

## Setup (about 5 minutes, once)

1. On github.com create a **new public repository** called `nowline-data`
   (public is what makes Actions minutes and Pages free).
2. Upload this folder's files: **Add file → Upload files**, drag in `nowline`, `events.json`,
   `requirements.txt`, `README.md`, **Commit**.
3. Add the workflow (Finder hides the `.github` folder, so create it on the website):
   **Add file → Create new file**, name it `.github/workflows/nowline-data.yml`, paste the contents
   of `workflow-nowline-data.yml`, **Commit**.
4. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
5. **Actions** tab → enable workflows if asked → open **Nowline data** → **Run workflow**.
6. When the run finishes (first run ~10–15 min), your feed lives at
   `https://<your-github-username>.github.io/nowline-data/latest.json`.
7. Tell Claude your GitHub username so the app points at it (it's one line in `CloudFeed.swift`).

History events are built a few at a time (10 minutes of work per run) and appear in the app's
history feed as they finish — all 13 are usually ready within the first hour.

## Notes

- GitHub runs scheduled jobs on a best-effort basis (often every 10–20 minutes). The app tops up
  with the newest radar frames directly, so "live" stays live.
- GitHub pauses scheduled workflows in a repository with no activity for 60 days — any commit
  (or re-enabling in the Actions tab) restarts it.
- Data: EUMETNET OPERA (CC BY 4.0), Met Office (Crown copyright), EUMETSAT, Open-Meteo (CC BY 4.0),
  place names © OpenStreetMap contributors.
- GitHub Pages is fine for testing and a beta. For a public App Store release, the same files can
  move to Cloudflare R2 (free tier, needs a card on file) without changing the app beyond the URL.
