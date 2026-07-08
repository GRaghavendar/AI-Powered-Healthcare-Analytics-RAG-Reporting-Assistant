# GitHub Publishing Checklist

Use this checklist when publishing the project for the first time.

## 1. Confirm The Project Is Clean

Make sure the repository does not contain:

- `.venv/`
- `.env`
- API keys or tokens
- local model cache files
- personal computer paths
- real patient data
- PHI
- private employer data

The included `.gitignore` excludes the common local files.

## 2. Create The GitHub Repository

1. Go to GitHub.
2. Click `New repository`.
3. Name the repository `ai-powered-healthcare-analytics-rag-reporting-assistant`.
4. Select `Public` if this is meant to be visible to recruiters or hiring managers.
5. Do not add a README, license, or `.gitignore` from GitHub.
6. Click `Create repository`.

## 3. Initialize Git Locally

Open PowerShell or Terminal inside the project folder and run:

```bash
git init
git add .
git commit -m "Initial AI-powered healthcare analytics RAG reporting assistant"
git branch -M main
```

## 4. Connect And Push

Replace `<your-github-username>` with your GitHub username:

```bash
git remote add origin https://github.com/<your-github-username>/ai-powered-healthcare-analytics-rag-reporting-assistant.git
git push -u origin main
```

## 5. Review On GitHub

After pushing, open the repository page and check:

- The README displays correctly.
- Dashboard screenshots are visible.
- Source folders are visible.
- `.venv/` is not uploaded.
- `.env` is not uploaded.
- No personal local file paths are visible.
- No real data or PHI is present.

## 6. If Git Is Not Installed

If `git --version` does not work, install Git for Windows from the official Git website, restart PowerShell, and try again.
