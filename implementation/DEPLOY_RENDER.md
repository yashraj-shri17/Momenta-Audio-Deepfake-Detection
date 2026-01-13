
# 🚀 Deploying to Render.com

We have optimized the application for Render's Free Tier (Cloud Optimized).

## Prerequisites
1.  **GitHub Account**
2.  **Render.com Account**

## Step 1: Push Code to GitHub
Since `git` needs your credentials, you must do this manually in your terminal:

```bash
# Initialize Repo
git init
git add .
git commit -m "Initial Commit - Momenta Detector"

# Connect to GitHub (Create a new 'Empty' repo on GitHub first!)
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render
1.  Login to [dashboard.render.com](https://dashboard.render.com).
2.  Click **"New +"** button -> Select **"Web Service"**.
3.  Connect your GitHub Repository.
4.  **Configuration**:
    *   **Name**: `momenta-detector` (or any name)
    *   **Environment**: `Docker` (It should detect this automatically)
    *   **Instance Type**: `Free`
5.  Click **"Create Web Service"**.

## 🕒 Wait for Build
Render will verify the Dockerfile, download the lightweight CPU components, and start the app.
*   *Build Time*: ~3-5 minutes.

Your app will be live at: `https://momenta-detector.onrender.com` (example URL).
