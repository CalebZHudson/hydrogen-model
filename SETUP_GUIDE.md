# Putting the app on GitHub (about 10 minutes)

## A. GitHub Pages: runs in the visitor's browser, only a GitHub account needed

1. Sign in at github.com (create a free account if you don't have one).
2. Click **+** (top right) > **New repository**.
   - Name: `hydrogen-model` (any name works; it becomes part of the web address).
   - Choose **Public**. Leave everything else as it is. Click **Create repository**.
3. On the new repo's page click **uploading an existing file**.
   Drag in all the files from this folder (`index.html`, `app.py`, `h2_model.py`, `requirements.txt`, `README.md`, `SETUP_GUIDE.md`).
   (Drag the files themselves, not the folder.) Click **Commit changes**.
4. Go to **Settings** (top of the repo) > **Pages** (left menu).
   - Under *Build and deployment*, Source: **Deploy from a branch**.
   - Branch: **main**, folder **/ (root)**. Click **Save**.
5. Wait 1 to 2 minutes and refresh the Pages settings page. It shows
   *Your site is live at* `https://<your-username>.github.io/hydrogen-model/`.
6. Open the link. The first load takes 20 to 40 seconds while Python downloads into the browser; then press **Solve**.

Anyone with the link can use it. Visitors do not need to sign in.

**Updating:** open a file on GitHub, click the pencil icon, edit, **Commit changes**. The site updates within a minute or two.

## B. Streamlit Community Cloud: runs on a server, faster first load

Use the same repo from steps 1 to 3 above.

1. Go to https://share.streamlit.io and choose **Continue with GitHub**; allow access.
2. Click **Create app** > **Deploy a public app from GitHub**.
3. Repository: `<your-username>/hydrogen-model`; Branch: `main`; Main file path: `app.py`. Click **Deploy**.
4. After 2 to 3 minutes you get a link like `https://hydrogen-model.streamlit.app`.

Visitors do not need to sign in. If nobody opens it for a few days it sleeps; the next visitor clicks
*Yes, get this app back up* and waits about 30 seconds.

## If something goes wrong

- **GitHub Pages shows a 404:** wait two more minutes, and check that `index.html` is at the top level of the repo
  (not inside a folder).
- **Page stays on "Loading Python" for more than 2 minutes:** open the browser console (F12 > Console) and send the red
  error message. University or work networks sometimes block cdn.jsdelivr.net; try another network.
- **Streamlit Cloud error during install:** check that `requirements.txt` is in the repo.
