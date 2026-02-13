# 🚀 Deployment Guide - UPI Fraud Detection

## Architecture
- **Frontend**: Netlify (Static hosting)
- **Backend**: Render (Python API)
- **Database**: MongoDB Atlas (Cloud database)

---

## Step 1: Deploy Database (MongoDB Atlas)

### 1.1 Create MongoDB Atlas Account
1. Go to https://www.mongodb.com/cloud/atlas/register
2. Sign up for free
3. Create a new cluster (Free M0 tier)

### 1.2 Setup Database
1. Click "Connect" on your cluster
2. Add your IP address (or use 0.0.0.0/0 for all IPs)
3. Create database user with username/password
4. Get connection string:
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/
   ```

### 1.3 Save Connection String
You'll need this for backend deployment.

---

## Step 2: Deploy Backend (Render)

### 2.1 Prepare Backend
Already done! Files created:
- ✅ `Procfile` - Tells Render how to run the app
- ✅ `requirements.txt` - Updated with all dependencies
- ✅ `render.yaml` - Render configuration
- ✅ `runtime.txt` - Python version

### 2.2 Push to GitHub
```bash
cd UPI
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/upi-fraud-detection.git
git push -u origin main
```

### 2.3 Deploy on Render
1. Go to https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: upi-fraud-detection-api
   - **Root Directory**: Backend
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn mainapp:app`
   - **Instance Type**: Free

### 2.4 Add Environment Variables
In Render dashboard, add:
```
MONGODB_URI = mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/
JWT_SECRET_KEY = your-super-secret-random-string-here
PORT = 10000
```

### 2.5 Deploy
Click "Create Web Service" and wait for deployment.

Your backend URL will be: `https://upi-fraud-detection-api.onrender.com`

---

## Step 3: Deploy Frontend (Netlify)

### 3.1 Update API URL
Edit `frontend/.env.production`:
```
VITE_API_URL=https://upi-fraud-detection-api.onrender.com
```

### 3.2 Build Frontend Locally (Test)
```bash
cd frontend
npm install
npm run build
```

### 3.3 Deploy to Netlify

#### Option A: Drag & Drop (Easiest)
1. Go to https://app.netlify.com
2. Sign up/Login
3. Drag the `frontend/dist` folder to Netlify
4. Done!

#### Option B: GitHub (Recommended)
1. Push code to GitHub (if not done)
2. Go to https://app.netlify.com
3. Click "Add new site" → "Import from Git"
4. Connect GitHub repository
5. Configure:
   - **Base directory**: frontend
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
   - **Environment variables**:
     ```
     VITE_API_URL = https://upi-fraud-detection-api.onrender.com
     ```
6. Click "Deploy site"

Your frontend URL will be: `https://your-site-name.netlify.app`

---

## Step 4: Update CORS

Update `mainapp.py` to allow your Netlify domain:

```python
from flask_cors import CORS

# Update CORS to allow your Netlify domain
CORS(app, origins=[
    "http://localhost:3000",
    "https://your-site-name.netlify.app"
])
```

Redeploy backend on Render.

---

## Step 5: Test Deployment

### 5.1 Test Backend
```bash
curl https://upi-fraud-detection-api.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true
}
```

### 5.2 Test Frontend
1. Go to `https://your-site-name.netlify.app`
2. Register a new user
3. Login
4. Check a transaction
5. View history

---

## Alternative Deployment Options

### Backend Alternatives

#### Railway (Similar to Render)
1. Go to https://railway.app
2. Connect GitHub
3. Deploy from repo
4. Add environment variables
5. Done!

#### Heroku (Paid)
1. Install Heroku CLI
2. `heroku create upi-fraud-api`
3. `git push heroku main`
4. `heroku config:set MONGODB_URI=...`

#### PythonAnywhere (Free)
1. Go to https://www.pythonanywhere.com
2. Upload code
3. Configure WSGI
4. Add environment variables

### Frontend Alternatives

#### Vercel
1. Go to https://vercel.com
2. Import GitHub repo
3. Set base directory to `frontend`
4. Deploy

#### GitHub Pages
1. Build: `npm run build`
2. Push `dist` folder to `gh-pages` branch
3. Enable GitHub Pages in repo settings

---

## Environment Variables Summary

### Backend (Render/Railway/Heroku)
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
JWT_SECRET_KEY=your-secret-key-here
PORT=10000
```

### Frontend (Netlify/Vercel)
```
VITE_API_URL=https://your-backend-url.onrender.com
```

---

## Troubleshooting

### Backend Issues

**Problem**: "Application failed to start"
- Check logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Check Python version matches `runtime.txt`

**Problem**: "MongoDB connection failed"
- Verify MONGODB_URI is correct
- Check MongoDB Atlas IP whitelist (use 0.0.0.0/0)
- Verify database user credentials

**Problem**: "Model not loaded"
- Upload `hmm_fraud_model.pkl` to GitHub
- Check file size (GitHub has 100MB limit)
- Consider using Git LFS for large files

### Frontend Issues

**Problem**: "API calls failing"
- Check VITE_API_URL is correct
- Verify CORS is configured on backend
- Check browser console for errors

**Problem**: "Build failed"
- Run `npm install` locally
- Fix any TypeScript/ESLint errors
- Check `package.json` scripts

---

## Cost Breakdown

### Free Tier (Recommended for Testing)
- **MongoDB Atlas**: Free (512MB storage)
- **Render**: Free (750 hours/month, sleeps after 15 min inactivity)
- **Netlify**: Free (100GB bandwidth/month)
- **Total**: $0/month

### Paid Tier (Production)
- **MongoDB Atlas**: $9/month (2GB storage)
- **Render**: $7/month (always on, 512MB RAM)
- **Netlify**: Free (sufficient for most apps)
- **Total**: ~$16/month

---

## Post-Deployment

### 1. Custom Domain (Optional)
- Buy domain from Namecheap/GoDaddy
- Add to Netlify: Settings → Domain management
- Update DNS records

### 2. SSL Certificate
- Automatically provided by Netlify and Render
- Your sites will have HTTPS

### 3. Monitoring
- Render: Built-in logs and metrics
- Netlify: Analytics dashboard
- MongoDB Atlas: Performance monitoring

### 4. Continuous Deployment
- Push to GitHub → Auto-deploys to Render & Netlify
- No manual deployment needed!

---

## Quick Deploy Commands

```bash
# 1. Push to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Backend deploys automatically on Render

# 3. Frontend deploys automatically on Netlify

# Done! 🎉
```

---

## Support

If you encounter issues:
1. Check Render logs: Dashboard → Logs
2. Check Netlify logs: Deploys → Deploy log
3. Check MongoDB Atlas: Metrics → Performance
4. Test locally first: `npm run dev` and `python mainapp.py`

---

## Next Steps

1. ✅ Deploy database (MongoDB Atlas)
2. ✅ Deploy backend (Render)
3. ✅ Deploy frontend (Netlify)
4. ✅ Test everything
5. 🎉 Share your app!

Your app will be live at:
- Frontend: `https://your-app.netlify.app`
- Backend: `https://your-api.onrender.com`
