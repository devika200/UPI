# 🔐 UPI Fraud Detection System

AI-powered real-time fraud detection for UPI transactions using Hidden Markov Models and behavioral analysis.

## 🌟 Features

- **Real-time Fraud Detection** - Analyze transactions instantly
- **User Authentication** - Secure JWT-based login system
- **Transaction History** - Track all your transactions
- **ML-Powered Analysis** - Uses Hidden Markov Models for pattern detection
- **Behavioral Learning** - Learns from your transaction patterns
- **Risk Scoring** - Provides detailed risk assessment

## 🛠️ Tech Stack

### Frontend
- React 18
- Vite
- Axios
- React Router

### Backend
- Flask (Python)
- Flask-JWT-Extended
- Flask-CORS
- MongoDB (PyMongo)

### Machine Learning
- Hidden Markov Models (HMM)
- scikit-learn
- pandas, numpy
- hmmlearn

## 📦 Project Structure

```
UPI/
├── Backend/
│   ├── mainapp.py              # Main Flask application
│   ├── model.py                # HMM model implementation
│   ├── requirements.txt        # Python dependencies
│   ├── hmm_fraud_model.pkl     # Trained model
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── pages/              # React pages
│   │   ├── components/         # React components
│   │   └── ...
│   ├── package.json
│   └── ...
└── DEPLOYMENT_GUIDE.md         # Deployment instructions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB (local or Atlas)

### Backend Setup
```bash
cd Backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python mainapp.py
```

Backend runs on: `http://localhost:5000`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:3000`

## 🌐 Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete deployment instructions.

**Quick Deploy:**
- Frontend → Netlify
- Backend → Render
- Database → MongoDB Atlas

All free tier available!

## 📊 How It Works

1. **User Registration/Login** - Secure authentication with JWT
2. **Transaction Submission** - User submits transaction details
3. **Feature Extraction** - System calculates 20+ features from transaction
4. **Historical Analysis** - Compares with user's past behavior
5. **ML Prediction** - HMM model analyzes patterns
6. **Risk Assessment** - Returns fraud score and risk factors
7. **History Tracking** - Saves for future learning

## 🔒 Security Features

- Bcrypt password hashing
- JWT token authentication
- CORS protection
- Input validation
- SQL injection prevention
- XSS protection

## 📈 ML Model Details

**Algorithm**: Auto-Regressive Hidden Markov Model (AR-HMM)

**Features Used** (20 total):
- Transaction amount
- Amount difference from average
- Transaction frequency
- Time anomaly score
- Recipient statistics
- Risk score
- Time features (hour, day)
- Location hashes (10 features)

**Training**: Model trained on balanced dataset with fraud/legitimate transactions

## 🎯 API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - User login

### Fraud Detection
- `POST /api/check_fraud` - Check transaction for fraud (requires JWT)
- `GET /api/history` - Get user's transaction history (requires JWT)

### Admin
- `GET /health` - Health check
- `GET /api/admin/stats` - Database statistics

## 🧪 Testing

```bash
# Backend tests
cd Backend
python test_login.py

# Check MongoDB data
python check_mongodb.py
```

## 📝 Environment Variables

### Backend
```
MONGODB_URI=mongodb://localhost:27017/
JWT_SECRET_KEY=your-secret-key
PORT=5000
```

### Frontend
```
VITE_API_URL=http://localhost:5000
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Hidden Markov Model implementation using hmmlearn
- Flask framework for backend API
- React for frontend UI
- MongoDB for data storage

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Review [GITHUB_CHECKLIST.md](GITHUB_CHECKLIST.md)

---

Made with ❤️ for secure UPI transactions
