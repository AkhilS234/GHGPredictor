# GHGPredictor

This model predicts **GHG Emissions per Vehicle** for multiple vehicle companies by combining:
- **Historical data from 2020–2022**, and
- **Sentiment extracted from the company’s 2023 sustainability report (.txt)**

The app also computes **R²** to quantify how well predictions align with the **actual 2023** GHG/vehicle values.

---

## Quick start

```bash
# clone & enter
git clone https://github.com/<your-username>/GHGPredictor.git
cd GHGPredictor

# virtual env
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# run
python app.py

```

### App URL

- **Local:** [http://127.0.0.1:5000/](http://127.0.0.1:5000/)  
- **Alt local:** [http://localhost:5000/](http://localhost:5000/)  
- **Live (when deployed):** https://<your-app-name>.onrender.com/
