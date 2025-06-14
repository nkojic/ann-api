# ANN Flask API

## Pokretanje lokalno:
1. Kreiraj virtualno okruženje:
   python -m venv venv
2. Aktiviraj:
   - Windows: venv\Scripts\activate
   - Linux/macOS: source venv/bin/activate
3. Instaliraj zavisnosti:
   pip install -r requirements.txt
4. Pokreni server:
   python app.py

## Deploy na Render:
1. Commit-uj ovaj folder na GitHub
2. Na https://render.com klikni "New Web Service"
3. Poveži svoj GitHub repozitorijum
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python app.py`
