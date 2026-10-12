# ProductLens

RAG-powered product intelligence assistant for ShopSmart India.

## Setup

1. Open PowerShell and move to the project folder:

```powershell
Set-Location "C:\Users\2531797\OneDrive - Cognizant\Desktop\capstone 2\ProductLens"
```

2. Create and activate a virtual environment (example: `new_venv`):

```powershell
python -m venv new_venv
.\new_venv\Scripts\activate
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Add your OpenAI key to `.env`:

```env
OPENAI_API_KEY=your_key_here
```

## Run

Run the core RAG demo queries:

```powershell
python main.py
```

Run the RAGAS evaluation:

```powershell
python evaluation.py
```

Run the Streamlit UI (Shopper, Seller, Support Agent personas):

```powershell
streamlit run streamlit_app.py
```

## Notes

- `ragas==0.3.1` and `langchain-community==0.3.31` are pinned for compatibility.
- `pillow` is required because RAGAS imports `PIL` during package initialization.

## Quick Checks

Validate RAGAS import:

```powershell
python -c "import ragas; print('RAGAS import OK:', ragas.__version__)"
```

Validate evaluation module import:

```powershell
python -c "import evaluation; print('evaluation module import OK')"
```

## Troubleshooting

### 1. `ModuleNotFoundError` for `ragas` or `PIL`

- Make sure the virtual environment is activated.
- Reinstall dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 2. Streamlit returns: "I could not find enough information"

- This can happen when a metadata filter returns no chunks.
- Restart Streamlit after code changes:

```powershell
streamlit run streamlit_app.py
```

- In the UI, use **Clear Conversation** once and run the query again.

### 3. Command not found (for `Set-Location` or python path issues)

- Ensure you are running commands in PowerShell.
- Always run commands from the project folder shown in Setup step 1.
