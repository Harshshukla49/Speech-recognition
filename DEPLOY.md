# Streamlit Cloud Deployment

## Status: ✅ Fixed & Ready

**Changes Applied:**
- `runtime.txt`: `python-3.10.12`
- `requirements.txt`: compatible pinned versions (`tensorflow==2.15.0`, `protobuf==3.20.3`, `numpy==1.26.4`)

## Deploy Steps
```bash
git add .
git commit -m "Fix Streamlit Cloud deployment: Python 3.10 + TensorFlow 2.15 deps"
git push origin main
```

**Auto-deploy**: Streamlit Cloud triggers on push to main.

## Verify
1. Check [Streamlit Cloud dashboard](https://share.streamlit.io/)
2. Visit deployed app URL
3. Test upload/recording tabs

## Local Testing
```bash
streamlit run app.py  # http://localhost:8501
```

## Troubleshooting
- Train model: `python run_pipeline.py`
- Check logs in Streamlit Cloud dashboard
