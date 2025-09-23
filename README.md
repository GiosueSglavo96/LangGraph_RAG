# LangGraph_RAG

# 🌐 RAG & LangGraph Streamlit App

Un'applicazione interattiva basata su **Streamlit** che integra **RAG (Retrieval-Augmented Generation)**, **LangGraph**, e tool LLM personalizzati per rispondere a query generiche e specifiche (come il meteo) in maniera dinamica e intelligente.

---

## 🚀 Funzionalità principali

- 📁 Upload e configurazione di documenti (.pdf, .csv, .txt)
- 🧠 Ricerca intelligente basata su:
  - Semantic Search
  - Textual/BM25 Search
  - Hybrid Search
- 🔄 Maximal Marginal Relevance (MMR) supportato
- 🔍 Web search tramite tool `tavily_web_search`
- 🌦️ Previsioni meteo tramite tool `get_weather` con parsing intelligente delle date e della location
- 🧩 Interfaccia utente completa tramite **Streamlit Chat UI**
