# Relazione Tecnica: Architettura e Pipeline NLP del Progetto ALLMond

## 1. Introduzione
Questa relazione descrive l'architettura tecnica e la pipeline di Natural Language Processing (NLP) implementata per il progetto "ALLMond". Il sistema è stato progettato per rispondere a domande relative all'Università Politecnica delle Marche (UnivPM) utilizzando un approccio ibrido che combina tecniche di machine learning tradizionale per il filtraggio e Small Language Models (SLM) per la generazione di risposte, in linea con le moderne tendenze verso l'IA efficiente e "agentica" discusse nel paper di NVIDIA *"Small Language Models are the Future of Agentic AI"*.

## 2. Architettura del Sistema
Il progetto adotta un'architettura a microservizi orchestrata tramite Docker Compose, garantendo modularità, scalabilità e isolamento delle componenti.

Le componenti principali sono:
*   **Application (Frontend/BFF)**: Un server Node.js che gestisce l'interfaccia utente (UI) e funge da *Backend-for-Frontend*, inoltrando le richieste dell'utente ai servizi interni.
*   **Inference Service (Core NLP)**: Un servizio Python basato su FastAPI che implementa la logica decisionale e la pipeline NLP.
*   **Crawler Service**: Un microservizio dedicato al reperimento di informazioni aggiornate dal web, basato sulla libreria `crawl4ai`.
*   **Ollama Service**: Un'istanza dedicata per l'hosting e l'inferenza locale dei modelli di linguaggio (SLM).

## 3. Pipeline NLP
La pipeline di elaborazione del linguaggio naturale è strutturata in fasi sequenziali per ottimizzare le risorse e garantire la pertinenza delle risposte.

### 3.1 Preprocessing del Testo
Ogni input testuale viene processato da un modulo personalizzato `TextPreprocessor` (`utils/text_preprocessing.py`) che esegue:
1.  **Pulizia**: Rimozione di URL, caratteri speciali e normalizzazione (lowercase).
2.  **Tokenizzazione**: Suddivisione del testo in token utilizzando le risorse di NLTK per la lingua italiana.
3.  **Stemming**: Riduzione delle parole alla loro radice (utilizzando `SnowballStemmer`), preferito alla lemmatizzazione per la lingua italiana in questo contesto per robustezza e velocità.

### 3.2 Classificazione di Pertinenza
Prima di attivare i modelli generativi, il sistema verifica se la domanda è pertinente al dominio universitario.
*   **Modello**: Logistic Regression (implementato con `scikit-learn`).
*   **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) limitato alle 1000 feature più rilevanti.
*   **Funzionamento**: Se il classificatore predice la classe `0` (non pertinente), il sistema restituisce immediatamente una risposta predefinita, evitando l'uso costoso di crawler e LLM.

### 3.3 Retrieval-Augmented Generation (RAG)
Se la domanda è pertinente, viene attivata la pipeline RAG:
1.  **Crawling**: Il servizio `crawler` recupera il contenuto testuale aggiornato da una lista di URL predefiniti dell'UnivPM (`https://www.univpm.it/Entra/`, ecc.).
2.  **Context Aggregation**: I contenuti recuperati vengono concatenati.

### 3.4 Summarization & Generation con SLM
Questa fase implementa la visione "agentica" utilizzando modelli piccoli e specializzati (SLM) tramite Ollama.

*   **Fase 1: Summarization (Compressione)**
    *   **Modello**: `qwen3:0.6b` (Modello estremamente leggero).
    *   **Obiettivo**: Sintetizzare il vasto contenuto recuperato dal crawler per estrarre solo le informazioni chiave e rientrare nella context window del modello successivo.
    *   **Prompt**: "Riassumi il seguente testo rimuovendo le parti inutili..."

*   **Fase 2: Question Answering (Ragionamento)**
    *   **Modello**: `qwen3:1.7b` (Modello leggermente più capace).
    *   **Obiettivo**: Generare la risposta finale all'utente basandosi *esclusivamente* sul riassunto fornito.
    *   **Prompt**: "Rispondi alla domanda basandoti solo sul contesto fornito..."

## 4. Analisi dell'Uso degli SLM (Small Language Models)
L'implementazione riflette fedelmente i principi esposti nel paper *"Small Language Models are the Future of Agentic AI"*.

### 4.1 Efficienza e Costi
Invece di utilizzare un unico, enorme LLM generalista (es. GPT-4 o Llama-3-70B) per tutte le fasi, il sistema "catena" modelli specializzati di dimensioni ridotte (< 2B parametri).
*   **Vantaggio**: I modelli `qwen3:0.6b` e `1.7b` possono girare interamente su CPU o GPU consumer con latenze minime.
*   **Specializzazione**: Il modello da 0.6b è usato come "operaio" per la compressione del testo (task meccanico), mentre il modello da 1.7b è usato per l'interazione finale (task che richiede maggiore coerenza sintattica).

### 4.2 Architettura Modulare
Come suggerito nel paper, l'agente non è monolitico ma composto da *sub-agenti* (o fasi) che invocano il modello più adatto al compito specifico. Questo riduce il *time-to-first-token* e l'impronta energetica complessiva dell'applicazione.

## 5. Implementazione Tecnica
Il cuore del sistema risiede in `src/inference/main.py`.
*   **Robustezza**: Sono presenti meccanismi di fallback (es. se i modelli non sono caricati, o se il crawler fallisce).
*   **Training**: Lo script `scripts/train_pipeline.py` garantisce che il classificatore possa essere ri-addestrato facilmente su nuovi dati, mantenendo il filtro di pertinenza sempre aggiornato.
*   **Librerie Chiave**:
    *   `FastAPI`: Per l'API REST asincrona.
    *   `crawl4ai`: Per un crawling moderno ed efficiente (markdown-ready).
    *   `scikit-learn`: Per la parte di Machine Learning classico.
    *   `Ollama`: Per l'astrazione e l'orchestrazione degli SLM locali.

## 6. Conclusioni
La soluzione ALLMond dimostra come sia possibile costruire un sistema di QA efficace e moderno senza dipendere da costose API esterne o hardware enterprise. L'uso intelligente di una pipeline ibrida (ML Classico + RAG + SLM a cascata) garantisce un ottimo bilanciamento tra precisione, velocità e costi di esercizio.
