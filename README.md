<h3>Neuro-Symbolic AI for Legal Reasoning</h3>
This is a local, privacy-first AI system that combines Symbolic AI (Rule-Based Statutory Search) with Neural Reasoning (DeepSeek-R1) to generate professional legal opinions.

Unlike standard RAG (Retrieval-Augmented Generation) systems that merely summarize search results, LegalBrain uses a recursive logic engine to "read" case law, extract implied sections, and cross-reference them with a statutory database before forming an opinion.

<h3>Core Architecture</h3>
This system uses a Neuro-Symbolic approach:

Symbolic Layer (The "Lawyer"): Rigid rule-based retrieval using FAISS for Statutes (Sections/Acts) and Weaviate for Case Law.

Neural Layer (The "Thinker"): DeepSeek-R1 (via Ollama) performs Chain-of-Thought reasoning to resolve conflicts between rigid statutes and nuanced case precedents.
