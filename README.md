# AI-Chatbot-for-Technical-Support-Based-on-PDF-Documents
Group Project for the EESTech Challenge Patras 2025 Local Round. Implemented an AI Chatbot for Technical Support Based on PDF Documents using ChromaDB, FLAN-T5 and Streamlit.
After splitting the content, located in the pdf files found in \knowledge, into chunks , Vectors and Embeddings are stored in a ChromaDB database. 
After feeding these data to FLAN-T5 Language Model, a user may ask questions through a Streamlit-based User Interface, and get back not only the retrieved answer but also the exact used sources. 
With this project we achieved first place in the competition!
