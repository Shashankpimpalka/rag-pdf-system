rag-pdf-system/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py # PDF processing
│   │   ├── vector_store.py  # Vector DB operations
│   │   └── qa_service.py    # Question answering
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── uploads/                 # Temporary PDF storage
├── requirements.txt
├── .env
└── README.md