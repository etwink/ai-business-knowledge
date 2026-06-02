# Document Analysis System

A comprehensive Python application for analyzing collections of documents (COBOL code, Word documents, Excel sheets, HTML files, etc.) to generate process documentation, identify gaps, and suggest clarification questions using Azure OpenAI.

## Features

### 📚 Document Support
- **COBOL Files**: `.cob`, `.cbl`, `.cic`, `.cpy`, `.mps`, `.src`, `.ct1`, `.jcv`, `.prv`
- **Office Documents**: `.docx`, `.xlsx`
- **Web Content**: `.html`, `.htm`
- **Code & Text**: `.py`, `.txt`
- **Documents**: `.pdf`

### 🤖 LLM-Powered Analysis
- Integration with Azure OpenAI for intelligent document analysis
- Automatic document summarization
- Relationship and dependency identification
- Process flow analysis

### 📖 Process Document Generation
Automatically generates comprehensive process documentation that includes:
- Overview of all documents
- Integrated processes and workflows
- System dependencies and relationships
- Data flow mappings
- Decision points
- Systems and components inventory

### ⚠️ Gap Analysis
Identifies missing information:
- Missing process steps
- Undefined technical dependencies
- Incomplete data transformations
- Missing system integrations
- Error handling gaps
- Security considerations
- Resource requirements

### ❓ Smart Clarification Questions
Generates targeted questions to enhance documentation:
- Questions addressing identified gaps
- Technical clarification requests
- Edge case and error scenario questions
- Data flow and transformation questions

### 🎨 Interactive UI
Built with Streamlit for easy navigation and document management:
- Drag-and-drop file upload
- Step-by-step workflow
- Real-time progress indicators
- Export capabilities (Markdown)

### 💾 Session Management & Persistence
- Save analysis sessions to avoid re-processing documents
- Load previous sessions to continue work
- Automatic saving at each step
- Version control for documents

### 🔄 Interactive Refinement
- Ask users clarification questions directly in the UI
- Collect detailed answers to fill knowledge gaps
- Automatically refine documents based on user input
- Generate improved document versions
- Export both original and refined versions

## Project Structure

```
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .env                           # Environment variables (create from .env.example)
│
├── src/
│   ├── document_loaders/          # File format handlers
│   │   ├── __init__.py
│   │   ├── base.py               # Base loader class
│   │   └── loaders.py            # Format-specific loaders
│   │
│   ├── llm_integration/           # Azure OpenAI integration
│   │   ├── __init__.py
│   │   └── azure_client.py       # LLM client and prompt builders
│   │
│   ├── analyzers/                # Analysis engines
│   │   ├── __init__.py
│   │   └── document_analyzer.py  # Analysis logic
│   │
│   ├── storage/                  # Persistence & refinement
│   │   ├── __init__.py
│   │   ├── storage.py            # Session storage and loading
│   │   └── document_refinement.py # Document improvement logic
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── file_utils.py         # File handling utilities
│   │
│   └── __init__.py
│
├── sample_documents/              # Sample documents for testing
├── analysis_sessions/             # Saved analysis sessions (auto-created)
│
└── cobol_dependency_analyzer.py   # Existing COBOL analyzer (optional integration)
```

## Installation

### 1. Clone/Setup the Project
```bash
cd "AI Business Knowledge"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Azure OpenAI
1. Copy `.env.example` to `.env`
2. Add your Azure OpenAI credentials:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Usage

### Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload Documents** → Upload documents in various formats
2. **Analyze** → System analyzes each document and extracts key information (automatically saved)
3. **Review Process Document** → View the automatically generated process documentation
4. **Gap Analysis** → Identify missing information and gaps
5. **Clarification Questions** → Review AI-generated questions to enhance documentation
6. **Answer Questions** → Provide detailed answers to fill knowledge gaps
7. **Refined Document** → Review improved documentation based on your answers

### Session Management

- Each analysis is saved as a session with a unique name
- Sessions are persisted to disk in `analysis_sessions/` directory
- Load previous sessions to continue work without re-analyzing documents
- Original and refined documents are versioned (v1, v2_refined, etc.)

### Example Workflows

#### Quick One-Time Analysis
```
1. Upload Documents → 2. Analyze → 3. Review Process Document → Export
```

#### Comprehensive Analysis with Refinement
```
1. Upload Documents → 2. Analyze → 3. Review Process Document
→ 4. Gap Analysis → 5. Clarification Questions → 6. Answer Questions
→ 7. Refined Document → Export refined version
```

#### Continue Previous Work
```
1. Load Session from sidebar → 2. Review saved analysis
→ 3. Answer additional questions → 4. Generate refined document
```

### Example Use Cases

#### COBOL System Documentation
- Upload COBOL source files, COPY books, and related documentation
- System generates comprehensive process documentation
- Identifies undefined dependencies and missing error handling

#### Enterprise Process Mapping
- Upload Word documents describing processes
- Include Excel spreadsheets with data mappings
- System integrates all information into unified documentation

#### Technical Architecture Analysis
- Upload architecture documents, code files, and specifications
- System generates dependency and component analysis
- Identifies gaps in documentation

## Configuration

### Environment Variables

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_DEPLOYMENT_NAME=deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Document Processing
MAX_DOCUMENT_SIZE_MB=50
SUPPORTED_FORMATS=.py,.cobol,.cbl,.docx,.xlsx,.html,...
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
```

### File Size Limits
Default: 50MB per document (configurable in `.env`)

## Key Modules

### Document Loaders (`src/document_loaders/`)
- Handles multiple file formats
- Preserves document structure (tables, formatting)
- Supports COBOL fixed/free format detection

### LLM Integration (`src/llm_integration/`)
- `AzureLLMClient`: Manages Azure OpenAI API calls
- `PromptBuilder`: Constructs specialized prompts for different analysis tasks

### Analyzers (`src/analyzers/`)
- `DocumentAnalyzer`: Analyzes individual documents
- `ProcessDocumentBuilder`: Creates comprehensive process documentation
- `GapAnalyzer`: Identifies gaps and missing information
- `ClarificationQuestionGenerator`: Generates targeted questions

## Development

### Adding New File Format Support

1. Create a loader class in `src/document_loaders/loaders.py`:
```python
class CustomLoader(BaseDocumentLoader):
    def load(self) -> DocumentContent:
        # Implementation
        pass
```

2. Register in `get_loader()` factory function

3. Update `SUPPORTED_FORMATS` in `config.py`

### Customizing Analysis Prompts

Edit `PromptBuilder` methods in `src/llm_integration/azure_client.py` to modify:
- Document summarization
- Process document generation
- Gap analysis
- Question generation

## Troubleshooting

### Azure OpenAI Connection Issues
- Verify credentials in `.env` file
- Check endpoint URL format
- Ensure API version is correct

### Document Loading Errors
- Verify file format is in `SUPPORTED_FORMATS`
- Check file is not corrupted
- Ensure file size is under `MAX_DOCUMENT_SIZE_MB`

### Memory Issues with Large Documents
- Reduce `CHUNK_SIZE` in `.env`
- Process documents in smaller batches
- Increase system RAM

## Future Enhancements

- [ ] PDF support with text extraction
- [ ] Diagram generation from processes
- [ ] Version control for process documents
- [ ] Collaborative editing and comments
- [ ] Integration with knowledge bases
- [ ] Multi-language support
- [ ] Custom prompt templates
- [ ] Results caching

## License

[Specify your license here]

## Support

For issues or questions, please create an issue in the repository.
