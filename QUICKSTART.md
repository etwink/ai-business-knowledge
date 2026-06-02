# Quick Start Guide

Get your Document Analysis System up and running in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- Azure OpenAI account with API access
- Windows, macOS, or Linux

## Step 1: Setup

```bash
# Navigate to project directory
cd "AI Business Knowledge"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Azure OpenAI

1. Open `.env` file in the project root
2. Add your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_actual_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Where to get these values:
- **API Key**: Azure Portal → OpenAI resource → Keys and Endpoints
- **Endpoint**: Azure Portal → OpenAI resource → Keys and Endpoints
- **Deployment Name**: Azure OpenAI Studio → Deployments

## Step 3: Start the Application

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

If not, manually navigate to that URL.

## Step 4: Upload Documents

1. Click "Upload Documents" in the sidebar
2. Either:
   - **Upload files**: Click "Choose files" to upload from your computer
   - **Sample documents**: Place files in the `sample_documents` folder and use the "Load from Folder" option

Supported formats:
- COBOL: `.cob`, `.cbl`, `.cic`, `.cpy`, `.mps`, `.src`, `.ct1`, `.jcv`, `.prv`
- Office: `.docx`, `.xlsx`
- Web: `.html`
- Other: `.txt`, `.py`

## Step 5: Run Analysis

1. Go to "Analyze" step
2. Click "Start Analysis" button
3. Wait for all documents to be processed (progress bar shows status)
4. Review individual document summaries
5. ✅ **Analysis is automatically saved to the session**

## Step 6: Generate Process Document

1. Go to "Review Process Document" step
2. Click "Generate Process Document" button
3. Review the integrated process documentation
4. ✅ **Document is automatically saved**

## Step 7: Identify Gaps

1. Go to "Gap Analysis" step
2. Click "Analyze Gaps" button
3. Review identified gaps across 7 categories
4. ✅ **Gap analysis is automatically saved**

## Step 8: Review Questions

1. Go to "Questions" step
2. Click "Generate Clarification Questions" button
3. Review AI-generated questions that would enhance documentation
4. ✅ **Questions are automatically saved**

## Step 9: Answer Questions (NEW!)

1. Go to "Answer Questions" step
2. Provide detailed answers to each clarification question
3. Click "Submit Answers" button
4. ✅ **Your answers are automatically saved**

## Step 10: View Refined Document (NEW!)

1. Go to "Refined Document" step
2. Click "Generate Refined Document" button
3. System automatically incorporates your answers into the documentation
4. Review the improved sections
5. Export the refined version
6. ✅ **Refined document is automatically saved as v2_refined**

## Step 11: Save Your Work

Sessions are automatically saved, but you can:
- **Continue later**: Use "Load Session" to reload your work
- **Export**: Download Markdown files for sharing
- **Create new versions**: Generate refined documents based on feedback

## Resuming Previous Work

Instead of re-analyzing everything:

1. Open the application
2. Go to "Session Management" in sidebar
3. Select a previous session from the list
4. Click "Load Session" button
5. ✅ All your previous analysis, documents, and answers are restored
6. Continue from where you left off or generate refined documents

## Complete Workflow Example

## Troubleshooting

### Issue: "API key not configured"

**Solution**: Check `.env` file has valid Azure OpenAI credentials

```bash
# Verify .env exists
ls .env  # macOS/Linux
dir .env  # Windows
```

### Issue: "Module not found" errors

**Solution**: Make sure virtual environment is activated and dependencies are installed

```bash
# Activate venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Streamlit not starting

**Solution**: Ensure you're in the project root directory

```bash
# Check current location
pwd  # macOS/Linux
cd  # Windows

# Should be in "AI Business Knowledge" folder
cd "AI Business Knowledge"
streamlit run app.py
```

### Issue: Analysis is slow or times out

**Solution**: Try with fewer or smaller documents first
- Remove very large files (>50MB)
- Start with 1-2 documents
- Check internet connection to Azure

## Complete Workflow Example

### End-to-End Analysis with Refinement

```
Day 1:
├─ Upload COBOL files + Word docs + Excel sheets
├─ Run Analysis (documents analyzed, results saved)
├─ Generate Process Document (saved automatically)
├─ Analyze Gaps (identified and saved)
└─ Review Clarification Questions

Day 2 (Stakeholder Review):
├─ Load previous session (instantly reloaded)
├─ Go to "Answer Questions"
├─ Provide detailed answers from stakeholders
├─ Generate Refined Document
├─ Export improved documentation
└─ Download as Markdown for distribution
```

### Key Time Savings

- **Skip re-analysis**: Load session instead of re-uploading documents
- **Preserve work**: All steps saved automatically
- **Version control**: Keep both original (v1) and refined (v2_refined) documents
- **Easy sharing**: Export improved documents for team collaboration

## Example Workflow

### Sample 1: COBOL System Analysis with Refinement
```
1. Upload COBOL files + documentation
2. Run analysis (saved)
3. Generate process document (saved)
4. Review gaps (saved)
5. Ask clarification questions (saved)
6. Load session tomorrow
7. Answer questions with SME input
8. Generate refined document
9. Export refined version for architecture team
```

### Sample 2: Process Documentation with Feedback
```
1. Upload Word + Excel files describing processes
2. Run analysis and generate process document
3. Share clarification questions with business owners
4. Load session next week with stakeholder answers
5. Generate refined document incorporating feedback
6. Export for process improvement initiative
```

## Session Management Tips

### Organizing Sessions
- Use descriptive session names: `payroll_system_2024`, `invoice_process_v2`
- Create new sessions for different projects
- One session per document collection

### Loading Sessions
1. Sidebar → "Previous Sessions" → Select session → "Load Session"
2. All analysis, documents, and answers are restored
3. Continue where you left off or generate refined documents

## Next Steps

- Review [README.md](README.md) for full documentation
- Check [example_usage.py](example_usage.py) for programmatic usage
- Explore `src/` directory structure for advanced customization

## Tips & Best Practices

### Document Preparation
- **Clear naming**: Use descriptive file names (e.g., `COBOL_Main_Process.cob`)
- **Related docs**: Upload documents that describe the same system together
- **Format variety**: Include multiple formats (code, docs, data) for better analysis

### Analysis Quality
- **Start small**: Analyze 3-5 documents first to understand the workflow
- **Review summaries**: Check individual document summaries before generating process document
- **Use exports**: Download Markdown files for further editing and sharing

### Troubleshooting
- **Check `.env`**: Credentials are the most common issue
- **Internet**: Ensure stable connection to Azure
- **File size**: Keep individual files under 50MB
- **Format**: Ensure files are valid (not corrupted)

## Need Help?

1. Check error messages in Streamlit interface
2. Look at terminal/console output for details
3. Review [README.md](README.md) for common issues
4. Verify `.env` configuration

## Advanced Usage

For programmatic usage (Python scripts, automation):

```python
from src.document_loaders import get_loader
from src.analyzers import DocumentAnalyzer

# Load and analyze
loader = get_loader("path/to/document.docx")
doc = loader.load()

analyzer = DocumentAnalyzer()
analysis = analyzer.analyze_document(doc)

print(analysis.summary)
```

See [example_usage.py](example_usage.py) for complete examples.

---

**Happy analyzing!** 📚
