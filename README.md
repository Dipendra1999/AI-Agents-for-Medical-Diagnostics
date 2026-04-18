# AI Agents for Medical Diagnostics

Multi-agent medical-report analysis project built with LangChain and OpenAI models.  
The system simulates specialist review (cardiology, psychology, pulmonology) and then combines those opinions into a final multidisciplinary differential output.

## Important Disclaimer

This repository is for educational and research purposes only.  
It is **not** a medical device and must not be used for real-world diagnosis, treatment, or clinical decision-making.

## How It Works

- `Main.py` loads a sample medical report from `Medical Reports/`.
- Three specialist agents run in parallel:
  - Cardiologist
  - Psychologist
  - Pulmonologist
- A `MultidisciplinaryTeam` agent reviews all specialist outputs.
- Final results are written to `results/final_diagnosis.txt`.

## Project Structure

```text
AI-Agents-for-Medical-Diagnostics/
├── Main.py
├── Utils/
│   └── Agents.py
├── Medical Reports/
├── results/                      # generated output folder
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

## Requirements

- Python 3.10+ recommended
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/AI-Agents-for-Medical-Diagnostics.git
   cd AI-Agents-for-Medical-Diagnostics
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your API key file at the repository root as `apikey.env`:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Run the Project

```bash
python Main.py
```

If successful, the final output is saved to:

```text
results/final_diagnosis.txt
```

## Customizing Input Reports

- Add or update report files under `Medical Reports/`.
- Update the file path in `Main.py` to select which report is analyzed.

## Contributing

Please read `CONTRIBUTING.md` for contribution workflow, safety expectations, and PR guidelines.

## Notes

- Current default model in `Utils/Agents.py` is `gpt-5`.
- Keep API keys and sensitive data out of version control.