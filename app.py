# Import libraries
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image
import io
import hashlib
import pickle
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from google.colab import files
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time

# Configuration settings
CONFIG = {
    "llm_model": "gpt-3.5-turbo",
    "llm_temperature": 0.5,
    "max_pdf_size_mb": 50,
    "cache_file": "/content/pdf_cache.pkl",
    "min_content_length": 100,
    "max_content_per_task": 2000  # Limit content sent to agents
}

# Load environment variables
load_dotenv()

# Securely get OpenAI API key
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    return api_key

OPENAI_API_KEY = get_api_key()

# File-based caching for PDF content
def load_cache():
    if os.path.exists(CONFIG["cache_file"]):
        with open(CONFIG["cache_file"], "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CONFIG["cache_file"], "wb") as f:
        pickle.dump(cache, f)

def get_pdf_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Validate PDF
def validate_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        if os.path.getsize(file_path) / 1024 / 1024 > CONFIG["max_pdf_size_mb"]:
            return False, "PDF size exceeds 50 MB limit."
        text = ""
        for page in doc:
            text += page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        if len(text) < CONFIG["min_content_length"]:
            return False, f"PDF has insufficient text content ({len(text)} characters). Try a text-based PDF with tables and charts."
        return True, "PDF is valid."
    except Exception as e:
        return False, f"Invalid PDF: {e}. Ensure the PDF is not encrypted or corrupted."

# Enhanced PDF reading
def read_pdf(file_path):
    pdf_hash = get_pdf_hash(file_path)
    cache = load_cache()

    if pdf_hash in cache:
        print(f"Loading PDF content from cache for {file_path}...")
        return cache[pdf_hash]

    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text += pytesseract.image_to_string(image)

        text = text.encode("utf-8", "replace").decode("utf-8")
        cache[pdf_hash] = text
        save_cache(cache)
        print(f"PDF content cached successfully for {file_path}.")

    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

    return text

# Initialize LLM
llm = ChatOpenAI(
    model=CONFIG["llm_model"],
    temperature=CONFIG["llm_temperature"],
    openai_api_key=OPENAI_API_KEY
)

# Create agents
def create_agent(role, goal, backstory):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        llm=llm,
        max_iter=15
    )

AGENTS = {
    "pdf_analyst": create_agent(
        role="PDF Content Analyst",
        goal="Extract and analyze complex PDF content, focusing on CMG-related technical details",
        backstory="Expert in handling technical documents, including CMG simulation software"
    ),
    "data_extractor": create_agent(
        role="Data Extraction Specialist",
        goal="Extract structured data like tables and numerical information from CMG-related PDFs",
        backstory="Expert in organizing tabular and numerical data"
    ),
    "visualization_analyst": create_agent(
        role="Visualization Analyst",
        goal="Analyze charts, diagrams, and figures in CMG-related PDFs",
        backstory="Specialist in interpreting visual data representations"
    ),
    "summary_specialist": create_agent(
        role="Structured Summary Expert",
        goal="Create detailed technical summaries of CMG-related PDFs",
        backstory="Technical writer with expertise in summarizing simulation documents"
    ),
    "recommendation_agent": create_agent(
        role="Recommendation Specialist",
        goal="Generate actionable recommendations based on CMG-related analysis",
        backstory="Expert in synthesizing insights for simulation optimization"
    ),
    "question_answerer": create_agent(
        role="Question-Answering Specialist",
        goal="Answer user questions about the CMG-related PDF content accurately",
        backstory="Expert in understanding CMG technical documents"
    ),
}

# Create tasks
def create_task(description, expected_output, agent, context=None):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        context=context or []
    )

def get_tasks(pdf_content, agents):
    truncated_content = pdf_content[:CONFIG["max_content_per_task"]]
    analysis_task = create_task(
        description=f"Analyze and structure this CMG-related PDF content:\n{truncated_content}",
        expected_output="Structured document with section headers, key terms, tables, figures, and notations",
        agent=agents["pdf_analyst"]
    )

    data_extraction_task = create_task(
        description=f"Extract structured data from this CMG-related PDF content:\n{truncated_content}",
        expected_output="A JSON string containing extracted data with tables (as lists of dictionaries), statistics, and metrics",
        agent=agents["data_extractor"]
    )

    visualization_task = create_task(
        description=f"Analyze visualizations in this CMG-related PDF content:\n{truncated_content}",
        expected_output="Analysis of charts/diagrams, key trends, and significant visual elements",
        agent=agents["visualization_analyst"]
    )

    summary_task = create_task(
        description="Create detailed technical summary of the CMG-related PDF with proper formatting",
        expected_output="Markdown-formatted summary with key sections, terms, figures, data, and analysis",
        agent=agents["summary_specialist"],
        context=[analysis_task, data_extraction_task, visualization_task]
    )

    recommendation_task = create_task(
        description="Generate actionable recommendations based on the CMG-related analysis and summary",
        expected_output="List of prioritized recommendations with actionable steps and potential impact",
        agent=agents["recommendation_agent"],
        context=[analysis_task, data_extraction_task, visualization_task, summary_task]
    )

    question_answering_task = create_task(
        description="Answer user questions about the CMG-related PDF content based on the provided context",
        expected_output="Clear, concise answers to user questions in the format: **Question**: [User question] **Answer**: [Response]",
        agent=agents["question_answerer"],
        context=[analysis_task, data_extraction_task, visualization_task, summary_task]
    )

    return [
        analysis_task,
        data_extraction_task,
        visualization_task,
        summary_task,
        recommendation_task,
        question_answering_task
    ]

# Plot extracted data (line and bar charts)
def plot_data(data, pdf_name):
    if not data:
        print("No data available for plotting.")
        return
    try:
        times = [d.get("time", 0) for d in data]
        rates = [d.get("rate", 0) for d in data]

        # Line plot
        plt.figure(figsize=(8, 4))
        plt.plot(times, rates, label="Production Rate")
        plt.xlabel("Time (days)")
        plt.ylabel("Production Rate (bbl/day)")
        plt.title(f"Production Rate - {pdf_name}")
        plt.legend()
        plt.savefig(f"production_line_plot_{pdf_name}.png")
        plt.show()
        print(f"Line plot saved as production_line_plot_{pdf_name}.png")

        # Bar plot
        plt.figure(figsize=(8, 4))
        plt.bar(times, rates, color="skyblue", label="Production Rate")
        plt.xlabel("Time (days)")
        plt.ylabel("Production Rate (bbl/day)")
        plt.title(f"Production Rate Bar Chart - {pdf_name}")
        plt.legend()
        plt.savefig(f"production_bar_plot_{pdf_name}.png")
        plt.show()
        print(f"Bar plot saved as production_bar_plot_{pdf_name}.png")

    except Exception as e:
        print(f"Error plotting data: {e}")

# Run the crew with progress bar
def run_crew(pdf_content, agents, tasks, pdf_name):
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks[:-1],  # Exclude question-answering task
        verbose=True
    )

    print(f"\n=== Processing {pdf_name} ===")
    task_count = len(tasks[:-1])
    progress_bar = tqdm(total=task_count, desc="Running Agents", unit="task")

    try:
        results = crew.kickoff()
    except Exception as e:
        print(f"Error during crew execution: {e}. Try a different PDF or check your API key.")
        progress_bar.close()
        return None, None

    # Display and save each task's output
    for task in tasks[:-1]:
        agent_name = task.agent.role
        output = task.output if task.output else "No output generated. Check if the PDF contains relevant content (e.g., tables, charts)."
        print(f"\n=== {agent_name} Output ===")
        print(output)

        # Save to individual file
        safe_pdf_name = pdf_name.replace(".", "_")
        with open(f"{agent_name.replace(' ', '_').lower()}_output_{safe_pdf_name}.txt", "w") as f:
            f.write(str(output))
        print(f"{agent_name} output saved to {agent_name.replace(' ', '_').lower()}_output_{safe_pdf_name}.txt")
        progress_bar.update(1)

    progress_bar.close()

    # Handle data extraction for plotting
    data_task_result = tasks[1].output
    extracted_data = {}
    if data_task_result and isinstance(data_task_result, str):
        try:
            extracted_data = json.loads(data_task_result)
        except json.JSONDecodeError:
            print("Data extraction output is not valid JSON. No plot generated.")
    else:
        print("Data extraction output is not a string or is empty. No plot generated.")

    plot_data(extracted_data.get("tables", []), safe_pdf_name)

    # Save combined results
    with open(f"analysis_summary_{safe_pdf_name}.md", "w") as f:
        f.write(str(results))
    print(f"\nCombined results saved to analysis_summary_{safe_pdf_name}.md")

    return results, tasks[-1]

# Interactive question-answering
def run_question_answering(question_task, agents, pdf_name):
    question_history = []
    safe_pdf_name = pdf_name.replace(".", "_")
    print(f"\n📢 Enter your questions about {pdf_name} (type 'exit' to stop):")
    while True:
        user_question = input("Your question: ")
        if user_question.lower() == "exit":
            break
        if not user_question.strip():
            print("Please enter a valid question")
            continue

        question_history.append(user_question)
        question_task.description = f"Answer this user question about the CMG-related PDF content:\n{user_question}"

        print("\n=== Question-Answering Agent Started ===")
        qa_crew = Crew(
            agents=[agents["question_answerer"]],
            tasks=[question_task],
            verbose=True
        )
        try:
            qa_result = qa_crew.kickoff()
            print(f"\n**Question**: {user_question}\n**Answer**: {qa_result}")
        except Exception as e:
            print(f"Error answering question: {e}. Try a different question or check your API key.")
        print("=== Question-Answering Agent Finished ===")

    with open(f"question_history_{safe_pdf_name}.txt", "w") as f:
        f.write("\n".join(question_history))
    print(f"Question history saved to question_history_{safe_pdf_name}.txt")

# Main execution
def main():
    print("Upload one or more PDF files (each under 50 MB):")
    uploaded = files.upload()
    if not uploaded:
        print("No files uploaded. Exiting.")
        return

    pdf_results = {}
    for pdf_name in uploaded.keys():
        print(f"\nValidating {pdf_name}...")
        is_valid, validation_message = validate_pdf(pdf_name)
        if not is_valid:
            print(f"Validation failed: {validation_message}")
            continue

        pdf_content = read_pdf(pdf_name)
        if not pdf_content or len(pdf_content) < CONFIG["min_content_length"]:
            print(f"PDF content is empty or too short ({len(pdf_content)} characters). Using fallback content.")
            pdf_content = """
            CMG Reservoir Simulation Report
            1. Introduction
            This report details a reservoir simulation using CMG software.

            2. Simulation Methodology
            - Grid Resolution: 100x100
            - Permeability: 50 mD
            - Finite Difference Method used

            3. Data Tables
            Time (days), Production Rate (bbl/day)
            0, 1000
            30, 950
            60, 900

            4. Figures
            - Pressure vs. Time: Declining trend observed
            - Recovery Rate: 90%

            5. Analysis
            Model accuracy is high, but runtime can be optimized.
            """

        tasks = get_tasks(pdf_content, AGENTS)
        results, question_task = run_crew(pdf_content, AGENTS, tasks, pdf_name)
        if results is None:
            print(f"Skipping {pdf_name} due to processing error.")
            continue

        pdf_results[pdf_name] = results
        run_question_answering(question_task, AGENTS, pdf_name)

    # Summarize all PDFs processed
    print("\n=== Processing Summary ===")
    for pdf_name, results in pdf_results.items():
        print(f"{pdf_name}:")
        print(results)
        print()

if __name__ == "__main__":
    main()