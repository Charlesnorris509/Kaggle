#!/usr/bin/env python
# coding: utf-8

# A minimal solution to the problem by creating solver agents

# In[ ]:
import os
import io
import shutil
import torch
import pandas as pd
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np
from git import Repo
import subprocess
import gc
from pathlib import Path
import fnmatch
from tqdm import tqdm
import kaggle_evaluation.konwinski_prize_inference_server

# Ensure open-source compliance
model_path = "/kaggle/input/codebert-base/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    vocab_file=os.path.join(model_path, "vocab.json"),
    merges_file=os.path.join(model_path, "merges.txt")
)

model = RobertaModel.from_pretrained(
    model_path,
    local_files_only=True,
    config=os.path.join(model_path, "config.json"),
    state_dict=torch.load(os.path.join(model_path, "pytorch_model.bin"))
)

def calculate_semantic_similarity(pred: str, truth: str, model, tokenizer) -> float:
    inputs = tokenizer([pred, truth], 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention_mask = inputs.attention_mask.unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
    
    return cosine_similarity(embeddings[0].cpu().numpy().reshape(1, -1), 
                            embeddings[1].cpu().numpy().reshape(1, -1))[0][0]

def find_most_related_file(text_input, repo_dir, top_n=1, max_file_size=4000):
    code_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(('.py', '.java', '.cpp', '.js', '.ts', '.c', '.h')):
                filepath = os.path.join(root, file)
                try:
                    if os.path.getsize(filepath) > max_file_size * 1024:
                        continue
                    
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    code_files.append((filepath, content))
                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
    
    if not code_files:
        print("No code files found in the repository directory.")
        return []
    
    code_df = pd.DataFrame(code_files, columns=['file_path', 'file_content'])
    
    file_contents = code_df['file_content'].values.astype(str).tolist()
    all_texts = [str(text_input)] + file_contents
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    text_vector = tfidf_matrix[0]
    file_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(text_vector, file_vectors).flatten()
    
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    top_files = [code_df.iloc[i]['file_path'] for i in top_indices]
    return top_files

def analyze_repo_structure(repo_path: str) -> str:
    parts = []
    
    parts.append("## Directory Structure ##")
    parts.append(get_repo_structure_fallback(repo_path))
    
    parts.append("\n## Key Files ##")
    parts.append(identify_important_files(repo_path))
    
    try:
        parts.append("\n## Development Insights ##")
        parts.append(get_git_aware_structure(repo_path))
    except:
        pass
    
    return "\n".join(parts)

def get_repo_structure_fallback(repo_path: str, max_depth: int = 3) -> str:
    structure = []
    
    def _walk(path: Path, current_depth: int):
        if current_depth > max_depth:
            return
            
        dir_entry = f"{'  ' * (current_depth-1)}📁 {path.name}/"
        structure.append(dir_entry)
        
        files = sorted([f for f in path.iterdir() if f.is_file()])
        for f in files[:5]:
            structure.append(f"{'  ' * current_depth}📄 {f.name}")
            
        dirs = sorted([d for d in path.iterdir() if d.is_dir() 
                      and d.name not in ['.git', '__pycache__', 'node_modules']])
        for d in dirs[:5]:
            _walk(d, current_depth + 1)
    
    try:
        _walk(Path(repo_path), 1)
        return "\n".join(structure)[:2000]
    except Exception as e:
        return f"Error generating structure: {str(e)}"

def get_git_aware_structure(repo_path: str) -> str:
    structure = []
    repo = Repo(repo_path)
    
    contributors = set()
    for commit in repo.iter_commits('HEAD', max_count=10):
        contributors.add(commit.author.name)
    
    modified = [item.a_path for item in repo.index.diff(None)]
    
    structure.append(f"Recent contributors: {', '.join(contributors)[:100]}")
    structure.append("Recent modified files:")
    structure.extend(modified[:10])
    
    return "\n".join(structure)

def identify_important_files(repo_path: str) -> str:
    priority_files = []
    
    important_patterns = [
        'requirements.txt', 'setup.py', 'package.json',
        'Dockerfile', 'Makefile', '*.md',
        'src/', 'lib/', 'main.py', 'app.py'
    ]
    
    for root, _, files in os.walk(repo_path):
        for f in files:
            path = Path(root) / f
            rel_path = path.relative_to(repo_path)
            
            if any(fnmatch.fnmatch(str(rel_path), pat) for pat in important_patterns):
                priority_files.append(f"* {rel_path}")
                
            if path.stat().st_size > 100000:
                priority_files.append(rel_path)
    
    return priority_files[:20]

class GithubIssueSolver:
    def __init__(self):
        model_name = "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_context_length = 16384
        self.max_file_size = 2000

    def _create_prompt(self, problem_statement: str, context: str, repo_structure: str) -> str:
        return f"""**Task**: Fix the GitHub issue by generating a correct git diff patch. Use strict step-by-step reasoning.
    
                **Format Requirements**:
                1. Output MUST start with 'diff --git'
                2. Understand where to modify from relevant files (max 3 files)
                3. Include precise line numbers
                4. Never write code comments unless present in original
                5. Include only necessary changes
                
                **Problem Analysis Framework**:
                1. Root Cause Identification:
                   - Identify specific components causing the issue
                   - Analyze error patterns from problem description
                
                2. Code Context Mapping:
                   - Match issue components to relevant code sections
                   - File structure: {repo_structure}
                   - Relevant code snippets: {context}
                
                3. Change Validation:
                   - Cross-verify each change against problem statement
                   - Ensure no unrelated code modifications
                
                **Example of Good Patch**:
                diff --git a/file.py b/file.py
                --- a/file.py
                +++ b/file.py
                @@ -12,7 +12,7 @@
                     try:
                -        result = process(data)
                +        result = process(data, timeout=30)
                     except TimeoutError:
                -        logger.warning("Timeout occurred")
                +        logger.error("Timeout (30s) exceeded", exc_info=True)
                
                **Current Issue**:
                {problem_statement}
                
                **Step-by-Step Process**:
                1. Identify key components needing modification
                2. Locate exact lines in relevant files
                3. Make minimal changes to fix issue
                4. Verify against all mentioned edge cases
                
                **Output Instructions**:
                - Start immediately with diff patch
                - Use exact file paths from repository
                - Include confidence score (0-100) as last line
                - If uncertain, output "SKIP" with reason
                
                **Begin Fix**:
                """

    def analyze_issue(self, problem_statement: str, repo_path: str) -> str:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            relevant_files = self._find_relevant_files(repo_path, problem_statement)
            if not relevant_files:
                print("No relevant files found")
                return None
            
            context = self._read_file_contents(repo_path, relevant_files)
            repo_structure = analyze_repo_structure(repo_path)
            prompt = self._create_prompt(problem_statement, context, repo_structure)
            
            messages = [
                {"role": "system", "content": """
                                    You are an expert software engineer and code reviewer specializing in resolving real-world GitHub issues by creating precise diff patches. Your role is to analyze the issue description and the most relevant code segments from the repository (pre-identified using cosine similarity) to propose effective and accurate code modifications.
                                    
                                    In this task, you will:
                                    
                                    1. **Understand the Issue:** Carefully analyze the provided GitHub issue description to identify the root cause and the functional or structural problem in the code.
                                    2. **Analyze Relevant Code:** Examine the provided code snippets or files identified as most relevant to the issue. Use your expertise to determine the exact areas requiring modification.
                                    3. **Generate a Diff Patch:** Create a detailed and well-structured diff patch that resolves the issue while maintaining the integrity and functionality of the codebase.
                                    4. **Provide Reasoning:** Accompany your patch with a clear, step-by-step explanation of your reasoning process, detailing how the proposed changes address the issue effectively.
                                    
                                    ### Guidelines:
                                    - **Independent Reasoning:** Your analysis and diff patch should be based solely on the issue description and the provided code snippets. Avoid referencing external solutions or implying prior knowledge of oracle modifications.
                                    - **Clarity and Precision:** Ensure that your diff patch is syntactically correct, adheres to best coding practices, and is easy to apply.
                                    - **Evidence-Based Reasoning:** Clearly justify your changes, linking them to specific parts of the issue description and code. Highlight how the modifications resolve the issue and improve the codebase.
                                    
                                    This task focuses on accurately resolving GitHub issues through diff patches while maintaining high standards of clarity, precision, and logical consistency.
                                                """},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.max_context_length
            ).to(self.model.device)
            
            with torch.inference_mode():
                try:
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        max_new_tokens=16384,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                    
                    response = self.tokenizer.decode(
                        generated_ids[0, inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                finally:
                    del inputs
                    torch.cuda.empty_cache()
                    gc.collect()
            
            if "diff --git" in response:
                diff_start = response.find("diff --git")
                return response[diff_start:]
            return None
            
        except Exception as e:
            print(f"Error generating solution: {str(e)}")
            return None
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _find_relevant_files(self, repo_path: str, problem_statement: str) -> list:
        relevant_files = []
        keywords = set(problem_statement.lower().split())
        
        for root, _, files in os.walk(repo_path):
            if len(relevant_files) >= 2:
                break
            
            for file in files:
                if file.endswith(('.py', '.java', '.cpp', '.h', '.c', '.js', '.ts')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(self.max_file_size * 50)
                            
                        if any(word in content.lower() for word in keywords):
                            relevant_files.append(os.path.relpath(file_path, repo_path))
                            if len(relevant_files) >= 2:
                                break
                    except Exception:
                        continue
                                
        print("num of relevant files", len(relevant_files))
        print("relevant files", relevant_files)
        return relevant_files

    def _read_file_contents(self, repo_path: str, files: list) -> str:
        contents = []
        total_lines = 0
        
        for file in files:
            if total_lines >= self.max_file_size:
                break
                
            try:
                with open(os.path.join(repo_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= self.max_file_size // len(files):
                            break
                        lines.append(line)
                    contents.append(f"File: {file}\n{''.join(lines)}")
                    total_lines += len(lines)
            except Exception:
                continue
        print("file content: ", "\n".join(contents)[:200])
        return "\n".join(contents)

# Global solver instance
solver = None

def predict(
    problem_statement: str, 
    repo_archive: io.BytesIO, 
    pip_packages_archive: io.BytesIO, 
    env_setup_cmds_templates: list[str]
) -> str:
    global solver
    
    repo_path = os.path.join(os.getcwd(), 'repo')
    
    try:
        if solver is None:
            solver = GithubIssueSolver()
        
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        archive_path = os.path.join(os.getcwd(), 'repo_archive.tar')
        with open(archive_path, 'wb') as f:
            f.write(repo_archive.read())
            
        os.makedirs(repo_path, exist_ok=True)
        shutil.unpack_archive(archive_path, repo_path)
        os.remove(archive_path)
        
        sol = solver.analyze_issue(problem_statement, repo_path)
        print("sol: ", sol)
        return sol
        
    except Exception as e:
        print(f"Error in predict function: {str(e)}")
        return None
    finally:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        torch.cuda.empty_cache()
        gc.collect()

inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',
            '/kaggle/tmp/konwinski-prize/',
        ),
        use_concurrency=True,
    )
