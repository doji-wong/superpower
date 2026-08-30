import json
import os
import re
import math
import argparse
from collections import Counter
from pathlib import Path

# Try to import dspy, if not installed, fail gracefully.
try:
    import dspy
except ImportError:
    print("Error: dspy is not installed. Please run: pip install dspy-ai")
    exit(1)

# --- TF-IDF Logic ---
STOP = {
    'a', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'before', 'by', 'for',
    'from', 'in', 'into', 'is', 'it', 'its', 'my', 'need', 'needs', 'of', 'on',
    'or', 'our', 'so', 'that', 'the', 'them', 'this', 'to', 'use', 'want',
    'we', 'when', 'with', 'you', 'your', 'help', 'me', 'i',
}

def stem(t):
    for suf in ['ally', 'ing', 'ed', 'es', 'al']:
        if len(t) > len(suf) + 3 and t.endswith(suf):
            t = t[:-len(suf)]
            break
    if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
        t = t[:-1]
    if len(t) > 4 and t.endswith('e'):
        t = t[:-1]
    if len(t) > 4 and t[-1] == t[-2] and t[-1] not in 'aeiou':
        t = t[:-1]
    if len(t) > 3 and t.endswith('y'):
        t = t[:-1] + 'i'
    return t

def tokenize(text):
    text = re.sub(r'[^a-z0-9\s-]', ' ', text.lower())
    tokens = [stem(t) for t in re.split(r'[\s-]+', text) if len(t) > 2 and t not in STOP]
    return tokens

class TFIDFCatalog:
    def __init__(self):
        self.docs = {}
        self.idf = {}
        self.n = 0
    
    def add_skill(self, name, description):
        name_tokens = tokenize(name.replace('-', ' '))
        tokens = name_tokens + name_tokens + tokenize(description)
        self.docs[name] = Counter(tokens)
        
    def build_idf(self):
        df = Counter()
        self.n = len(self.docs)
        for tf in self.docs.values():
            for term in tf.keys():
                df[term] += 1
        self.idf = {term: math.log(1 + self.n / (1 + count)) for term, count in df.items()}
        
    def get_vec(self, tf):
        return {term: count * self.idf.get(term, 0) for term, count in tf.items()}
        
    def cosine(self, a, b):
        dot = sum(a[t] * b.get(t, 0) for t in a)
        na = sum(w * w for w in a.values())
        nb = sum(w * w for w in b.values())
        if not na or not nb: return 0
        return dot / (math.sqrt(na) * math.sqrt(nb))
        
    def rank(self, prompt):
        pv = self.get_vec(Counter(tokenize(prompt)))
        scores = []
        for name, tf in self.docs.items():
            scores.append((name, self.cosine(pv, self.get_vec(tf))))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# --- Data Loading ---
ROOT = Path(__file__).parent.parent
SKILLS_DIR = ROOT / 'skills'
CASES_DIR = ROOT / 'evals' / 'cases'

def load_skills():
    skills = {}
    for entry in os.scandir(SKILLS_DIR):
        if entry.is_dir():
            skill_md = Path(entry.path) / 'SKILL.md'
            if skill_md.exists():
                content = skill_md.read_text('utf-8')
                m = re.search(r'^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)', content)
                if m:
                    name_m = re.search(r'^name:\s*(.+)$', m.group(1), re.M)
                    desc_m = re.search(r'^description:\s*(.+)$', m.group(1), re.M)
                    if name_m and desc_m:
                        skills[name_m.group(1).strip()] = desc_m.group(1).strip()
    return skills

def load_cases():
    cases = {}
    if CASES_DIR.exists():
        for file in CASES_DIR.glob('*.json'):
            try:
                cases[file.stem] = json.loads(file.read_text('utf-8'))
            except Exception as e:
                print(f"Failed to parse {file}: {e}")
    return cases

# --- DSPy Setup ---
class GenerateDescription(dspy.Signature):
    """Generate a single-sentence description for a coding assistant skill. The description must be highly distinct from other skills, matching the vocabulary of positive prompts, but avoiding the vocabulary of negative prompts."""
    skill_name = dspy.InputField(desc="The name of the skill")
    current_description = dspy.InputField(desc="The current description")
    positive_prompts = dspy.InputField(desc="Prompts that SHOULD route to this skill. Incorporate these keywords!")
    negative_prompts = dspy.InputField(desc="Prompts that should NOT route to this skill. Avoid these concepts.")
    optimized_description = dspy.OutputField(desc="A single concise sentence describing the skill, loaded with the vocabulary from the positive prompts.")

class DescriptionOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateDescription)
    
    def forward(self, skill_name, current_description, positive_prompts, negative_prompts):
        return self.generate(
            skill_name=skill_name,
            current_description=current_description,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts
        )

# --- Metric ---
def evaluate_description(skill_name, new_description, positive_prompts, negative_prompts, catalog_skills):
    # Create a temporary catalog
    temp_catalog = TFIDFCatalog()
    for name, desc in catalog_skills.items():
        if name == skill_name:
            temp_catalog.add_skill(name, new_description)
        else:
            temp_catalog.add_skill(name, desc)
    temp_catalog.build_idf()
    
    score = 0
    # Evaluate positive prompts
    for p in positive_prompts:
        prompt_text = p.get('prompt', '')
        top_k = p.get('top_k', 3)
        ranking = temp_catalog.rank(prompt_text)
        idx = next((i for i, r in enumerate(ranking) if r[0] == skill_name), -1)
        if idx == 0 and ranking[idx][1] > 0:
            score += 1.0 # rank 1
        elif 0 <= idx < top_k and ranking[idx][1] > 0:
            score += 0.5 # top k
            
    # Evaluate negative prompts
    for p in negative_prompts:
        prompt_text = p.get('prompt', '')
        ranking = temp_catalog.rank(prompt_text)
        if ranking[0][0] == skill_name and ranking[0][1] > 0:
            score -= 1.0 # Failed, ranked #1 for a negative prompt
            
    # Normalize score
    total_possible = len(positive_prompts)
    return max(0.0, score / total_possible) if total_possible > 0 else 0.0

def dspy_metric(example, pred, trace=None):
    # Needs access to global catalog_skills, we can capture it in a closure when running
    return evaluate_description(
        example.skill_name,
        pred.optimized_description,
        example.positive_prompts,
        example.negative_prompts,
        global_catalog_skills
    )

global_catalog_skills = {}

def update_skill_md(skill_name, new_description):
    skill_md_path = SKILLS_DIR / skill_name / 'SKILL.md'
    if not skill_md_path.exists():
        return
    content = skill_md_path.read_text('utf-8')
    new_content = re.sub(
        r'^description:\s*.+$',
        f'description: {new_description}',
        content,
        flags=re.M
    )
    skill_md_path.write_text(new_content, 'utf-8')
    print(f"Updated {skill_name}/SKILL.md")

class AntigravityLM(dspy.LM):
    def __init__(self, model="agy", **kwargs):
        super().__init__(model=model, **kwargs)
        
    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            prompt = "\n".join(m.get("content", "") for m in messages if "content" in m)
        if not prompt:
            return [""]
        import subprocess
        try:
            result = subprocess.run(
                ['agy', '-p', prompt],
                text=True,
                capture_output=True,
                check=True
            )
            return [result.stdout.strip()]
        except subprocess.CalledProcessError as e:
            print("agy failed:", e.stderr)
            return [""]

def main():
    parser = argparse.ArgumentParser(description="Optimize skill descriptions using DSPy.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use with DSPy")
    parser.add_argument("--ollama", type=str, help="Use Ollama with the specified local model (e.g., 'llama3.2', 'mistral') instead of an API")
    parser.add_argument("--agy", action="store_true", help="Use the current Antigravity (agy) model via the CLI.")
    parser.add_argument("--skill", type=str, help="Specific skill to optimize. If omitted, optimizes all skills.")
    parser.add_argument("--dry-run", action="store_true", help="Print optimized descriptions without updating SKILL.md")
    parser.add_argument("--tune-prompt", action="store_true", help="Use DSPy COPRO to optimize the GenerateDescription prompt first.")
    args = parser.parse_args()

    # Configure DSPy LM
    if args.agy:
        print("Using the local Antigravity (agy) model via CLI...")
        lm = AntigravityLM()
    elif args.ollama:
        print(f"Using local Ollama model: {args.ollama}")
        lm = dspy.OllamaLocal(model=args.ollama)
    else:
        # Note: Requires OPENAI_API_KEY environment variable. You can change this to Anthropics or others.
        lm = dspy.LM(model=args.model)
    dspy.configure(lm=lm)
    
    skills = load_skills()
    cases = load_cases()
    global global_catalog_skills
    global_catalog_skills = skills.copy()
    
    # Create DSPy Examples
    dataset = []
    for skill_name, current_desc in skills.items():
        if skill_name not in cases:
            continue
        case_data = cases[skill_name]
        pos_prompts = case_data.get('trigger', {}).get('positive', [])
        neg_prompts = case_data.get('trigger', {}).get('negative', [])
        
        example = dspy.Example(
            skill_name=skill_name,
            current_description=current_desc,
            positive_prompts=pos_prompts,
            negative_prompts=neg_prompts
        ).with_inputs('skill_name', 'current_description', 'positive_prompts', 'negative_prompts')
        dataset.append(example)

    optimizer = DescriptionOptimizer()

    if args.tune_prompt:
        print("Tuning prompt with COPRO...")
        from dspy.teleprompt import COPRO
        teleprompter = COPRO(metric=dspy_metric, depth=3)
        optimizer = teleprompter.compile(optimizer, trainset=dataset, eval_kwargs=dict(num_threads=1))
        print("Prompt tuning complete.")

    target_dataset = dataset
    if args.skill:
        target_dataset = [ex for ex in dataset if ex.skill_name == args.skill]
        if not target_dataset:
            print(f"Skill '{args.skill}' not found or has no eval cases.")
            return

    for example in target_dataset:
        print(f"\n--- Optimizing: {example.skill_name} ---")
        current_score = evaluate_description(
            example.skill_name,
            example.current_description,
            example.positive_prompts,
            example.negative_prompts,
            skills
        )
        print(f"Current Description: {example.current_description}")
        print(f"Current Score: {current_score:.2f}")
        
        # Predict new description
        pred = optimizer(
            skill_name=example.skill_name,
            current_description=example.current_description,
            positive_prompts=str(example.positive_prompts),
            negative_prompts=str(example.negative_prompts)
        )
        new_desc = pred.optimized_description
        
        new_score = evaluate_description(
            example.skill_name,
            new_desc,
            example.positive_prompts,
            example.negative_prompts,
            skills
        )
        
        print(f"New Description: {new_desc}")
        print(f"New Score: {new_score:.2f}")
        
        if new_score > current_score or (new_score == current_score and new_desc != example.current_description):
            print("Improvement found!")
            if not args.dry_run:
                update_skill_md(example.skill_name, new_desc)
                # Update our running catalog so future evaluations in the loop use the new IDF!
                skills[example.skill_name] = new_desc
                global_catalog_skills = skills.copy()
        else:
            print("No improvement. Keeping current description.")

if __name__ == '__main__':
    main()
