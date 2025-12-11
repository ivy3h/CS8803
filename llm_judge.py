"""LLM Judge for evaluating model outputs using OpenAI-compatible APIs."""

import os
import json
import time
from typing import List, Dict, Any, Optional
import openai


class LLMJudge:
    """Evaluates model outputs by comparing them to expected outputs via LLM."""
    
    def __init__(self, api_key, base_url="https://api.openai.com/v1", 
                 model_name="gpt-4o", max_retries=3, retry_delay=1.0):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    def _create_judge_prompt(self, question, model_output, expected_output, options=None):
        prompt = f"""You are an expert medical AI evaluator. Your task is to compare a model's answer to a medical question against the expected correct answer and rate the quality.

Question: {question}

{f"Options:\n{options}\n" if options else ""}

Expected Answer: {expected_output}

Model's Answer: {model_output}

Please evaluate the model's answer based on:
1. Factual accuracy compared to the expected answer
2. Clinical reasoning quality
3. Completeness of the explanation
4. Whether the correct option was selected (if applicable)

Provide a score from 0-100 where:
- 0-20: Completely incorrect, wrong answer selected
- 21-40: Partially correct but major errors or wrong answer
- 41-60: Mostly correct reasoning but minor errors
- 61-80: Correct answer with good reasoning
- 81-100: Excellent answer with thorough, accurate reasoning

Respond ONLY with a JSON object in this exact format:
{{"score": <number 0-100>, "reasoning": "<brief explanation of the score>"}}"""
        
        return prompt
    
    def judge_output(self, question, model_output, expected_output, options=None):
        """Judge a single model output. Returns dict with score (0-100) and reasoning."""
        prompt = self._create_judge_prompt(question, model_output, expected_output, options)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert medical AI evaluator. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                if "score" not in result:
                    raise ValueError("Response missing 'score' field")
                
                score = float(result["score"])
                if not (0 <= score <= 100):
                    raise ValueError(f"Score {score} out of valid range [0, 100]")
                
                return {
                    "score": score,
                    "reasoning": result.get("reasoning", "No reasoning provided")
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error parsing judge response (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to get valid judge response after {self.max_retries} attempts")
                    return {"score": 50.0, "reasoning": f"Error: {str(e)}"}
                    
            except Exception as e:
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"Failed to get judge response after {self.max_retries} attempts")
                    return {"score": 50.0, "reasoning": f"API Error: {str(e)}"}
    
    def judge_batch(self, examples, verbose=True):
        """Judge a batch of examples. Each example should have question, model_output, expected_output."""
        results = []
        
        for i, example in enumerate(examples):
            if verbose and i % 10 == 0:
                print(f"Judging example {i+1}/{len(examples)}...")
            
            result = self.judge_output(
                question=example.get("question", ""),
                model_output=example.get("model_output", ""),
                expected_output=example.get("expected_output", ""),
                options=example.get("options", None)
            )
            
            results.append(result)
            
            if i < len(examples) - 1:
                time.sleep(0.1)
        
        if verbose:
            avg_score = sum(r["score"] for r in results) / len(results) if results else 0
            print(f"Batch judging complete. Average score: {avg_score:.2f}")
        
        return results


def test_judge():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping test")
        return
    
    judge = LLMJudge(api_key=api_key)
    
    result = judge.judge_output(
        question="What is the most likely diagnosis for a patient with chest pain radiating to the left arm?",
        model_output="The most likely diagnosis is myocardial infarction (heart attack). The chest pain radiating to the left arm is a classic symptom.",
        expected_output="Myocardial infarction. The patient presents with classic symptoms of acute coronary syndrome.",
        options="A. Myocardial infarction\nB. Pneumonia\nC. Anxiety\nD. GERD"
    )
    
    print(f"Test result: Score={result['score']}, Reasoning={result['reasoning']}")


if __name__ == "__main__":
    test_judge()
