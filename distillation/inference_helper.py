"""
Helper functions for chess move evaluation inference and visualization.

This module provides utilities for:
- Loading and processing chess evaluation data
- Running batch inference with vLLM
- Extracting and normalizing move answers
- Calculating evaluation statistics
- Creating visualizations
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
from vllm import SamplingParams
import torch


def create_conversation(input_text: str) -> List[Dict[str, str]]:
    """
    Create conversation format matching the training script.
    
    Args:
        input_text: The chess position and candidate moves
    
    Returns:
        List of message dictionaries for the chat template
    """
    return [
        {
            "role": "system",
            "content": "Classify the better move. Output format: MoveA or MoveB"
        },
        {
            "role": "user",
            "content": input_text
        },
    ]


def extract_move_answer(text: str) -> str:
    """
    Extract MoveA or MoveB from the generated text - looking at the END of the text.
    
    Args:
        text: Generated text from the model
    
    Returns:
        Normalized move answer (MOVEA or MOVEB) or None if not found
    """
    # Look at the last 200 characters where the answer should be
    text_end = text[-200:] if len(text) > 200 else text
    
    # Look for MoveA or MoveB (case insensitive, with optional colon and move notation)
    match = re.search(r'Move([AB])(?::[a-z0-9]+)?', text_end, re.IGNORECASE)
    if match:
        return f"MOVE{match.group(1).upper()}"
    
    # If not found at the end, search the whole text (take the LAST occurrence)
    matches = list(re.finditer(r'Move([AB])(?::[a-z0-9]+)?', text, re.IGNORECASE))
    if matches:
        return f"MOVE{matches[-1].group(1).upper()}"
    
    return None


def normalize_expected_answer(expected_output: str) -> str:
    """
    Normalize the expected answer to just MOVEA or MOVEB.
    
    Args:
        expected_output: Raw expected output from dataset
    
    Returns:
        Normalized answer (MOVEA or MOVEB)
    """
    match = re.search(r'Move([AB])', expected_output, re.IGNORECASE)
    if match:
        return f"MOVE{match.group(1).upper()}"
    return expected_output.upper()


def load_chess_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and filter chess evaluation dataset.
    
    Args:
        data_path: Path to the chess output JSON file
    
    Returns:
        List of valid chess samples (without errors)
    """
    print("Loading chess dataset...")
    with open(data_path, 'r') as f:
        chess_data = json.load(f)
    
    # Filter out samples with errors
    chess_data = [item for item in chess_data if 'error' not in item]
    print(f"Loaded {len(chess_data)} valid chess samples")
    
    return chess_data


def create_prompts(chess_data: List[Dict], tokenizer) -> List[str]:
    """
    Create formatted prompts from chess data using tokenizer's chat template.
    
    Args:
        chess_data: List of chess evaluation samples
        tokenizer: HuggingFace tokenizer with chat template
    
    Returns:
        List of formatted prompts ready for inference
    """
    prompts = []
    for item in chess_data:
        conversation = create_conversation(item['input'])
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)
    
    print(f"Created {len(prompts)} prompts for inference")
    return prompts


def run_inference(llm, prompts: List[str], max_tokens: int = 2048, temperature: float = 0.0):
    """
    Run batch inference on prompts using vLLM.
    
    Args:
        llm: vLLM LLM instance
        prompts: List of formatted prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        vLLM generation outputs
    """
    print("\nRunning inference on all samples...")
    print(f"Using max_tokens={max_tokens} to ensure complete responses")
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def process_inference_results(
    outputs,
    chess_data: List[Dict],
    progress_interval: int = 100
) -> Tuple[List[Dict], int, int]:
    """
    Process inference outputs and calculate accuracy statistics.
    
    Args:
        outputs: vLLM generation outputs
        chess_data: Original chess dataset
        progress_interval: Print progress every N samples
    
    Returns:
        Tuple of (results list, teacher_correct count, student_correct count)
    """
    results = []
    teacher_correct = 0
    student_correct = 0
    
    for idx, output in enumerate(outputs):
        item = chess_data[idx]
        generated_text = output.outputs[0].text.strip()
        
        # Extract the teacher's answer
        teacher_full_response = item['response']['generated_text']
        if 'assistant' in teacher_full_response:
            teacher_response_part = teacher_full_response.split('assistant')[-1].strip()
        else:
            teacher_response_part = teacher_full_response
        
        teacher_answer = extract_move_answer(teacher_response_part)
        
        # Extract the student's answer
        student_answer = extract_move_answer(generated_text)
        
        # Get the expected (correct) answer and normalize it
        expected_answer = normalize_expected_answer(item['expected_output'])
        
        # Check correctness
        teacher_is_correct = (teacher_answer == expected_answer) if teacher_answer else False
        student_is_correct = (student_answer == expected_answer) if student_answer else False
        
        if teacher_is_correct:
            teacher_correct += 1
        if student_is_correct:
            student_correct += 1
        
        results.append({
            'question_id': idx,
            'input': item['input'],
            'expected_answer': expected_answer,
            'teacher_answer': teacher_answer,
            'teacher_correct': teacher_is_correct,
            'student_answer': student_answer,
            'student_correct': student_is_correct,
            'student_raw_output': generated_text
        })
        
        # Print progress
        if (idx + 1) % progress_interval == 0:
            print(f"Processed {idx + 1}/{len(chess_data)} samples...")
            print(f"  Current Teacher Accuracy: {teacher_correct}/{idx+1} ({teacher_correct/(idx+1)*100:.1f}%)")
            print(f"  Current Student Accuracy: {student_correct}/{idx+1} ({student_correct/(idx+1)*100:.1f}%)")
    
    return results, teacher_correct, student_correct


def save_evaluation_results(
    results: List[Dict],
    teacher_correct: int,
    student_correct: int,
    output_file: str = 'static/chess_evaluation_results.json'
) -> Dict[str, Any]:
    """
    Save evaluation results to JSON file with summary statistics.
    
    Args:
        results: List of processed results
        teacher_correct: Number of correct teacher predictions
        student_correct: Number of correct student predictions
        output_file: Path to output JSON file
    
    Returns:
        Summary dictionary with all statistics
    """
    import os
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    total_samples = len(results)
    teacher_accuracy = (teacher_correct / total_samples) * 100
    student_accuracy = (student_correct / total_samples) * 100
    
    summary = {
        'total_samples': total_samples,
        'teacher_correct': teacher_correct,
        'teacher_accuracy': teacher_accuracy,
        'student_correct': student_correct,
        'student_accuracy': student_accuracy,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Total Samples: {total_samples}")
    print(f"Teacher Accuracy: {teacher_correct}/{total_samples} ({teacher_accuracy:.2f}%)")
    print(f"Student Accuracy: {student_correct}/{total_samples} ({student_accuracy:.2f}%)")
    print(f"\nResults saved to: {output_file}")
    
    return summary


def print_sample_results(results: List[Dict], num_samples: int = 5):
    """
    Print sample results for verification.
    
    Args:
        results: List of processed results
        num_samples: Number of samples to display
    """
    print(f"\n{'='*80}")
    print("Sample Results (first 5):")
    print(f"{'='*80}")
    for i in range(min(num_samples, len(results))):
        r = results[i]
        print(f"\nQuestion {i+1}:")
        print(f"  Expected: {r['expected_answer']}")
        print(f"  Teacher:  {r['teacher_answer']} {'✓' if r['teacher_correct'] else '✗'}")
        print(f"  Student:  {r['student_answer']} {'✓' if r['student_correct'] else '✗'}")
        print(f"  Student output length: {len(r['student_raw_output'])} chars")
        if r['student_answer']:
            print(f"  Student raw (last 100 chars): ...{r['student_raw_output'][-100:]}")


def print_detailed_statistics(data: Dict[str, Any]):
    """
    Print detailed evaluation statistics.
    
    Args:
        data: Summary dictionary from evaluation results
    """
    total_samples = data['total_samples']
    teacher_correct = data['teacher_correct']
    teacher_accuracy = data['teacher_accuracy']
    student_correct = data['student_correct']
    student_accuracy = data['student_accuracy']
    
    print("="*70)
    print("CHESS MOVE EVALUATION RESULTS")
    print("="*70)
    print(f"\nDataset Size: {total_samples} chess positions")
    print(f"\n{'Model':<20} {'Correct':<15} {'Incorrect':<15} {'Accuracy':<15}")
    print("-"*70)
    print(f"{'Teacher (Qwen3-30B)':<20} {teacher_correct:<15} {total_samples - teacher_correct:<15} {teacher_accuracy:.2f}%")
    print(f"{'Student (Qwen3-0.6B)':<20} {student_correct:<15} {total_samples - student_correct:<15} {student_accuracy:.2f}%")
    print("-"*70)
    
    # Calculate agreement statistics
    both_correct = sum(1 for r in data['results'] if r['teacher_correct'] and r['student_correct'])
    both_wrong = sum(1 for r in data['results'] if not r['teacher_correct'] and not r['student_correct'])
    teacher_right_student_wrong = sum(1 for r in data['results'] if r['teacher_correct'] and not r['student_correct'])
    student_right_teacher_wrong = sum(1 for r in data['results'] if not r['teacher_correct'] and r['student_correct'])
    
    print(f"\nAgreement Analysis:")
    print(f"  Both Correct:                    {both_correct} ({both_correct/total_samples*100:.1f}%)")
    print(f"  Both Wrong:                      {both_wrong} ({both_wrong/total_samples*100:.1f}%)")
    print(f"  Teacher Right, Student Wrong:    {teacher_right_student_wrong} ({teacher_right_student_wrong/total_samples*100:.1f}%)")
    print(f"  Student Right, Teacher Wrong:    {student_right_teacher_wrong} ({student_right_teacher_wrong/total_samples*100:.1f}%)")
    
    # Knowledge retention
    if teacher_correct > 0:
        retention_rate = (both_correct / teacher_correct) * 100
        print(f"\nKnowledge Retention: {retention_rate:.2f}%")
        print(f"  (Student correctly answered {retention_rate:.1f}% of questions the teacher got right)")
    
    print("="*70)


def create_visualization(data: Dict[str, Any], output_file: str = 'static/chess_evaluation_results.png'):
    """
    Create comprehensive visualization of evaluation results.
    
    Args:
        data: Summary dictionary from evaluation results
        output_file: Path to save the visualization
    """
    import os
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    total_samples = data['total_samples']
    teacher_correct = data['teacher_correct']
    teacher_accuracy = data['teacher_accuracy']
    student_correct = data['student_correct']
    student_accuracy = data['student_accuracy']
    
    # Calculate agreement statistics
    both_correct = sum(1 for r in data['results'] if r['teacher_correct'] and r['student_correct'])
    both_wrong = sum(1 for r in data['results'] if not r['teacher_correct'] and not r['student_correct'])
    teacher_right_student_wrong = sum(1 for r in data['results'] if r['teacher_correct'] and not r['student_correct'])
    student_right_teacher_wrong = sum(1 for r in data['results'] if not r['teacher_correct'] and r['student_correct'])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Chess Move Evaluation: Teacher vs Student Model Performance', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    models = ['Teacher\n(Qwen3-30B)', 'Student\n(Qwen3-0.6B)']
    accuracies = [teacher_accuracy, student_accuracy]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Correct vs Incorrect Counts
    ax2 = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    correct_counts = [teacher_correct, student_correct]
    incorrect_counts = [total_samples - teacher_correct, total_samples - student_correct]
    bars1 = ax2.bar(x - width/2, correct_counts, width, label='Correct', color='#06A77D', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='#D62246', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax2.set_title('Correct vs Incorrect Answers', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Teacher', 'Student'])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Agreement Analysis Pie Chart
    ax3 = axes[1, 0]
    agreement_data = [both_correct, both_wrong, teacher_right_student_wrong, student_right_teacher_wrong]
    labels = [f'Both Correct\n({both_correct})', 
              f'Both Wrong\n({both_wrong})', 
              f'Teacher Right\nStudent Wrong\n({teacher_right_student_wrong})',
              f'Student Right\nTeacher Wrong\n({student_right_teacher_wrong})']
    colors_pie = ['#06A77D', '#D62246', '#F77F00', '#9B59B6']
    explode = (0.05, 0.05, 0.05, 0.05)
    ax3.pie(agreement_data, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=explode, textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax3.set_title('Teacher-Student Agreement Analysis', fontsize=13, fontweight='bold')
    
    # 4. Performance Metrics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    table_data = [
        ['Metric', 'Teacher', 'Student'],
        ['Total Questions', str(total_samples), str(total_samples)],
        ['Correct Answers', str(teacher_correct), str(student_correct)],
        ['Incorrect Answers', str(total_samples - teacher_correct), str(total_samples - student_correct)],
        ['Accuracy', f'{teacher_accuracy:.2f}%', f'{student_accuracy:.2f}%'],
        ['Model Size', '30B params', '0.6B params'],
        ['Size Ratio', '1x', '50x smaller']
    ]
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
            table[(i, j)].set_text_props(weight='bold' if j == 0 else 'normal')
    
    ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()


def print_sample_predictions(data: Dict[str, Any], num_samples: int = 5):
    """
    Print sample predictions for verification.
    
    Args:
        data: Summary dictionary from evaluation results
        num_samples: Number of samples to display
    """
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 5 questions)")
    print("="*70)
    for i, result in enumerate(data['results'][:num_samples]):
        print(f"\nQuestion {i+1}:")
        print(f"  Input: {result['input'][:100]}...")
        print(f"  Expected: {result['expected_answer']}")
        print(f"  Teacher: {result['teacher_answer']} {'✓' if result['teacher_correct'] else '✗'}")
        print(f"  Student: {result['student_answer']} {'✓' if result['student_correct'] else '✗'}")



def process_base_model_results(
    outputs,
    chess_data: List[Dict],
    progress_interval: int = 100
) -> Tuple[List[Dict], int]:
    """
    Process base model inference outputs and calculate accuracy statistics.
    
    Args:
        outputs: vLLM generation outputs
        chess_data: Original chess dataset
        progress_interval: Print progress every N samples
    
    Returns:
        Tuple of (results list, base_correct count)
    """
    results_base = []
    base_correct = 0
    
    for idx, output in enumerate(outputs):
        item = chess_data[idx]
        generated_text = output.outputs[0].text.strip()
        
        # Extract the base model's answer
        base_answer = extract_move_answer(generated_text)
        
        # Get the expected (correct) answer and normalize it
        expected_answer = normalize_expected_answer(item['expected_output'])
        
        # Check correctness
        base_is_correct = (base_answer == expected_answer) if base_answer else False
        
        if base_is_correct:
            base_correct += 1
        
        results_base.append({
            'question_id': idx,
            'base_answer': base_answer,
            'base_correct': base_is_correct,
            'base_raw_output': generated_text
        })
        
        # Print progress
        if (idx + 1) % progress_interval == 0:
            print(f"Processed {idx + 1}/{len(chess_data)} samples...")
            print(f"  Current Base Model Accuracy: {base_correct}/{idx+1} ({base_correct/(idx+1)*100:.1f}%)")
    
    return results_base, base_correct


def save_base_model_results(
    results_base: List[Dict],
    base_correct: int,
    output_file: str = 'static/chess_evaluation_base_model.json'
) -> Dict[str, Any]:
    """
    Save base model evaluation results to JSON file.
    
    Args:
        results_base: List of base model results
        base_correct: Number of correct base model predictions
        output_file: Path to output JSON file
    
    Returns:
        Summary dictionary with base model statistics
    """
    import os
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    total_samples = len(results_base)
    base_accuracy = (base_correct / total_samples) * 100
    
    base_summary = {
        'total_samples': total_samples,
        'base_correct': base_correct,
        'base_accuracy': base_accuracy,
        'results': results_base
    }
    
    with open(output_file, 'w') as f:
        json.dump(base_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Base Model Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Total Samples: {total_samples}")
    print(f"Base Model Accuracy: {base_correct}/{total_samples} ({base_accuracy:.2f}%)")
    print(f"\nResults saved to: {output_file}")
    
    return base_summary


def merge_all_results(
    student_file: str = 'static/chess_evaluation_results.json',
    base_file: str = 'static/chess_evaluation_base_model.json',
    output_file: str = 'static/chess_evaluation_comprehensive.json'
) -> Dict[str, Any]:
    """
    Merge results from teacher, base, and student models into comprehensive summary.
    
    Args:
        student_file: Path to student model results JSON
        base_file: Path to base model results JSON
        output_file: Path to save comprehensive results
    
    Returns:
        Comprehensive summary dictionary
    """
    import os
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    print("Merging results from all three models...")
    
    # Load fine-tuned student results
    with open(student_file, 'r') as f:
        student_data = json.load(f)
    
    # Load base model results
    with open(base_file, 'r') as f:
        base_data = json.load(f)
    
    # Merge the results
    merged_results = []
    for i in range(len(student_data['results'])):
        student_result = student_data['results'][i]
        base_result = base_data['results'][i]
        
        merged_results.append({
            'question_id': i,
            'input': student_result['input'],
            'expected_answer': student_result['expected_answer'],
            'teacher_answer': student_result['teacher_answer'],
            'teacher_correct': student_result['teacher_correct'],
            'base_answer': base_result['base_answer'],
            'base_correct': base_result['base_correct'],
            'student_answer': student_result['student_answer'],
            'student_correct': student_result['student_correct'],
            'base_raw_output': base_result['base_raw_output'],
            'student_raw_output': student_result['student_raw_output']
        })
    
    # Create comprehensive summary
    comprehensive_summary = {
        'total_samples': student_data['total_samples'],
        'teacher_correct': student_data['teacher_correct'],
        'teacher_accuracy': student_data['teacher_accuracy'],
        'base_correct': base_data['base_correct'],
        'base_accuracy': base_data['base_accuracy'],
        'student_correct': student_data['student_correct'],
        'student_accuracy': student_data['student_accuracy'],
        'results': merged_results
    }
    
    # Save comprehensive results
    with open(output_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2)
    
    improvement = comprehensive_summary['student_accuracy'] - comprehensive_summary['base_accuracy']
    
    print(f"\n{'='*80}")
    print(f"Comprehensive Results Summary")
    print(f"{'='*80}")
    print(f"Total Samples: {comprehensive_summary['total_samples']}")
    print(f"\nTeacher (Qwen3-30B):     {comprehensive_summary['teacher_correct']}/{comprehensive_summary['total_samples']} ({comprehensive_summary['teacher_accuracy']:.2f}%)")
    print(f"Base (Qwen3-0.6B):       {comprehensive_summary['base_correct']}/{comprehensive_summary['total_samples']} ({comprehensive_summary['base_accuracy']:.2f}%)")
    print(f"Student (Fine-tuned):    {comprehensive_summary['student_correct']}/{comprehensive_summary['total_samples']} ({comprehensive_summary['student_accuracy']:.2f}%)")
    print(f"\nImprovement over base:   {improvement:+.2f} percentage points")
    print(f"\nResults saved to: {output_file}")
    
    return comprehensive_summary


def print_comprehensive_statistics(data: Dict[str, Any]):
    """
    Print detailed statistics for three-model comparison.
    
    Args:
        data: Comprehensive summary dictionary
    """
    total_samples = data['total_samples']
    teacher_correct = data['teacher_correct']
    teacher_accuracy = data['teacher_accuracy']
    base_correct = data['base_correct']
    base_accuracy = data['base_accuracy']
    student_correct = data['student_correct']
    student_accuracy = data['student_accuracy']
    
    print("="*80)
    print("CHESS MOVE EVALUATION RESULTS - THREE MODEL COMPARISON")
    print("="*80)
    print(f"\nDataset Size: {total_samples} chess positions")
    print(f"\n{'Model':<25} {'Correct':<15} {'Incorrect':<15} {'Accuracy':<15}")
    print("-"*80)
    print(f"{'Teacher (Qwen3-30B)':<25} {teacher_correct:<15} {total_samples - teacher_correct:<15} {teacher_accuracy:.2f}%")
    print(f"{'Base (Qwen3-0.6B)':<25} {base_correct:<15} {total_samples - base_correct:<15} {base_accuracy:.2f}%")
    print(f"{'Student (Fine-tuned)':<25} {student_correct:<15} {total_samples - student_correct:<15} {student_accuracy:.2f}%")
    print("-"*80)
    
    # Calculate improvement
    improvement = student_accuracy - base_accuracy
    print(f"\n🎯 Fine-tuning Improvement: {improvement:+.2f} percentage points")
    print(f"   ({student_correct - base_correct:+d} more correct answers)")
    
    # Calculate knowledge transfer
    knowledge_gap = teacher_accuracy - base_accuracy
    knowledge_captured = (student_accuracy - base_accuracy) / knowledge_gap * 100 if knowledge_gap > 0 else 0
    print(f"\n📚 Knowledge Transfer: {knowledge_captured:.1f}%")
    print(f"   (Captured {knowledge_captured:.1f}% of the gap between base and teacher)")
    
    print("="*80)


def create_comprehensive_visualization(
    data: Dict[str, Any],
    output_file: str = 'static/chess_evaluation_comprehensive.png'
):
    """
    Create comprehensive visualization comparing teacher, base, and student models.
    
    Args:
        data: Comprehensive summary dictionary
        output_file: Path to save the visualization
    """
    import os
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    total_samples = data['total_samples']
    teacher_correct = data['teacher_correct']
    teacher_accuracy = data['teacher_accuracy']
    base_correct = data['base_correct']
    base_accuracy = data['base_accuracy']
    student_correct = data['student_correct']
    student_accuracy = data['student_accuracy']
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Chess Move Evaluation: Teacher vs Base vs Fine-tuned Student', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Accuracy Comparison Bar Chart (larger, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    models = ['Teacher\n(Qwen3-30B)', 'Base\n(Qwen3-0.6B)', 'Student\n(Fine-tuned 0.6B)']
    accuracies = [teacher_accuracy, base_accuracy, student_accuracy]
    colors = ['#2E86AB', '#E63946', '#06A77D']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Guess (50%)')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Improvement Visualization
    ax2 = fig.add_subplot(gs[0, 2])
    improvement = student_accuracy - base_accuracy
    improvement_data = [base_accuracy, improvement]
    colors_imp = ['#E63946', '#06A77D']
    ax2.bar(['Base', 'Improvement'], improvement_data, color=colors_imp, alpha=0.85, 
            edgecolor='black', linewidth=2, bottom=[0, base_accuracy])
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fine-tuning Impact', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=base_accuracy, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axhline(y=student_accuracy, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.text(0.5, student_accuracy + 2, f'+{improvement:.1f}%', 
             ha='center', fontsize=11, fontweight='bold', color='green')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Correct vs Incorrect Counts
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(3)
    width = 0.35
    correct_counts = [teacher_correct, base_correct, student_correct]
    incorrect_counts = [total_samples - teacher_correct, total_samples - base_correct, 
                        total_samples - student_correct]
    bars1 = ax3.bar(x - width/2, correct_counts, width, label='Correct', 
                    color='#06A77D', alpha=0.85, edgecolor='black')
    bars2 = ax3.bar(x + width/2, incorrect_counts, width, label='Incorrect', 
                    color='#D62246', alpha=0.85, edgecolor='black')
    ax3.set_ylabel('Number of Questions', fontsize=11, fontweight='bold')
    ax3.set_title('Correct vs Incorrect', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Teacher', 'Base', 'Student'], fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Knowledge Transfer Gauge
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['Base\nModel', 'Knowledge\nCaptured', 'Remaining\nGap']
    values = [base_accuracy, student_accuracy - base_accuracy, 
              teacher_accuracy - student_accuracy]
    colors_gauge = ['#E63946', '#06A77D', '#FFA500']
    bars = ax4.bar(categories, values, color=colors_gauge, alpha=0.85, 
                   edgecolor='black', linewidth=2, bottom=[0, base_accuracy, student_accuracy])
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Knowledge Transfer Analysis', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylim(0, 100)
    ax4.axhline(y=teacher_accuracy, color='blue', linestyle='--', alpha=0.5, 
                linewidth=2, label='Teacher Level')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 5. Agreement Matrix
    ax5 = fig.add_subplot(gs[1, 2])
    all_correct = sum(1 for r in data['results'] if r['teacher_correct'] and r['base_correct'] and r['student_correct'])
    teacher_student = sum(1 for r in data['results'] if r['teacher_correct'] and r['student_correct'])
    teacher_base = sum(1 for r in data['results'] if r['teacher_correct'] and r['base_correct'])
    base_student = sum(1 for r in data['results'] if r['base_correct'] and r['student_correct'])
    
    agreement_labels = ['All 3\nCorrect', 'Teacher &\nStudent', 'Teacher &\nBase', 'Base &\nStudent']
    agreement_values = [all_correct, teacher_student - all_correct, 
                        teacher_base - all_correct, base_student - all_correct]
    colors_agree = ['#06A77D', '#2E86AB', '#9B59B6', '#F77F00']
    bars = ax5.bar(agreement_labels, agreement_values, color=colors_agree, 
                   alpha=0.85, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('Model Agreement', fontsize=12, fontweight='bold', pad=10)
    ax5.tick_params(axis='x', labelsize=8)
    for bar, val in zip(bars, agreement_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 6. Performance Summary Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    knowledge_gap = teacher_accuracy - base_accuracy
    knowledge_captured = (student_accuracy - base_accuracy) / knowledge_gap * 100 if knowledge_gap > 0 else 0
    
    table_data = [
        ['Metric', 'Teacher (30B)', 'Base (0.6B)', 'Student (0.6B)', 'Improvement'],
        ['Total Questions', str(total_samples), str(total_samples), str(total_samples), '-'],
        ['Correct Answers', str(teacher_correct), str(base_correct), str(student_correct), 
         f'+{student_correct - base_correct}'],
        ['Accuracy', f'{teacher_accuracy:.2f}%', f'{base_accuracy:.2f}%', 
         f'{student_accuracy:.2f}%', f'+{improvement:.2f}%'],
        ['Model Size', '30B params', '0.6B params', '0.6B params', 'Same as base'],
        ['Training', 'Teacher', 'Pretrained only', 'Distilled from teacher', '-'],
        ['Knowledge Captured', '-', '-', f'{knowledge_captured:.1f}%', '-']
    ]
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.18, 0.18, 0.18, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style the header row
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors and highlight improvement column
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            if j == 4 and i in [2, 3]:  # Highlight improvement values
                table[(i, j)].set_facecolor('#C8E6C9')
                table[(i, j)].set_text_props(weight='bold', color='#1B5E20')
            table[(i, j)].set_text_props(weight='bold' if j == 0 else 'normal')
    
    ax6.set_title('Comprehensive Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()


def print_comprehensive_sample_predictions(data: Dict[str, Any], num_samples: int = 5):
    """
    Print sample predictions for three-model comparison.
    
    Args:
        data: Comprehensive summary dictionary
        num_samples: Number of samples to display
    """
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 5 questions)")
    print("="*80)
    for i, result in enumerate(data['results'][:num_samples]):
        print(f"\nQuestion {i+1}:")
        print(f"  Input: {result['input'][:100]}...")
        print(f"  Expected: {result['expected_answer']}")
        print(f"  Teacher:  {result['teacher_answer']} {'✓' if result['teacher_correct'] else '✗'}")
        print(f"  Base:     {result['base_answer']} {'✓' if result['base_correct'] else '✗'}")
        print(f"  Student:  {result['student_answer']} {'✓' if result['student_correct'] else '✗'}")
