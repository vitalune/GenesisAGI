import torch
import random
from typing import Dict, List, Tuple

class ArithmeticVocab:
    """tokenizer for arithmetic expressions"""
    def __init__(self):
        # Tokens: digits 0-9, operators, special tokens
        self.tokens = ['<PAD>', '<START>', '<END>', '='] + \
                      [str(i) for i in range(10)] + \
                      ['+', '-', '*']
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        
        # Special token IDs
        self.pad_id = self.token_to_id['<PAD>']
        self.start_id = self.token_to_id['<START>']
        self.end_id = self.token_to_id['<END>']
    
    def encode(self, text: str) -> List[int]:
        """convert string to token IDs"""
        tokens = []
        for char in text:
            if char == ' ':
                continue
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                raise ValueError(f"Unknown token: {char}")
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """convert token IDs back to string"""
        return ''.join([self.id_to_token[id] for id in token_ids 
                       if id not in [self.pad_id, self.start_id, self.end_id]])

VOCAB = ArithmeticVocab()

def generate_arithmetic_problem(operation: str, max_num: int = 20) -> Tuple[str, int]:
    """
    generate single arithmetic problem
    
    args:
        operation: 'add', 'subtract', or 'multiply'
        max_num: maximum number to use in problem
    
    returns:
        (problem_string, answer)
    """
    a = random.randint(1, max_num)
    
    if operation == 'add':
        b = random.randint(1, max_num)
        answer = a + b
        op_symbol = '+'
        
    elif operation == 'subtract':
        # Ensure non-negative results
        b = random.randint(1, a)
        answer = a - b
        op_symbol = '-'
        
    elif operation == 'multiply':
        # Keep multiplication manageable
        b = random.randint(1, min(max_num, 10))
        answer = a * b
        op_symbol = '*'
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    problem = f"{a}{op_symbol}{b}"
    return problem, answer

def format_problem(problem: str, answer: int) -> Tuple[str, str]:
    """
    format problem for seq2seq training
    
    input:  "<START> 5 + 3 ="
    target: "8 <END>"
    """
    input_str = f"{problem}="
    target_str = f"{answer}"
    
    return input_str, target_str

def tokenize_problem(input_str: str, target_str: str, max_len: int = 10) -> Dict[str, torch.Tensor]:
    """
    convert problem strings to tensors
    
    returns:
        {
            'input': tensor of input token IDs,
            'target': tensor of target token IDs,
            'input_str': original input string,
            'target_str': original target string
        }
    """
    # encode
    input_ids = [VOCAB.start_id] + VOCAB.encode(input_str)
    target_ids = VOCAB.encode(target_str) + [VOCAB.end_id]
    
    # pad to max_len
    input_ids = input_ids + [VOCAB.pad_id] * (max_len - len(input_ids))
    target_ids = target_ids + [VOCAB.pad_id] * (max_len - len(target_ids))
    
    # truncate if too long
    input_ids = input_ids[:max_len]
    target_ids = target_ids[:max_len]
    
    return {
        'input': torch.tensor(input_ids, dtype=torch.long),
        'target': torch.tensor(target_ids, dtype=torch.long),
        'input_str': input_str,
        'target_str': target_str
    }

def generate_arithmetic_batch(operation: str, 
                              batch_size: int = 32, 
                              max_num: int = 20,
                              max_len: int = 10) -> Dict[str, torch.Tensor]:
    """
    generate batch of arithmetic problems
    
    args:
        operation: 'add', 'subtract', or 'multiply'
        batch_size: number of problems in batch
        max_num: maximum number in problems
        max_len: maximum sequence length
    
    returns:
        {
            'input': (batch_size, max_len) tensor,
            'target': (batch_size, max_len) tensor,
            'problems': list of problem strings,
            'answers': list of answer strings
        }
    """
    batch_inputs = []
    batch_targets = []
    problems = []
    answers = []
    
    for _ in range(batch_size):
        # generate problem
        problem, answer = generate_arithmetic_problem(operation, max_num)
        input_str, target_str = format_problem(problem, answer)
        
        # tokenize
        tokenized = tokenize_problem(input_str, target_str, max_len)
        
        batch_inputs.append(tokenized['input'])
        batch_targets.append(tokenized['target'])
        problems.append(input_str)
        answers.append(target_str)
    
    return {
        'input': torch.stack(batch_inputs),
        'target': torch.stack(batch_targets),
        'problems': problems,
        'answers': answers,
        'operation': operation
    }

def generate_mixed_batch(batch_size: int = 32,
                        operation_mix: Dict[str, float] = None,
                        max_num: int = 20,
                        max_len: int = 10) -> Dict[str, torch.Tensor]:
    """
    generate batch w/ mixed operations
    
    args:
        operation_mix: Dict like {'add': 0.5, 'multiply': 0.3, 'subtract': 0.2}
                      If None, defaults to equal mix
    """
    if operation_mix is None:
        operation_mix = {'add': 0.33, 'subtract': 0.33, 'multiply': 0.34}
    
    # determine num of each operation
    operations = []
    for op, proportion in operation_mix.items():
        count = int(batch_size * proportion)
        operations.extend([op] * count)
    
    # Fill remaining to reach batch_size
    while len(operations) < batch_size:
        operations.append(random.choice(list(operation_mix.keys())))
    
    random.shuffle(operations)
    
    # generate batch
    batch_inputs = []
    batch_targets = []
    problems = []
    answers = []
    
    for operation in operations:
        problem, answer = generate_arithmetic_problem(operation, max_num)
        input_str, target_str = format_problem(problem, answer)
        tokenized = tokenize_problem(input_str, target_str, max_len)
        
        batch_inputs.append(tokenized['input'])
        batch_targets.append(tokenized['target'])
        problems.append(input_str)
        answers.append(target_str)
    
    return {
        'input': torch.stack(batch_inputs),
        'target': torch.stack(batch_targets),
        'problems': problems,
        'answers': answers,
        'operations': operations
    }